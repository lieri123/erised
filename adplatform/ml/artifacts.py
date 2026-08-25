# artifacts.py — where a model artifact comes from.
#
# One process can treat models/current as a directory and reload on mtime. N
# Fargate tasks share no disk, so each needs to fetch, and fetching files
# individually has no atomic point: a task can pull the new model.json next to
# the old metadata.json and score against unverified features.
#
# So: versioned prefixes, never mutated after upload,
#
#     s3://bucket/models/v20260115_031200/{model,calibrator,metadata}
#
# and one small object naming the live one,
#
#     s3://bucket/models/current.json -> {"model_version": "v20260115_031200"}
#
# Promotion is a single PUT of that pointer, and S3 is read-after-write
# consistent on it, so every replica sees either the old version or the new one.
# They converge within one MODEL_REFRESH_SECONDS window without coordinating.
# Rollback is the same PUT with an older version — worth having, since the
# promotion gates catch a bad model but not one that gates fine and misbehaves
# on real traffic.

from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Protocol

from ..settings import settings

log = logging.getLogger(__name__)

# calibrator.pkl is optional; an uncalibrated model still serves.
MODEL_FILE = "model.json"
CALIBRATOR_FILE = "calibrator.pkl"
METADATA_FILE = "metadata.json"
POINTER_FILE = "current.json"

REQUIRED_FILES = (MODEL_FILE, METADATA_FILE)
OPTIONAL_FILES = (CALIBRATOR_FILE,)


@dataclass(frozen=True)
class Resolved:
    """
    A local directory holding a complete artifact. `token` identifies it for
    change detection: an mtime for local, a model_version for S3.
    """
    directory: Path
    token: str
    model_version: str


class ArtifactStore(Protocol):
    def resolve(self, current_token: Optional[str]) -> Optional[Resolved]:
        """
        Return the live artifact if it differs from `current_token`, else None.
        Returns None rather than raising when nothing is published yet; that is
        a normal cold start and the gateway serves baseline CTR until then.
        """
        ...

    def describe(self) -> str:
        ...


# ---------------------------------------------------------------------------
# Local
# ---------------------------------------------------------------------------

class LocalArtifactStore:
    """A directory, usually a bind mount, keyed on metadata.json's mtime."""

    def __init__(self, directory: str | Path | None = None):
        self.directory = Path(directory or settings.model_dir)

    def resolve(self, current_token: Optional[str]) -> Optional[Resolved]:
        meta_path = self.directory / METADATA_FILE
        if not meta_path.exists():
            return None

        token = str(meta_path.stat().st_mtime)
        if token == current_token:
            return None

        try:
            version = str(json.loads(meta_path.read_text()).get("model_version", "unknown"))
        except Exception:
            # Half-written file; the next refresh tick gets the complete one.
            log.warning("metadata.json at %s is not readable JSON yet", self.directory)
            return None

        return Resolved(directory=self.directory, token=token, model_version=version)

    def describe(self) -> str:
        return f"local:{self.directory}"


# ---------------------------------------------------------------------------
# S3
# ---------------------------------------------------------------------------

class S3ArtifactStore:
    """
    Downloads the pointed-at version into a per-version directory under the
    cache dir and returns that directory, so the caller still loads from disk.

    Downloads land in a temp dir and are renamed into place: a task that dies
    mid-download leaves no partial directory for a later boot to mistake for a
    complete one.
    """

    def __init__(
        self,
        bucket: str | None = None,
        prefix: str | None = None,
        cache_dir: str | Path | None = None,
        region: str | None = None,
        client=None,
    ):
        self.bucket = bucket or settings.model_s3_bucket
        if not self.bucket:
            raise ValueError(
                "MODEL_ARTIFACT_BACKEND=s3 but MODEL_S3_BUCKET is empty"
            )
        raw_prefix = prefix if prefix is not None else settings.model_s3_prefix
        self.prefix = raw_prefix.strip("/") + "/" if raw_prefix.strip("/") else ""
        self.cache_dir = Path(cache_dir or settings.model_cache_dir)
        self.region = region or settings.aws_region
        self._client = client

    # boto3 is imported lazily so the local path does not need it installed.

    @property
    def client(self):
        if self._client is None:
            import boto3

            self._client = boto3.client("s3", region_name=self.region)
        return self._client

    def _key(self, *parts: str) -> str:
        return self.prefix + "/".join(parts)

    def read_pointer(self) -> Optional[dict]:
        try:
            body = self.client.get_object(
                Bucket=self.bucket, Key=self._key(POINTER_FILE)
            )["Body"].read()
        except Exception as e:
            # NoSuchKey on a fresh bucket is expected; other errors are not, but
            # neither is fatal, since we keep serving what is already loaded.
            if type(e).__name__ in {"NoSuchKey", "ClientError"}:
                log.debug("no model pointer at s3://%s/%s (%s)",
                          self.bucket, self._key(POINTER_FILE), e)
            else:
                log.warning("could not read model pointer: %s", e)
            return None

        try:
            return json.loads(body)
        except Exception:
            log.error("model pointer at s3://%s/%s is not valid JSON",
                      self.bucket, self._key(POINTER_FILE))
            return None

    def resolve(self, current_token: Optional[str]) -> Optional[Resolved]:
        pointer = self.read_pointer()
        if not pointer:
            return None

        version = str(pointer.get("model_version", "")).strip()
        if not version:
            log.error("model pointer has no model_version field")
            return None
        if version == current_token:
            return None

        # Already cached: skip the download. Common on a task restart mid-deploy
        # and on rollback.
        target = self.cache_dir / version
        if self._is_complete(target):
            log.info("model %s already cached at %s", version, target)
            return Resolved(directory=target, token=version, model_version=version)

        if not self._download(version, target):
            return None

        return Resolved(directory=target, token=version, model_version=version)

    @staticmethod
    def _is_complete(directory: Path) -> bool:
        return all((directory / name).exists() for name in REQUIRED_FILES)

    def _download(self, version: str, target: Path) -> bool:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=f".{version}.", dir=self.cache_dir))

        try:
            for name in REQUIRED_FILES:
                self.client.download_file(
                    self.bucket, self._key(version, name), str(staging / name)
                )
            for name in OPTIONAL_FILES:
                try:
                    self.client.download_file(
                        self.bucket, self._key(version, name), str(staging / name)
                    )
                except Exception:
                    log.info("no %s for model %s — serving uncalibrated", name, version)

            # Atomic when the target does not exist and both are on the same
            # filesystem, which staging being inside cache_dir guarantees.
            try:
                os.replace(staging, target)
            except OSError:
                # Something got there first. Check what it left behind.
                shutil.rmtree(staging, ignore_errors=True)
                if not self._is_complete(target):
                    raise

            log.info("downloaded model %s from s3://%s/%s to %s",
                     version, self.bucket, self._key(version), target)
            return True

        except Exception:
            shutil.rmtree(staging, ignore_errors=True)
            log.exception("failed to download model %s; keeping previous model", version)
            return False

    def describe(self) -> str:
        return f"s3://{self.bucket}/{self.prefix}"


# ---------------------------------------------------------------------------

def build_store() -> ArtifactStore:
    """Called once, at CtrModel construction."""
    backend = settings.model_artifact_backend.lower()
    if backend == "s3":
        return S3ArtifactStore()
    if backend == "local":
        return LocalArtifactStore()
    raise ValueError(
        f"MODEL_ARTIFACT_BACKEND must be 'local' or 's3', got {backend!r}"
    )
