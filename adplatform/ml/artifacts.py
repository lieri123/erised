# artifacts.py — where a model artifact comes from.
#
# WHY THIS EXISTS
#
# With one process, `models/current` is a directory on disk and reloading means
# stat'ing metadata.json. With N Fargate tasks there is no shared disk, and the
# naive fixes are all wrong in the same way:
#
#   - EFS mounted into every task: works, costs money, and turns a read-mostly
#     artifact fetch into a network filesystem on the serving path.
#   - Bake the model into the image: every retrain is a redeploy, and the
#     nightly training job cannot ship its own output.
#   - Each task polls S3 for `model.json` directly: a task can observe the new
#     model.json alongside the old metadata.json, load a booster whose
#     feature_version it has not verified, and score garbage. There is no
#     moment at which "the artifact" changes atomically.
#
# The pointer solves the last one. Versioned artifacts are written to immutable
# prefixes that are never mutated after upload:
#
#     s3://bucket/models/v20260115_031200/{model.json,calibrator.pkl,metadata.json}
#
# and a single small object names the live one:
#
#     s3://bucket/models/current.json  ->  {"model_version": "v20260115_031200"}
#
# Promotion is one PUT of that pointer. S3 gives read-after-write consistency on
# it, so the swap is atomic from every replica's point of view: a task either
# sees the old version or the new one, never a mixture. Replicas converge within
# one MODEL_REFRESH_SECONDS window without coordinating with each other.
#
# Rollback is the same PUT with an older version string, which is worth more
# than it sounds — the promotion gates in train_ctr.py stop a bad model from
# shipping, but nothing stops a model that passes gates and behaves badly on
# real traffic.

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

# The three files that make up an artifact. calibrator.pkl is optional — an
# uncalibrated model still serves, it just serves worse.
MODEL_FILE = "model.json"
CALIBRATOR_FILE = "calibrator.pkl"
METADATA_FILE = "metadata.json"
POINTER_FILE = "current.json"

REQUIRED_FILES = (MODEL_FILE, METADATA_FILE)
OPTIONAL_FILES = (CALIBRATOR_FILE,)


@dataclass(frozen=True)
class Resolved:
    """
    A local directory holding a complete artifact, plus the token that identifies
    it. `token` is compared against the last-loaded token to decide whether any
    work is needed — it is an mtime for local, a model_version for S3.
    """
    directory: Path
    token: str
    model_version: str


class ArtifactStore(Protocol):
    def resolve(self, current_token: Optional[str]) -> Optional[Resolved]:
        """
        Return the live artifact if it differs from `current_token`, else None.
        Must not raise on "nothing published yet" — that is a normal cold start,
        and the gateway serves baseline CTR until a model exists.
        """
        ...

    def describe(self) -> str:
        ...


# ---------------------------------------------------------------------------
# Local
# ---------------------------------------------------------------------------

class LocalArtifactStore:
    """
    The original behaviour: a directory, usually a bind mount, whose
    metadata.json mtime is the version token. Unchanged so that dev, tests, and
    docker-compose keep working exactly as before.
    """

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
            # A half-written metadata.json. Do not swap on it; the next refresh
            # tick will pick up the complete file.
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
    Pointer-based distribution. Downloads into a per-version directory under the
    cache dir and hands back that directory; the caller loads from local disk as
    it always has.

    Downloads go to a temp directory and are renamed into place, so a task that
    dies mid-download leaves no partial version dir that a later boot would
    mistake for a complete one.
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

    # -- boto3 is imported lazily so the local path never needs it installed --

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
            # NoSuchKey on a fresh bucket is expected. Anything else is worth a
            # log line, but neither is fatal — we keep serving what we have.
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

        # A version that is on the same value as an already-cached dir does not
        # need re-downloading — this is the common case on a task restart during
        # a deploy, and it takes the cold start from seconds to nothing.
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

            # os.replace on a directory is atomic when the target does not exist
            # and both live on the same filesystem, which is guaranteed here
            # because staging was created inside cache_dir.
            try:
                os.replace(staging, target)
            except OSError:
                # Another thread or an earlier partial run got there first.
                # Whatever is at `target` is complete or it is not; check.
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
    """Pick a store from configuration. Called once, at CtrModel construction."""
    backend = settings.model_artifact_backend.lower()
    if backend == "s3":
        return S3ArtifactStore()
    if backend == "local":
        return LocalArtifactStore()
    raise ValueError(
        f"MODEL_ARTIFACT_BACKEND must be 'local' or 's3', got {backend!r}"
    )
