# test_artifacts.py — artifact resolution, for both backends.
#
# The S3 tests run against a fake client rather than moto or a real bucket. The
# behaviour under test is entirely ours — pointer semantics, caching, partial
# download cleanup — and none of it depends on S3 being faithfully emulated.
# What it does depend on is boto3's method signatures, so the fake implements
# exactly the four calls artifacts.py makes and nothing else: a typo in an
# argument name still fails here.

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from adplatform.ml.artifacts import (
    METADATA_FILE,
    MODEL_FILE,
    POINTER_FILE,
    LocalArtifactStore,
    S3ArtifactStore,
)


def write_artifact(directory: Path, version: str, *, calibrator: bool = True) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / MODEL_FILE).write_text('{"learner": "fake"}')
    if calibrator:
        (directory / "calibrator.pkl").write_bytes(b"\x80\x04fake")
    (directory / METADATA_FILE).write_text(json.dumps({
        "model_version": version,
        "feature_version": "v3",
        "n_features": 17,
        "negative_keep_rate": 0.1,
        "promoted": True,
    }))


# ---------------------------------------------------------------------------
# Local
# ---------------------------------------------------------------------------

class TestLocalArtifactStore:

    def test_missing_directory_is_not_an_error(self, tmp_path):
        # Cold start before the first training run. The gateway serves baseline
        # CTR; it must not crash or log an exception.
        store = LocalArtifactStore(tmp_path / "nope")
        assert store.resolve(None) is None

    def test_resolves_a_present_artifact(self, tmp_path):
        write_artifact(tmp_path, "v1")
        resolved = LocalArtifactStore(tmp_path).resolve(None)
        assert resolved is not None
        assert resolved.model_version == "v1"
        assert resolved.directory == tmp_path

    def test_unchanged_artifact_resolves_to_none(self, tmp_path):
        write_artifact(tmp_path, "v1")
        store = LocalArtifactStore(tmp_path)
        first = store.resolve(None)
        assert store.resolve(first.token) is None

    def test_touching_metadata_triggers_a_reload(self, tmp_path):
        write_artifact(tmp_path, "v1")
        store = LocalArtifactStore(tmp_path)
        first = store.resolve(None)

        meta = tmp_path / METADATA_FILE
        os.utime(meta, (first_ts := 2_000_000_000, first_ts))

        assert store.resolve(first.token) is not None

    def test_unparseable_metadata_does_not_swap(self, tmp_path):
        # A metadata.json caught mid-write. Returning a Resolved here would make
        # ctr_model record the token and never retry the complete file.
        write_artifact(tmp_path, "v1")
        (tmp_path / METADATA_FILE).write_text('{"model_version": "v1"')
        assert LocalArtifactStore(tmp_path).resolve(None) is None


# ---------------------------------------------------------------------------
# S3
# ---------------------------------------------------------------------------

class FakeS3:
    """
    Implements only get_object, download_file, put_object, head_object.
    `objects` maps key -> bytes.
    """

    def __init__(self, objects: dict[str, bytes] | None = None):
        self.objects = dict(objects or {})
        self.downloads: list[str] = []
        self.get_calls: list[str] = []
        self.fail_keys: set[str] = set()

    def get_object(self, Bucket: str, Key: str):  # noqa: N803 — boto3's casing
        self.get_calls.append(Key)
        if Key not in self.objects:
            raise KeyError(f"NoSuchKey: {Key}")
        return {"Body": _Body(self.objects[Key])}

    def download_file(self, Bucket: str, Key: str, Filename: str):  # noqa: N803
        if Key in self.fail_keys:
            raise RuntimeError(f"simulated failure downloading {Key}")
        if Key not in self.objects:
            raise KeyError(f"NoSuchKey: {Key}")
        self.downloads.append(Key)
        Path(Filename).write_bytes(self.objects[Key])

    def put_object(self, Bucket: str, Key: str, Body: bytes, **kwargs):  # noqa: N803
        self.objects[Key] = Body

    def head_object(self, Bucket: str, Key: str):  # noqa: N803
        if Key not in self.objects:
            raise KeyError(f"NoSuchKey: {Key}")
        return {"ContentLength": len(self.objects[Key])}


class _Body:
    def __init__(self, data: bytes):
        self._data = data

    def read(self) -> bytes:
        return self._data


def s3_objects(version: str, *, with_calibrator: bool = True) -> dict[str, bytes]:
    meta = json.dumps({
        "model_version": version,
        "feature_version": "v3",
        "n_features": 17,
        "promoted": True,
    }).encode()
    objects = {
        f"models/{POINTER_FILE}": json.dumps({"model_version": version}).encode(),
        f"models/{version}/{MODEL_FILE}": b'{"learner": "fake"}',
        f"models/{version}/{METADATA_FILE}": meta,
    }
    if with_calibrator:
        objects[f"models/{version}/calibrator.pkl"] = b"\x80\x04fake"
    return objects


def make_store(tmp_path, client) -> S3ArtifactStore:
    return S3ArtifactStore(
        bucket="test-bucket", prefix="models",
        cache_dir=tmp_path / "cache", region="us-east-1", client=client,
    )


class TestS3ArtifactStore:

    def test_no_pointer_is_not_an_error(self, tmp_path):
        # Bucket exists, nothing published yet.
        store = make_store(tmp_path, FakeS3({}))
        assert store.resolve(None) is None

    def test_downloads_the_version_the_pointer_names(self, tmp_path):
        client = FakeS3(s3_objects("v20260115_031200"))
        resolved = make_store(tmp_path, client).resolve(None)

        assert resolved is not None
        assert resolved.model_version == "v20260115_031200"
        assert (resolved.directory / MODEL_FILE).exists()
        assert (resolved.directory / METADATA_FILE).exists()
        assert (resolved.directory / "calibrator.pkl").exists()

    def test_token_is_the_version_not_an_mtime(self, tmp_path):
        # This is what makes N replicas agree: they compare the same string,
        # not their own local file timestamps.
        client = FakeS3(s3_objects("v1"))
        resolved = make_store(tmp_path, client).resolve(None)
        assert resolved.token == "v1"

    def test_unchanged_pointer_downloads_nothing(self, tmp_path):
        client = FakeS3(s3_objects("v1"))
        store = make_store(tmp_path, client)
        store.resolve(None)
        before = len(client.downloads)

        assert store.resolve("v1") is None
        assert len(client.downloads) == before

    def test_pointer_flip_pulls_the_new_version(self, tmp_path):
        client = FakeS3(s3_objects("v1"))
        store = make_store(tmp_path, client)
        store.resolve(None)

        client.objects.update(s3_objects("v2"))
        resolved = store.resolve("v1")

        assert resolved is not None
        assert resolved.model_version == "v2"
        assert resolved.directory.name == "v2"

    def test_rollback_to_a_cached_version_skips_the_download(self, tmp_path):
        # A task that has already seen v1 and gets pointed back at it should not
        # re-fetch. This is also the restart-during-deploy path.
        client = FakeS3(s3_objects("v1"))
        store = make_store(tmp_path, client)
        store.resolve(None)
        client.objects.update(s3_objects("v2"))
        store.resolve("v1")

        client.objects[f"models/{POINTER_FILE}"] = json.dumps({"model_version": "v1"}).encode()
        downloads_before = len(client.downloads)
        resolved = store.resolve("v2")

        assert resolved.model_version == "v1"
        assert len(client.downloads) == downloads_before

    def test_missing_calibrator_still_resolves(self, tmp_path):
        client = FakeS3(s3_objects("v1", with_calibrator=False))
        resolved = make_store(tmp_path, client).resolve(None)
        assert resolved is not None
        assert not (resolved.directory / "calibrator.pkl").exists()

    def test_failed_download_leaves_no_partial_version_dir(self, tmp_path):
        # The bug this guards against: a half-downloaded directory that a later
        # boot mistakes for a complete artifact and loads a truncated booster.
        client = FakeS3(s3_objects("v1"))
        client.fail_keys.add(f"models/v1/{METADATA_FILE}")
        store = make_store(tmp_path, client)

        assert store.resolve(None) is None
        cache = tmp_path / "cache"
        assert not (cache / "v1").exists()
        # And no leftover staging directories either.
        assert list(cache.iterdir()) == []

    def test_pointer_without_model_version_is_rejected(self, tmp_path):
        client = FakeS3({f"models/{POINTER_FILE}": json.dumps({"oops": True}).encode()})
        assert make_store(tmp_path, client).resolve(None) is None

    def test_malformed_pointer_json_is_rejected(self, tmp_path):
        client = FakeS3({f"models/{POINTER_FILE}": b"not json at all"})
        assert make_store(tmp_path, client).resolve(None) is None

    def test_empty_bucket_setting_is_a_configuration_error(self, tmp_path):
        with pytest.raises(ValueError, match="MODEL_S3_BUCKET"):
            S3ArtifactStore(bucket="", cache_dir=tmp_path, client=FakeS3({}))

    def test_prefix_is_normalised(self, tmp_path):
        for prefix in ("models", "/models", "models/", "/models/"):
            store = S3ArtifactStore(bucket="b", prefix=prefix,
                                    cache_dir=tmp_path, client=FakeS3({}))
            assert store._key(POINTER_FILE) == "models/current.json"

    def test_empty_prefix_puts_the_pointer_at_the_root(self, tmp_path):
        store = S3ArtifactStore(bucket="b", prefix="",
                                cache_dir=tmp_path, client=FakeS3({}))
        assert store._key(POINTER_FILE) == "current.json"


class TestConvergence:
    """
    The property the whole design exists for: replicas that start at different
    times, with different caches, converge on the version the pointer names.
    """

    def test_three_replicas_converge_on_the_pointer(self, tmp_path):
        client = FakeS3(s3_objects("v1"))
        replicas = [make_store(tmp_path / f"r{i}", client) for i in range(3)]

        tokens = [r.resolve(None).token for r in replicas]
        assert tokens == ["v1", "v1", "v1"]

        # Promotion: one pointer PUT.
        client.objects.update(s3_objects("v2"))
        client.objects[f"models/{POINTER_FILE}"] = json.dumps({"model_version": "v2"}).encode()

        tokens = [r.resolve(t).token for r, t in zip(replicas, tokens)]
        assert tokens == ["v2", "v2", "v2"]

    def test_a_replica_joining_late_lands_on_the_current_version(self, tmp_path):
        client = FakeS3(s3_objects("v1"))
        early = make_store(tmp_path / "early", client)
        early.resolve(None)

        client.objects.update(s3_objects("v2"))
        client.objects[f"models/{POINTER_FILE}"] = json.dumps({"model_version": "v2"}).encode()

        # A task that scales up after the promotion never sees v1 at all.
        late = make_store(tmp_path / "late", client)
        assert late.resolve(None).model_version == "v2"
        assert not (tmp_path / "late" / "cache" / "v1").exists()
