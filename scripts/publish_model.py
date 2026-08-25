#!/usr/bin/env python3
"""
publish_model.py — upload a trained artifact to S3 and make it live.

    python -m scripts.publish_model models/v20260115_031200
    python -m scripts.publish_model models/v20260115_031200 --dry-run
    python -m scripts.publish_model --rollback v20260114_031200
    python -m scripts.publish_model --show

This is the second half of `train_ctr.py`. Training writes a versioned directory
and copies it to `models/current` locally; publishing puts that directory in S3
under an immutable prefix and then, only if every file landed, writes the
pointer.

ORDER MATTERS AND IT IS NOT NEGOTIABLE. Files first, pointer last. Reversed, a
replica that polls between the two reads a pointer naming a version whose
model.json does not exist yet, fails the download, and logs an exception every
refresh tick until the upload finishes. Nothing catches fire — the gateway keeps
serving the previous model — but you get a wall of alarming logs for a
successful promotion, which is how people learn to ignore logs.

The pointer PUT is the commit. Everything before it is invisible.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from adplatform.ml.artifacts import (  # noqa: E402
    CALIBRATOR_FILE,
    METADATA_FILE,
    MODEL_FILE,
    POINTER_FILE,
)
from adplatform.settings import settings  # noqa: E402

REQUIRED = (MODEL_FILE, METADATA_FILE)
OPTIONAL = (CALIBRATOR_FILE,)


def _client(region: str):
    import boto3

    return boto3.client("s3", region_name=region)


def _key(prefix: str, *parts: str) -> str:
    base = prefix.strip("/")
    joined = "/".join(parts)
    return f"{base}/{joined}" if base else joined


def _read_metadata(directory: Path) -> dict:
    meta_path = directory / METADATA_FILE
    if not meta_path.exists():
        raise SystemExit(f"{meta_path} not found — is that a version directory?")
    return json.loads(meta_path.read_text())


def publish(directory: Path, bucket: str, prefix: str, region: str,
            dry_run: bool, force: bool) -> str:
    meta = _read_metadata(directory)
    version = str(meta.get("model_version", "")).strip()
    if not version:
        raise SystemExit("metadata.json has no model_version")

    # The promotion gates already ran in train_ctr.py. Publishing an artifact
    # that failed them is a deliberate act, so it needs a deliberate flag.
    if not meta.get("promoted") and not force:
        failures = meta.get("gate_failures") or ["(not recorded)"]
        raise SystemExit(
            f"{version} did not pass promotion gates:\n  - "
            + "\n  - ".join(failures)
            + "\nPublish anyway with --force if you mean it."
        )

    missing = [name for name in REQUIRED if not (directory / name).exists()]
    if missing:
        raise SystemExit(f"{directory} is missing {', '.join(missing)}")

    uploads = [n for n in REQUIRED] + [n for n in OPTIONAL if (directory / n).exists()]

    print(f"publishing {version}")
    print(f"  from : {directory}")
    print(f"  to   : s3://{bucket}/{_key(prefix, version)}/")
    print(f"  files: {', '.join(uploads)}")

    if dry_run:
        print("\n--dry-run set; nothing uploaded, pointer untouched")
        return version

    s3 = _client(region)

    # 1. Files, into an immutable per-version prefix.
    for name in uploads:
        s3.upload_file(str(directory / name), bucket, _key(prefix, version, name))
        print(f"  uploaded {name}")

    # 2. Pointer. This is the moment the model goes live everywhere.
    pointer = {
        "model_version": version,
        "published_at": datetime.now(timezone.utc).isoformat(),
        "feature_version": meta.get("feature_version"),
        "metrics": (meta.get("metrics") or {}).get("model"),
    }
    s3.put_object(
        Bucket=bucket,
        Key=_key(prefix, POINTER_FILE),
        Body=json.dumps(pointer, indent=2).encode(),
        ContentType="application/json",
        CacheControl="no-cache",
    )
    print(f"\n  pointer -> {version}")
    print(f"  replicas converge within MODEL_REFRESH_SECONDS "
          f"({settings.model_refresh_seconds}s)")
    return version


def rollback(version: str, bucket: str, prefix: str, region: str) -> None:
    """
    Point at an already-uploaded version. Verifies the files are there first,
    because the whole value of a rollback is that it works on the first try
    while something is on fire.
    """
    s3 = _client(region)
    for name in REQUIRED:
        try:
            s3.head_object(Bucket=bucket, Key=_key(prefix, version, name))
        except Exception:
            raise SystemExit(
                f"s3://{bucket}/{_key(prefix, version, name)} does not exist — "
                "refusing to point at an incomplete version"
            )

    pointer = {
        "model_version": version,
        "published_at": datetime.now(timezone.utc).isoformat(),
        "rollback": True,
    }
    s3.put_object(
        Bucket=bucket,
        Key=_key(prefix, POINTER_FILE),
        Body=json.dumps(pointer, indent=2).encode(),
        ContentType="application/json",
        CacheControl="no-cache",
    )
    print(f"rolled back to {version}")


def show(bucket: str, prefix: str, region: str) -> None:
    s3 = _client(region)
    try:
        body = s3.get_object(Bucket=bucket, Key=_key(prefix, POINTER_FILE))["Body"].read()
    except Exception as e:
        raise SystemExit(f"no pointer at s3://{bucket}/{_key(prefix, POINTER_FILE)}: {e}")
    print(json.dumps(json.loads(body), indent=2))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("directory", nargs="?", type=Path,
                    help="a version directory, e.g. models/v20260115_031200")
    ap.add_argument("--bucket", default=settings.model_s3_bucket)
    ap.add_argument("--prefix", default=settings.model_s3_prefix)
    ap.add_argument("--region", default=settings.aws_region)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true",
                    help="publish even if the artifact failed its promotion gates")
    ap.add_argument("--rollback", metavar="VERSION",
                    help="point at an already-uploaded version")
    ap.add_argument("--show", action="store_true", help="print the live pointer")
    args = ap.parse_args()

    if not args.bucket:
        raise SystemExit("no bucket — set MODEL_S3_BUCKET or pass --bucket")

    if args.show:
        show(args.bucket, args.prefix, args.region)
    elif args.rollback:
        rollback(args.rollback, args.bucket, args.prefix, args.region)
    elif args.directory:
        publish(args.directory, args.bucket, args.prefix, args.region,
                args.dry_run, args.force)
    else:
        ap.error("give a version directory, or --rollback VERSION, or --show")


if __name__ == "__main__":
    main()
