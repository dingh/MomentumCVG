"""Build the canonical weekly straddle observation table (Sprint 005 D2).

Reads A1/A2 through an accepted input snapshot manifest, builds exactly one
long-straddle observation per A1 ``(ticker, entry_date)`` key, and publishes the
Parquet artifact plus its lineage receipt under a snapshot-keyed derived root.

The transform's economics are frozen module constants, not options: no flag on
this CLI changes an economic value, and none can overwrite a canonical artifact
that would differ from the one already published.

Usage
-----
    # Inspect what would be produced, writing nothing
    python scripts/build_straddle_observations.py \\
        --snapshot-root C:/MomentumCVG_env/snapshots/20260724T045049097520Z_40b16886 \\
        --dry-run

    # Publish to the default derived root
    python scripts/build_straddle_observations.py \\
        --snapshot-root C:/MomentumCVG_env/snapshots/20260724T045049097520Z_40b16886

    # Publish a new transform version somewhere else
    python scripts/build_straddle_observations.py \\
        --snapshot-root C:/MomentumCVG_env/snapshots/20260724T045049097520Z_40b16886 \\
        --output-root C:/MomentumCVG_env/derived_v2
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.features.straddle_observations import (
    DEFAULT_DERIVED_ROOT,
    TRANSFORM_CONFIG,
    StraddleObservationStructuralError,
    content_digest,
    derived_dir_for_snapshot,
    load_surface_frames,
    observation_coverage,
    publish_observations,
    resolve_surface_inputs,
    transform_surface_frames,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Build the canonical surface-derived straddle observation table",
    )
    parser.add_argument(
        "--snapshot-root",
        required=True,
        help="Accepted input snapshot root (A1/A2 are resolved from its manifest)",
    )
    parser.add_argument(
        "--output-root",
        default=str(DEFAULT_DERIVED_ROOT),
        help=(
            "Derived artifact root; the table is written to "
            "<output-root>/<snapshot_id>/ (default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve, build, and report coverage without writing any file",
    )
    return parser.parse_args()


def _log_coverage(coverage: dict) -> None:
    logger.info("Rows: %s (unique keys: %s)", coverage["row_count"], coverage["key_count"])
    logger.info("Status counts:")
    for status, count in coverage["status_counts"].items():
        logger.info("  %-24s %s", status, count)
    logger.info("Missing reasons:")
    for reason, count in coverage["missing_reason_counts"].items():
        logger.info("  %-32s %s", reason, count)
    logger.info("Populated fields:")
    for column, count in coverage["non_null_counts"].items():
        logger.info("  %-24s %s", column, count)


def main() -> int:
    """Resolve the snapshot, build observations, and publish or report."""
    args = parse_args()

    try:
        inputs = resolve_surface_inputs(args.snapshot_root)
        logger.info("Snapshot: %s (build %s)", inputs.snapshot_id, inputs.build_id)
        logger.info("Manifest: %s", inputs.manifest_path)
        logger.info("A1: %s", inputs.meta_path)
        logger.info("A2: %s", inputs.quotes_path)
        logger.info("Transform config: %s", json.dumps(TRANSFORM_CONFIG, sort_keys=True))

        meta_df, quotes_df = load_surface_frames(inputs)
        logger.info(
            "Loaded A1 rows=%s, A2 body rows=%s; A1 key digest matches the manifest",
            len(meta_df),
            len(quotes_df),
        )

        observations = transform_surface_frames(meta_df, quotes_df)
        _log_coverage(observation_coverage(observations))
        logger.info("Content digest: %s", content_digest(observations))

        destination = derived_dir_for_snapshot(inputs.snapshot_id, args.output_root)
        if args.dry_run:
            logger.info("Dry run: nothing written. Destination would be %s", destination)
            return 0

        result = publish_observations(
            observations,
            inputs=inputs,
            destination_dir=destination,
            meta_row_count=len(meta_df),
            quote_row_count=len(quotes_df),
        )
        if result.written:
            logger.info("Published %s", result.observations_path)
            logger.info("Published %s", result.lineage_path)
        else:
            logger.info(
                "Identical artifact already published at %s; nothing rewritten",
                result.observations_path,
            )
        return 0

    except StraddleObservationStructuralError as exc:
        logger.error("Aborted without writing an artifact: %s", exc)
        return 1
    except Exception as exc:  # noqa: BLE001 - CLI boundary
        logger.error("Error: %s", exc, exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
