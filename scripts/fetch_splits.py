"""
Fetch ORATS stock split history for a ticker universe.

Two universe modes
------------------
* **Full universe (legacy).** Read the ticker list from a parquet file
  (default ``C:/MomentumCVG_env/cache/all_tickers.parquet``, single ``ticker``
  column) and write combined split history to the broad default
  ``C:/MomentumCVG_env/cache/splits_hist.parquet``.

* **Scoped C5 universe.** Pass ``--ticker-universe`` pointing at the C4 liquid
  universe (CSV or parquet with a ``Ticker``/``ticker`` column, loaded via
  ``load_ticker_universe``) plus an explicit ``--out`` so scoped split history
  is written to a dedicated file such as
  ``C:/MomentumCVG_env/input/adjusted_liquid/splits_hist_liquid.parquet``.
  Scoped mode refuses to write to the broad default cache file.

Checkpoint/resume is built-in — safe to Ctrl-C and re-run. In scoped mode the
working checkpoint lives next to the output as ``<name>.checkpoint.parquet`` so
the validated scoped file is only written once the fetch passes all safety
checks.

Token
-----
Provide the ORATS API token via ``--token`` OR the ``ORATS_API_TOKEN``
environment variable. ``--token`` wins when both are present. The token is
never logged.

Usage:
    # Legacy full-universe fetch (token via env var)
    #   PowerShell:  $env:ORATS_API_TOKEN = "YOUR_TOKEN"
    python scripts/fetch_splits.py

    # Scoped C5 fetch for the C4 liquid universe
    python scripts/fetch_splits.py ^
        --ticker-universe C:/MomentumCVG_env/input/liquidity/liquid_tickers.csv ^
        --out             C:/MomentumCVG_env/input/adjusted_liquid/splits_hist_liquid.parquet ^
        --token           YOUR_TOKEN

    # Slow down requests (default 0.7 s between tickers)
    python scripts/fetch_splits.py --rate-limit 1.5
"""

import argparse
import logging
import os
import sys
import tempfile
from pathlib import Path

import pandas as pd

# ── resolve project root so src/ imports work regardless of cwd ───────────────
_SCRIPT_DIR   = Path(__file__).resolve().parent     # scripts/
_PROJECT_ROOT = _SCRIPT_DIR.parent                  # MomentumCVG/
sys.path.insert(0, str(_PROJECT_ROOT))

from src.data.corporate_actions import OratsCorporateActionsFetcher  # noqa: E402
from src.data.ticker_universe import load_ticker_universe  # noqa: E402

# ── defaults ──────────────────────────────────────────────────────────────────
DEFAULT_TICKERS_PATH = Path("C:/MomentumCVG_env/cache/all_tickers.parquet")
DEFAULT_OUTPUT_PATH  = Path("C:/MomentumCVG_env/cache/splits_hist.parquet")

REQUIRED_COLUMNS = ("ticker", "split_date", "divisor")

# ── logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fetch ORATS stock split history for a ticker universe.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--token",
        default=None,
        help=(
            "ORATS API token. Falls back to the ORATS_API_TOKEN environment "
            "variable when omitted. Never logged."
        ),
    )
    p.add_argument(
        "--tickers",
        type=Path,
        default=None,
        help=(
            "Legacy full-universe ticker source: path to a parquet file with a "
            "single 'ticker' column (default "
            f"{DEFAULT_TICKERS_PATH}). Mutually exclusive with "
            "--ticker-universe."
        ),
    )
    p.add_argument(
        "--ticker-universe",
        type=Path,
        default=None,
        dest="ticker_universe",
        metavar="PATH",
        help=(
            "Scoped C5 ticker source: CSV or parquet with a 'Ticker'/'ticker' "
            "column (e.g. the C4 liquid_tickers.csv), loaded via "
            "load_ticker_universe. When set, --out/--output MUST be given "
            "explicitly so scoped split history is never written to the broad "
            f"default {DEFAULT_OUTPUT_PATH}."
        ),
    )
    p.add_argument(
        "--output",
        "--out",
        type=Path,
        default=None,
        dest="output",
        metavar="PATH",
        help=(
            "Output parquet path for split history (also used as checkpoint). "
            f"Defaults to {DEFAULT_OUTPUT_PATH} in full-universe mode; REQUIRED "
            "explicitly in scoped (--ticker-universe) mode."
        ),
    )
    p.add_argument(
        "--rate-limit",
        type=float,
        default=0.7,
        metavar="SECONDS",
        help="Seconds to sleep between API requests.",
    )
    p.add_argument(
        "--max-retries",
        type=int,
        default=5,
        help="Max retry attempts per request on 429 / 5xx responses.",
    )
    return p.parse_args(argv)


def _resolve_token(cli_token: str | None) -> str | None:
    """Return the ORATS token: ``--token`` wins, else ``ORATS_API_TOKEN`` env."""
    return cli_token or os.environ.get("ORATS_API_TOKEN")


def _checkpoint_path_for(output_path: Path, scoped: bool) -> Path:
    """Working checkpoint path.

    In scoped mode the checkpoint is a sidecar file so the validated scoped
    output is only written after all safety checks pass. In legacy mode the
    output file doubles as the checkpoint (preserves historical resume
    behaviour that writes progress straight to ``splits_hist.parquet``).
    """
    if not scoped:
        return output_path
    return output_path.with_name(f"{output_path.stem}.checkpoint{output_path.suffix}")


def _write_parquet_atomically(df: pd.DataFrame, output_path: Path) -> None:
    """Write a parquet beside its destination, then atomically replace it."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=output_path.parent,
        prefix=f".{output_path.name}.",
        suffix=".tmp",
        delete=False,
    ) as tmp:
        temp_path = Path(tmp.name)

    try:
        df.to_parquet(temp_path, index=False)
        os.replace(temp_path, output_path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def _normalize_split_tickers(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with ``ticker`` stripped and upper-cased."""
    out = df.copy()
    out["ticker"] = out["ticker"].astype(str).str.strip().str.upper()
    return out


def _dedupe_split_rows(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Drop exact-duplicate rows; return (deduped, conflicts).

    ``conflicts`` holds rows that share a ``(ticker, split_date)`` key but carry
    a different ``divisor`` — i.e. genuine disagreements that survive exact
    de-duplication.
    """
    deduped = df.drop_duplicates(subset=list(REQUIRED_COLUMNS)).reset_index(drop=True)
    conflict_mask = deduped.duplicated(subset=["ticker", "split_date"], keep=False)
    conflicts = deduped[conflict_mask].sort_values(["ticker", "split_date"])
    return deduped, conflicts


def _fail(message: str) -> None:
    """Log and exit with the message as the exit status (fail fast)."""
    logger.error(message)
    sys.exit(message)


def _load_legacy_tickers(tickers_path: Path) -> list[str]:
    """Load the legacy full-universe ticker list from a parquet file."""
    if not tickers_path.exists():
        _fail(
            f"Tickers file not found: {tickers_path}\n"
            "Run build_ticker_universe.py first, or pass --ticker-universe."
        )
    tickers_df = pd.read_parquet(tickers_path)
    if "ticker" not in tickers_df.columns:
        _fail(f"Expected a 'ticker' column in {tickers_path}")
    tickers = tickers_df["ticker"].dropna().unique().tolist()
    logger.info("Loaded %d tickers from %s", len(tickers), tickers_path)
    return tickers


def _report_and_validate(
    splits_df: pd.DataFrame,
    fetch_universe: list[str],
    scoped: bool,
    output_path: Path,
) -> pd.DataFrame:
    """Normalize, dedupe, validate and report fetched split history.

    Returns the cleaned DataFrame ready to write. In scoped mode any unsafe
    condition (conflicting duplicates, invalid divisors, or tickers outside the
    input universe) triggers fail-fast before writing. In legacy mode the same
    conditions are reported as warnings to preserve historical behaviour.
    """
    universe_set = {t.strip().upper() for t in fetch_universe}

    # ── normalize ticker casing/whitespace ────────────────────────────────────
    if not splits_df.empty:
        splits_df = _normalize_split_tickers(splits_df)

    # ── deduplicate; detect (ticker, split_date) conflicts ────────────────────
    if not splits_df.empty:
        deduped, conflicts = _dedupe_split_rows(splits_df)
    else:
        deduped = splits_df
        conflicts = splits_df

    # ── divisor sanity ────────────────────────────────────────────────────────
    if not deduped.empty:
        divisor = pd.to_numeric(deduped["divisor"], errors="coerce")
        null_divisors = int(divisor.isna().sum())
        nonpositive_divisors = int((divisor <= 0).sum())
        min_divisor = float(divisor.min()) if divisor.notna().any() else None
        max_divisor = float(divisor.max()) if divisor.notna().any() else None
    else:
        null_divisors = nonpositive_divisors = 0
        min_divisor = max_divisor = None

    # ── universe membership ───────────────────────────────────────────────────
    output_tickers = set(deduped["ticker"]) if not deduped.empty else set()
    intersection = output_tickers & universe_set
    outside = sorted(output_tickers - universe_set)

    # ── split-date range ──────────────────────────────────────────────────────
    if not deduped.empty:
        split_dates = pd.to_datetime(deduped["split_date"], errors="coerce")
        min_split_date = split_dates.min()
        max_split_date = split_dates.max()
    else:
        min_split_date = max_split_date = None

    missing_columns = [c for c in REQUIRED_COLUMNS if c not in deduped.columns]

    # ── report ────────────────────────────────────────────────────────────────
    logger.info("─" * 60)
    logger.info("Split-history validation report")
    logger.info("  output path              : %s", output_path)
    logger.info("  row count                : %d", len(deduped))
    logger.info("  unique ticker count      : %d", len(output_tickers))
    logger.info("  columns present          : %s", list(deduped.columns))
    logger.info("  required columns present : %s",
                "yes" if not missing_columns else f"NO (missing {missing_columns})")
    logger.info("  split_date min           : %s", min_split_date)
    logger.info("  split_date max           : %s", max_split_date)
    logger.info("  input universe tickers   : %d", len(universe_set))
    logger.info("  output split tickers     : %d", len(output_tickers))
    logger.info("  intersection w/ universe : %d", len(intersection))
    logger.info("  outside-universe tickers : %d", len(outside))
    if outside:
        logger.info("    examples               : %s", outside[:10])
    logger.info("  null divisors            : %d", null_divisors)
    logger.info("  nonpositive divisors     : %d", nonpositive_divisors)
    logger.info("  divisor min / max        : %s / %s", min_divisor, max_divisor)
    logger.info("─" * 60)

    # ── enforcement ───────────────────────────────────────────────────────────
    if missing_columns:
        _fail(
            "Fetched split history is missing required columns "
            f"{missing_columns}; expected {list(REQUIRED_COLUMNS)}."
        )

    if not conflicts.empty:
        examples = conflicts.head(10).to_dict("records")
        if scoped:
            _fail(
                "Conflicting split rows detected: same (ticker, split_date) with "
                f"differing divisor. Refusing to write scoped split file. "
                f"Examples: {examples}"
            )
        logger.warning(
            "Conflicting split rows detected (same ticker/split_date, different "
            "divisor); keeping first per key. Examples: %s", examples,
        )
        deduped = deduped.drop_duplicates(subset=["ticker", "split_date"], keep="first")

    if null_divisors or nonpositive_divisors:
        message = (
            f"Invalid divisors detected: {null_divisors} null, "
            f"{nonpositive_divisors} nonpositive. Divisors must be positive and "
            "non-null for safe split adjustment."
        )
        if scoped:
            _fail(message + " Refusing to write scoped split file.")
        logger.warning(message + " (legacy mode: written unchanged)")

    if outside:
        message = (
            f"{len(outside)} output ticker(s) are outside the input universe: "
            f"{outside[:10]}."
        )
        if scoped:
            _fail(message + " Refusing to write scoped split file.")
        logger.warning(message + " (legacy mode: written unchanged)")

    return deduped


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    scoped = args.ticker_universe is not None

    # ── conflict guard: legacy --tickers vs scoped --ticker-universe ──────────
    if scoped and args.tickers is not None:
        _fail(
            "Cannot use --tickers and --ticker-universe together. --tickers is "
            "the legacy full-universe parquet source; --ticker-universe is the "
            "scoped C5 source. Choose exactly one."
        )

    # ── scoped safety guard: explicit output required ─────────────────────────
    if scoped and args.output is None:
        _fail(
            "Scoped/filtered split fetch (--ticker-universe) requires an "
            "explicit output path via --out/--output. Refusing to write scoped "
            f"split history to the broad default {DEFAULT_OUTPUT_PATH}."
        )

    output_path = args.output if args.output is not None else DEFAULT_OUTPUT_PATH

    # ── token resolution (never logged) ───────────────────────────────────────
    token = _resolve_token(args.token)
    if not token:
        _fail(
            "An ORATS API token is required. Pass --token or set the "
            "ORATS_API_TOKEN environment variable."
        )

    # ── resolve fetch universe ────────────────────────────────────────────────
    if scoped:
        fetch_universe = load_ticker_universe(args.ticker_universe)
        logger.info(
            "ticker_universe: loaded %d tickers from %s",
            len(fetch_universe), args.ticker_universe,
        )
    else:
        tickers_path = args.tickers if args.tickers is not None else DEFAULT_TICKERS_PATH
        fetch_universe = _load_legacy_tickers(tickers_path)

    logger.info("mode        : %s", "scoped (C5)" if scoped else "full universe")
    logger.info("output path : %s", output_path)

    # ── checkpoint / resume status ────────────────────────────────────────────
    checkpoint_path = _checkpoint_path_for(output_path, scoped)
    if checkpoint_path.exists():
        existing = pd.read_parquet(checkpoint_path)
        already_done = existing["ticker"].nunique() if "ticker" in existing.columns else 0
        remaining = len(fetch_universe) - already_done
        logger.info(
            "Checkpoint found (%s) — %d tickers already fetched, %d remaining.",
            checkpoint_path, already_done, remaining,
        )
    else:
        logger.info("No checkpoint found — starting fresh.")

    # ── fetch ─────────────────────────────────────────────────────────────────
    fetcher = OratsCorporateActionsFetcher(
        token=token,
        rate_limit=args.rate_limit,
        max_retries=args.max_retries,
    )

    splits_df = fetcher.fetch_all_splits(
        tickers=fetch_universe,
        checkpoint_path=checkpoint_path,   # saves after every ticker; resumes on re-run
    )

    # ── normalize, validate, report (fail-fast in scoped mode) ────────────────
    cleaned = _report_and_validate(splits_df, fetch_universe, scoped, output_path)

    # ── write validated output ────────────────────────────────────────────────
    _write_parquet_atomically(cleaned, output_path)
    logger.info("Done. Saved %d rows to %s", len(cleaned), output_path)


if __name__ == "__main__":
    main()
