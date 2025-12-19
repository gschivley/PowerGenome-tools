"""Build the in-browser ATB options index.

The web app cannot read Parquet directly in PyScript/Pyodide, so we ship a
precomputed JSON index under `web/data/atb_options.json`.

This script extracts the unique combinations needed for the Settings tab:
- data_year
- technology
- tech_detail
- cost_case

Usage:
    # Use the default Parquet hosted on GitHub (PowerGenome-data, via Git LFS media URL)
    python web/build_atb_options.py

    # Or provide a local parquet
    python web/build_atb_options.py \
        --parquet /path/to/technology_costs_atb.parquet \
        --out web/data/atb_options.json

    # Or download from a specific GitHub repo/path/ref
    python web/build_atb_options.py \
        --github gschivley/PowerGenome-data:data/technology_costs_atb.parquet@main \
        --out web/data/atb_options.json

Notes:
- Requires pandas + a parquet engine (pyarrow recommended).
"""

from __future__ import annotations

import argparse
import json
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd

DEFAULT_GITHUB_SPEC = (
    "gschivley/PowerGenome-data:data/technology_costs_atb.parquet@main"
)


def _parse_github_spec(spec: str) -> tuple[str, str, str, str]:
    """Parse OWNER/REPO:PATH@REF -> (owner, repo, path, ref)."""
    spec = (spec or "").strip()
    if not spec:
        raise ValueError("Empty GitHub spec")

    if ":" not in spec:
        raise ValueError(
            "GitHub spec must look like 'OWNER/REPO:PATH@REF' (missing ':')"
        )

    repo_part, path_part = spec.split(":", 1)
    if "/" not in repo_part:
        raise ValueError(
            "GitHub spec must look like 'OWNER/REPO:PATH@REF' (missing 'OWNER/REPO')"
        )
    owner, repo = repo_part.split("/", 1)
    owner = owner.strip()
    repo = repo.strip()

    if "@" in path_part:
        path, ref = path_part.rsplit("@", 1)
    else:
        path, ref = path_part, "main"

    path = path.strip().lstrip("/")
    ref = ref.strip() or "main"

    if not owner or not repo or not path:
        raise ValueError("Invalid GitHub spec; expected OWNER/REPO:PATH@REF")

    return owner, repo, path, ref


def _github_media_url(owner: str, repo: str, ref: str, path: str) -> str:
    # media.githubusercontent.com serves Git LFS objects correctly.
    return f"https://media.githubusercontent.com/media/{owner}/{repo}/{ref}/{path}"


def _read_first_text_line(path: Path) -> str:
    try:
        with path.open("rb") as f:
            chunk = f.read(200)
        try:
            return chunk.decode("utf-8", errors="ignore").splitlines()[0].strip()
        except Exception:
            return ""
    except Exception:
        return ""


def download_to_temp(url: str) -> Path:
    """Download URL to a temporary file and return the path."""
    url = str(url).strip()
    if not url:
        raise ValueError("Empty URL")
    if not url.startswith("https://"):
        raise ValueError(f"Only 'https://' URLs are allowed (got {url!r})")

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".parquet")
    tmp_path = Path(tmp.name)

    try:
        with urllib.request.urlopen(url) as resp:
            while True:
                chunk = resp.read(1024 * 1024)
                if not chunk:
                    break
                tmp.write(chunk)
    except urllib.error.HTTPError as exc:
        tmp.close()
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass
        raise RuntimeError(f"Download failed ({exc.code}) for {url}") from exc
    except Exception as exc:
        tmp.close()
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass
        raise RuntimeError(f"Download failed for {url}: {exc}") from exc
    finally:
        try:
            tmp.close()
        except Exception:
            pass

    # Detect Git LFS pointer downloads (common if using raw.githubusercontent.com)
    first_line = _read_first_text_line(tmp_path)
    if first_line.startswith("version https://git-lfs.github.com/spec/v1"):
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass
        raise RuntimeError(
            "Downloaded a Git LFS pointer file (not the parquet bytes). "
            "Use a media.githubusercontent.com URL, or pass --github so the script uses the media URL automatically."
        )

    return tmp_path


def _pick_column(df: pd.DataFrame, candidates: list[str]) -> str:
    lower_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        key = cand.lower()
        if key in lower_map:
            return lower_map[key]
    raise KeyError(
        f"None of these columns found: {candidates}. Available: {list(df.columns)}"
    )


def build_index(parquet_path: Path) -> list[dict]:
    df = pd.read_parquet(parquet_path).query(
        "parameter == 'capex_mw' and parameter_value > 0"
    )

    col_year = _pick_column(df, ["data_year", "atb_data_year", "atb_year", "year"])
    col_tech = _pick_column(df, ["technology", "tech", "atb_technology"])
    col_detail = _pick_column(
        df, ["tech_detail", "technology_detail", "detail", "atb_tech_detail"]
    )
    col_case = _pick_column(df, ["cost_case", "case", "atb_cost_case"])

    out = (
        df[[col_year, col_tech, col_detail, col_case]]
        .dropna()
        .drop_duplicates()
        .rename(
            columns={
                col_year: "data_year",
                col_tech: "technology",
                col_detail: "tech_detail",
                col_case: "cost_case",
            }
        )
    )

    # Normalize types / whitespace
    out["data_year"] = out["data_year"].astype(int)
    out["technology"] = out["technology"].astype(str).str.strip()
    out["tech_detail"] = out["tech_detail"].astype(str).str.strip()
    out["cost_case"] = out["cost_case"].astype(str).str.strip()

    out = out[
        (out["technology"] != "")
        & (out["tech_detail"] != "")
        & (out["cost_case"] != "")
    ]

    records = out.to_dict(orient="records")
    records.sort(
        key=lambda r: (
            r["data_year"],
            r["technology"],
            r["tech_detail"],
            r["cost_case"],
        )
    )
    return records


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build web/data/atb_options.json from the ATB parquet"
    )
    source_group = parser.add_mutually_exclusive_group(required=False)
    source_group.add_argument(
        "--parquet",
        type=Path,
        help="Path to technology_costs_atb.parquet (local file)",
    )
    source_group.add_argument(
        "--github",
        type=str,
        help=(
            "GitHub source spec 'OWNER/REPO:PATH@REF' (downloads via media.githubusercontent.com, works with Git LFS). "
            f"Default: {DEFAULT_GITHUB_SPEC}"
        ),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("web/data/atb_options.json"),
        help="Output JSON path (default: web/data/atb_options.json)",
    )
    args = parser.parse_args()

    parquet_path = None
    temp_download = None
    try:
        if args.parquet:
            parquet_path = args.parquet
        else:
            github_spec = args.github or DEFAULT_GITHUB_SPEC
            owner, repo, path, ref = _parse_github_spec(github_spec)
            url = _github_media_url(owner, repo, ref, path)
            print(f"Downloading parquet from: {url}")
            temp_download = download_to_temp(url)
            parquet_path = temp_download

        records = build_index(parquet_path)
    finally:
        if temp_download is not None:
            try:
                temp_download.unlink(missing_ok=True)
            except Exception:
                pass

    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = {"options": records}
    args.out.write_text(json.dumps(payload, indent=2) + "\n")

    print(f"Wrote {len(records):,} option rows to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
