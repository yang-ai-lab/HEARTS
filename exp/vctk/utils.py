"""
Utility functions and constants for VCTK dataset experiments.
"""

import os
import re
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
from loguru import logger

# Path constants
# Override with the VCTK_DATA_DIR environment variable to point at your local
# VCTK dataset directory (containing speaker-info.txt).
VCTK_DATA_DIR = Path(os.environ.get("VCTK_DATA_DIR", "data/VCTK"))
SPEAKER_INFO_PATH = VCTK_DATA_DIR / "speaker-info.txt"

# Global cache for speaker metadata
_SPEAKER_DF_CACHE: Optional[pd.DataFrame] = None


def load_speaker_info(speaker_info_path: Path = SPEAKER_INFO_PATH) -> pd.DataFrame:
    """
    Parse speaker-info.txt file and return as DataFrame.

    Args:
        speaker_info_path: Path to speaker-info.txt file

    Returns:
        DataFrame with columns: speaker_id, age, gender, ACCENTS, REGION
    """
    with speaker_info_path.open() as f:
        lines = [line.strip() for line in f if line.strip()]

    if not lines:
        return pd.DataFrame()

    headers = re.split(r"\s{2,}", lines[0])
    records = []
    for line in lines[1:]:
        parts = re.split(r"\s{2,}", line)
        if len(parts) < len(headers):
            parts += [""] * (len(headers) - len(parts))
        records.append(parts[: len(headers)])

    df = pd.DataFrame(records, columns=headers)

    # Rename columns for consistency
    column_map = {"ID": "speaker_id", "AGE": "age", "GENDER": "gender"}
    df.rename(columns=column_map, inplace=True)

    # Clean and convert data
    if "age" in df.columns:
        df["age"] = pd.to_numeric(df["age"], errors="coerce")
    if "gender" in df.columns:
        df["gender"] = df["gender"].str.strip()
    if "ACCENTS" in df.columns:
        df["ACCENTS"] = df["ACCENTS"].str.strip()
    if "REGION" in df.columns:
        df["REGION"] = df["REGION"].str.strip()
    else:
        df["REGION"] = ""

    return df


def _get_cached_speaker_df(speaker_info_path: Path = SPEAKER_INFO_PATH) -> pd.DataFrame:
    """
    Get cached speaker DataFrame or load it if not cached.

    Args:
        speaker_info_path: Path to speaker-info.txt file

    Returns:
        Cached or newly loaded speaker DataFrame
    """
    global _SPEAKER_DF_CACHE

    if _SPEAKER_DF_CACHE is None:
        logger.info(f"Loading speaker info from {speaker_info_path}")
        _SPEAKER_DF_CACHE = load_speaker_info(speaker_info_path)
        logger.info(f"Loaded metadata for {len(_SPEAKER_DF_CACHE)} speakers")

    return _SPEAKER_DF_CACHE


def get_speaker_metadata(
    speaker_id: str, speaker_info_path: Path = SPEAKER_INFO_PATH
) -> Dict[str, Any]:
    """
    Get metadata for a specific speaker.

    Args:
        speaker_id: Speaker identifier (e.g., "p225")
        speaker_info_path: Path to speaker-info.txt file

    Returns:
        Dictionary with speaker metadata (age, gender, accent, region)
    """
    df = _get_cached_speaker_df(speaker_info_path)

    if df.empty or "speaker_id" not in df.columns:
        return {}

    speaker_row = df[df["speaker_id"] == speaker_id]
    if speaker_row.empty:
        return {}

    row = speaker_row.iloc[0]
    metadata = {
        "speaker_id": speaker_id,
        "age": row.get("age"),
        "gender": row.get("gender", ""),
        "accent": row.get("ACCENTS", ""),
        "region": row.get("REGION", ""),
    }

    return metadata
