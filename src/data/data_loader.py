"""
Data loading module that pairs MODIS imagery with wheat futures prices.
- Parses cleaned CSV price data
- Aligns each image date with price features as-of image date
- Targets are prices 20 days in the future
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
import tifffile
from sklearn.preprocessing import StandardScaler

from src.data.data_preprocessing import clean_wheat_csv
from src.utils.config import (
    FORECAST_HORIZON,
    IMAGE_CHANNELS,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
)

DATA_DIR = Path(__file__).resolve().parent
DEFAULT_PRICE_CSV = DATA_DIR / "futures_data" / "US_Wheat_Futures_cleaned_2023_2025.csv"
DEFAULT_IMAGE_DIR = DATA_DIR / "modis_cnn_tifs"
DATE_FMT = "%Y_%m_%d"

# Price features we keep for the tabular branch
PRICE_FEATURE_COLUMNS = [
    "Price",
    "Open",
    "High",
    "Low",
    "Volume",
    "ChangePct",
    "MA5",
    "MA20",
]


@dataclass
class Sample:
    date: datetime
    image: np.ndarray
    tabular: np.ndarray
    target: float


def _normalize_change_column(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure Change % column is numeric."""
    if "Change %" in df.columns:
        df["ChangePct"] = (
            df["Change %"]
            .astype(str)
            .str.replace("%", "", regex=False)
            .astype(float)
        )
    elif "ChangePct" not in df.columns:
        df["ChangePct"] = np.nan
    return df


def _compute_price_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add rolling features without leakage."""
    df = _normalize_change_column(df)
    df["MA5"] = df["Price"].rolling(window=5, min_periods=3).mean()
    df["MA20"] = df["Price"].rolling(window=20, min_periods=5).mean()
    return df


def load_price_dataframe(csv_path: Path = DEFAULT_PRICE_CSV) -> pd.DataFrame:
    """Load, clean, and enrich the price dataframe."""
    df = clean_wheat_csv(csv_path)
    df = _compute_price_features(df)
    return df


def _extract_date_from_filename(path: Path) -> datetime:
    return datetime.strptime(path.stem, DATE_FMT)


def _latest_price_features_before(
    price_df: pd.DataFrame, anchor_date: datetime, feature_cols: Sequence[str]
) -> Optional[np.ndarray]:
    """Get latest available price features on or before anchor_date."""
    mask = price_df["Date"] <= anchor_date
    if not mask.any():
        return None
    row = price_df.loc[mask].iloc[-1]
    features = row[list(feature_cols)]
    if features.isna().any():
        return None
    return features.to_numpy(dtype=np.float32)


def _future_price(
    price_df: pd.DataFrame, anchor_date: datetime, horizon_days: int
) -> Optional[float]:
    """Price at the first available date on/after anchor_date + horizon."""
    target_date = anchor_date + timedelta(days=horizon_days)
    future_rows = price_df[price_df["Date"] >= target_date]
    if future_rows.empty:
        return None
    return float(future_rows.iloc[0]["Price"])


def _resize_and_normalize_image(image: np.ndarray) -> np.ndarray:
    """Resize to config dims and min-max normalize per image."""
    image = image.astype(np.float32)
    # Replace NaNs
    if np.isnan(image).any():
        image = np.nan_to_num(image, nan=0.0)

    # Per-image robust min-max (2-98 percentile)
    lo, hi = np.percentile(image, [2, 98])
    if hi > lo:
        image = np.clip((image - lo) / (hi - lo), 0.0, 1.0)
    else:
        image = np.zeros_like(image, dtype=np.float32)

    image_tf = tf.convert_to_tensor(image)
    image_tf = tf.image.resize(image_tf, (IMAGE_HEIGHT, IMAGE_WIDTH))
    return image_tf.numpy().astype(np.float32)


def load_modis_image(
    image_path: Path,
    target_size: Tuple[int, int] = (IMAGE_HEIGHT, IMAGE_WIDTH),
    num_channels: int = IMAGE_CHANNELS,
) -> np.ndarray:
    """Load a MODIS .tif, trim/pad channels, resize, and normalize."""
    arr = tifffile.imread(str(image_path))

    if arr.ndim == 2:
        arr = np.expand_dims(arr, axis=-1)

    if arr.shape[-1] > num_channels:
        arr = arr[..., :num_channels]
    elif arr.shape[-1] < num_channels:
        # Repeat last channel to reach desired count
        repeats = num_channels // arr.shape[-1] + 1
        arr = np.repeat(arr, repeats=repeats, axis=-1)[..., :num_channels]

    arr = _resize_and_normalize_image(arr)
    return arr


def _build_samples(
    image_dir: Path,
    price_df: pd.DataFrame,
    horizon_days: int,
    feature_cols: Sequence[str],
) -> List[Sample]:
    samples: List[Sample] = []
    for tif_path in sorted(image_dir.glob("*.tif")):
        try:
            img_date = _extract_date_from_filename(tif_path)
        except ValueError:
            continue

        tabular = _latest_price_features_before(price_df, img_date, feature_cols)
        target_price = _future_price(price_df, img_date, horizon_days)

        if tabular is None or target_price is None:
            continue

        image = load_modis_image(tif_path)
        samples.append(
            Sample(
                date=img_date,
                image=image,
                tabular=tabular,
                target=target_price,
            )
        )
    return samples


def _to_arrays(
    samples: Iterable[Sample],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[datetime]]:
    images, tabular, targets, dates = [], [], [], []
    for s in samples:
        images.append(s.image)
        tabular.append(s.tabular)
        targets.append(s.target)
        dates.append(s.date)
    return (
        np.stack(images).astype(np.float32),
        np.stack(tabular).astype(np.float32),
        np.array(targets, dtype=np.float32),
        dates,
    )


def _split_by_year(
    samples: List[Sample],
    train_years: Sequence[int] = (2023,),
    val_years: Sequence[int] = (2024,),
    test_years: Sequence[int] = (2025,),
):
    train = [s for s in samples if s.date.year in train_years]
    val = [s for s in samples if s.date.year in val_years]
    test = [s for s in samples if s.date.year in test_years]
    return train, val, test


def load_data(
    price_csv: Path = DEFAULT_PRICE_CSV,
    image_dir: Path = DEFAULT_IMAGE_DIR,
    horizon_days: int = FORECAST_HORIZON,
    feature_cols: Sequence[str] = PRICE_FEATURE_COLUMNS,
    train_years: Sequence[int] = (2023,),
    val_years: Sequence[int] = (2024,),
    test_years: Sequence[int] = (2025,),
    tabular_scaler: Optional[StandardScaler] = None,
) -> Tuple[
    Dict[str, np.ndarray],
    Dict[str, np.ndarray],
    Dict[str, np.ndarray],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    StandardScaler,
]:
    """
    Load aligned image/price data and split by year.

    Returns:
        X_train, X_val, X_test dicts with 'images' and 'tabular'
        y_train, y_val, y_test numpy arrays (target prices)
        fitted StandardScaler for tabular features
    """
    price_df = load_price_dataframe(Path(price_csv))
    samples = _build_samples(Path(image_dir), price_df, horizon_days, feature_cols)

    if not samples:
        raise ValueError(
            "No aligned samples found. Check image dates and price CSV coverage."
        )

    train_s, val_s, test_s = _split_by_year(
        samples, train_years, val_years, test_years
    )

    if not train_s or not val_s or not test_s:
        raise ValueError(
            "Insufficient samples per split: "
            f"train={len(train_s)}, val={len(val_s)}, test={len(test_s)}"
        )

    X_img_train, X_tab_train, y_train, _ = _to_arrays(train_s)
    X_img_val, X_tab_val, y_val, _ = _to_arrays(val_s)
    X_img_test, X_tab_test, y_test, _ = _to_arrays(test_s)

    scaler = tabular_scaler or StandardScaler()
    if tabular_scaler is None:
        X_tab_train = scaler.fit_transform(X_tab_train)
    else:
        X_tab_train = scaler.transform(X_tab_train)
    X_tab_val = scaler.transform(X_tab_val)
    X_tab_test = scaler.transform(X_tab_test)

    X_train = {"images": X_img_train, "tabular": X_tab_train}
    X_val = {"images": X_img_val, "tabular": X_tab_val}
    X_test = {"images": X_img_test, "tabular": X_tab_test}

    return X_train, X_val, X_test, y_train, y_val, y_test, scaler
