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
DEFAULT_PRICE_CSV = DATA_DIR / "futures_data" / "US Wheat Futures 1990-2025.csv"
DEFAULT_IMAGE_DIR = DATA_DIR / "modis_cnn_tifs"
DATE_FMT = "%Y_%m_%d"

# Price features we keep for the tabular branch
PRICE_FEATURE_COLUMNS = [
    "Price",
    "Open",
    "High",
    "Low",
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
    price_df: pd.DataFrame,
    anchor_date: datetime,
    horizon_days: int,
    min_days: int = 10,
    max_days: int = 30,
) -> Optional[float]:
    """
    Price at the date closest to horizon_days within [min_days, max_days] after anchor_date.
    Prioritizes dates closer to the target horizon (20 days).
    """
    min_target_date = anchor_date + timedelta(days=min_days)
    max_target_date = anchor_date + timedelta(days=max_days)
    target_date = anchor_date + timedelta(days=horizon_days)
    
    # Find prices within the window [min_days, max_days]
    future_rows = price_df[
        (price_df["Date"] >= min_target_date) & (price_df["Date"] <= max_target_date)
    ]
    if future_rows.empty:
        return None
    
    # Calculate distance from target horizon for each available date
    future_rows = future_rows.copy()
    future_rows["days_from_target"] = (
        (future_rows["Date"] - target_date).dt.total_seconds() / 86400
    ).abs()
    
    # Sort by distance from target horizon and pick the closest
    closest_row = future_rows.loc[future_rows["days_from_target"].idxmin()]
    return float(closest_row["Price"])


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
    total_images = 0
    skipped_no_tabular = 0
    skipped_no_target = 0
    skipped_invalid_date = 0
    # year -> {'total': count, 'skipped_tabular': count,
    #          'skipped_target': count, 'success': count}
    year_counts = {}
    skipped_examples = []  # Store examples of skipped dates for debugging
    
    price_min_date = price_df["Date"].min()
    price_max_date = price_df["Date"].max()
    
    for tif_path in sorted(image_dir.glob("*.tif")):
        total_images += 1
        try:
            img_date = _extract_date_from_filename(tif_path)
        except ValueError:
            skipped_invalid_date += 1
            continue

        year = img_date.year
        if year not in year_counts:
            year_counts[year] = {
                'total': 0, 
                'skipped_tabular': 0, 
                'skipped_target': 0, 
                'success': 0
            }
        year_counts[year]['total'] += 1

        tabular = _latest_price_features_before(price_df, img_date, feature_cols)
        target_price = _future_price(
            price_df, img_date, horizon_days, min_days=10, max_days=30
        )
        target_date = img_date + timedelta(days=horizon_days)

        if tabular is None:
            skipped_no_tabular += 1
            year_counts[year]['skipped_tabular'] += 1
            if len(skipped_examples) < 10:
                img_date_str = img_date.strftime('%Y-%m-%d')
                first_price_str = price_min_date.strftime('%Y-%m-%d')
                msg = (
                    f"{img_date_str} "
                    f"(no price data on/before image date, "
                    f"first price: {first_price_str})"
                )
                skipped_examples.append(msg)
            continue
        
        if target_price is None:
            skipped_no_target += 1
            year_counts[year]['skipped_target'] += 1
            if len(skipped_examples) < 10:
                min_target_date = img_date + timedelta(days=10)
                max_target_date = img_date + timedelta(days=30)
                skipped_examples.append(
                    f"{img_date.strftime('%Y-%m-%d')} "
                    f"(no price data in range {min_target_date.strftime('%Y-%m-%d')} to "
                    f"{max_target_date.strftime('%Y-%m-%d')}, "
                    f"last price: {price_max_date.strftime('%Y-%m-%d')})"
                )
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
        year_counts[year]['success'] += 1
    
    # Print detailed summary
    total_skipped = skipped_no_tabular + skipped_no_target
    print("\nImage/Price Alignment Summary:")
    print(f"  Total .tif files found: {total_images}")
    if skipped_invalid_date > 0:
        print(f"  Skipped (invalid date format): {skipped_invalid_date}")
    print(f"  Successfully aligned: {len(samples)}")
    print(f"  Skipped (no price data before image): {skipped_no_tabular}")
    print(
        f"  Skipped (no price data in range 10-30 days after image, "
        f"prioritizing {horizon_days} days): {skipped_no_target}"
    )
    print(f"  Total skipped: {total_skipped}")
    price_min_str = price_min_date.strftime('%Y-%m-%d')
    price_max_str = price_max_date.strftime('%Y-%m-%d')
    print(f"  Price data range: {price_min_str} to {price_max_str}")
    print("\n  Breakdown by year:")
    for year in sorted(year_counts.keys()):
        counts = year_counts[year]
        total_skipped_year = counts['skipped_tabular'] + counts['skipped_target']
        tab_skip = counts['skipped_tabular']
        tgt_skip = counts['skipped_target']
        print(
            f"    {year}: {counts['success']} aligned, "
            f"{total_skipped_year} skipped "
            f"({tab_skip} no tabular, {tgt_skip} no target) "
            f"(out of {counts['total']} images)"
        )
    
    if skipped_examples:

        num_examples = min(10, len(skipped_examples))
        print(f"\n  Example skipped dates (showing first {num_examples}):")
        for example in skipped_examples[len(skipped_examples)-num_examples:]:
            print(f"    - {example}")
    
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
    train_years: Sequence[int] = tuple(range(2009, 2022)),  # 2009-2021 (13 years)
    val_years: Sequence[int] = (2022, 2023),  # 2 years
    test_years: Sequence[int] = (2024, 2025),  # 2 years
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
    train_years: Sequence[int] = tuple(range(2009, 2022)),  # 2009-2021 (13 years)
    val_years: Sequence[int] = (2022, 2023),  # 2 years
    test_years: Sequence[int] = (2024, 2025),  # 2 years
    tabular_scaler: Optional[StandardScaler] = None,
) -> Tuple[
    Dict[str, np.ndarray],
    Dict[str, np.ndarray],
    Dict[str, np.ndarray],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    StandardScaler,
    List[datetime],
    pd.DataFrame,
]:
    """
    Load aligned image/price data and split by year.

    Args:
        price_csv: Path to price CSV file
        image_dir: Directory containing MODIS .tif files
        horizon_days: Target forecast horizon (e.g., 20 days). 
                     Looks for prices in range [10, 30] days after image,
                     prioritizing dates closest to horizon_days
        feature_cols: Price feature columns to extract
        train_years: Years to include in training set
        val_years: Years to include in validation set
        test_years: Years to include in test set
        tabular_scaler: Optional pre-fitted scaler for tabular features

    Returns:
        X_train, X_val, X_test dicts with 'images' and 'tabular'
        y_train, y_val, y_test numpy arrays (target prices)
        fitted StandardScaler for tabular features
        test_dates: List of datetime objects for test samples (for flexible PNL)
        price_df: Price dataframe (for flexible PNL)
    """
    price_df = load_price_dataframe(Path(price_csv))
    samples = _build_samples(
        Path(image_dir), price_df, horizon_days, feature_cols
    )

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
    X_img_test, X_tab_test, y_test, test_dates = _to_arrays(test_s)

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

    return X_train, X_val, X_test, y_train, y_val, y_test, scaler, test_dates, price_df
