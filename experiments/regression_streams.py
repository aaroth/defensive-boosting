"""Chronological bounded-regression streams used in the appendix experiments."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .real_streams import (
    PROCESSED_DIR,
    RAW_DIR,
    _add_feature,
    _download,
    _ensure_dirs,
    _normalize_rows,
    _online_standardize_matrix,
    _zip_member,
)


APPLIANCES_URL = (
    "https://archive.ics.uci.edu/static/public/374/"
    "appliances+energy+prediction.zip"
)
BIKE_URL = (
    "https://archive.ics.uci.edu/static/public/275/"
    "bike+sharing+dataset.zip"
)
TRAFFIC_URL = (
    "https://archive.ics.uci.edu/static/public/492/"
    "metro+interstate+traffic+volume.zip"
)


@dataclass
class RegressionStreamData:
    """A prequential regression stream with a chronological initialization prefix.

    Targets are affinely scaled to ``[0, 1]`` using bounds fixed before the
    run. Algorithms train on the initialization prefix, but all reported
    losses are on the subsequent evaluation stream.
    """

    name: str
    X_init: np.ndarray
    y_init: np.ndarray
    X: np.ndarray
    y: np.ndarray
    raw_y: np.ndarray
    target_min: float
    target_max: float
    target_unit: str
    clipped_fraction: float
    description: str

    @property
    def T(self) -> int:
        return int(self.y.shape[0])

    @property
    def dim(self) -> int:
        return int(self.X.shape[1])

    @property
    def initialization_rounds(self) -> int:
        return int(self.y_init.shape[0])

    def to_raw_scale(self, predictions: np.ndarray) -> np.ndarray:
        predictions = np.asarray(predictions, dtype=float)
        return self.target_min + predictions * (self.target_max - self.target_min)


def regression_streams(
    *,
    max_rows: int | None = None,
    dim: int = 128,
    initialization_fraction: float = 0.1,
    progress: bool = False,
) -> list[RegressionStreamData]:
    """Return the three naturally ordered regression streams."""

    loaders = [appliances_energy, bike_demand, interstate_traffic]
    streams = []
    for loader in loaders:
        if progress:
            print(f"Loading {loader.__name__.replace('_', ' ')}...", flush=True)
        stream = loader(
            max_rows=max_rows,
            dim=dim,
            initialization_fraction=initialization_fraction,
        )
        streams.append(stream)
        if progress:
            print(
                f"Loaded {stream.name}: {stream.initialization_rounds} initialization "
                f"and {stream.T} evaluation rounds",
                flush=True,
            )
    return streams


def _calendar_features(frame: pd.DataFrame, timestamps: pd.Series) -> None:
    hour = timestamps.dt.hour.to_numpy(dtype=float)
    minute = timestamps.dt.minute.to_numpy(dtype=float)
    time_of_day = hour + minute / 60.0
    weekday = timestamps.dt.dayofweek.to_numpy(dtype=float)
    day_of_year = timestamps.dt.dayofyear.to_numpy(dtype=float)
    frame["calendar_hour_sin"] = np.sin(2.0 * math.pi * time_of_day / 24.0)
    frame["calendar_hour_cos"] = np.cos(2.0 * math.pi * time_of_day / 24.0)
    frame["calendar_weekday_sin"] = np.sin(2.0 * math.pi * weekday / 7.0)
    frame["calendar_weekday_cos"] = np.cos(2.0 * math.pi * weekday / 7.0)
    frame["calendar_year_sin"] = np.sin(2.0 * math.pi * day_of_year / 365.25)
    frame["calendar_year_cos"] = np.cos(2.0 * math.pi * day_of_year / 365.25)
    frame["calendar_elapsed_days"] = (
        timestamps - timestamps.iloc[0]
    ).dt.total_seconds().to_numpy(dtype=float) / (24.0 * 60.0 * 60.0)


def _add_lagged_targets(
    frame: pd.DataFrame,
    timestamps: pd.Series,
    raw_targets: np.ndarray,
    lag_deltas: Iterable[pd.Timedelta],
) -> list[str]:
    """Add target values from exact earlier timestamps, never future rows."""

    target_by_time = pd.Series(raw_targets, index=pd.DatetimeIndex(timestamps))
    target_by_time = target_by_time.groupby(level=0, sort=False).last()
    names = []
    for delta in lag_deltas:
        total_minutes = int(delta.total_seconds() // 60)
        name = f"target_lag_{total_minutes}m"
        lookup_times = pd.DatetimeIndex(timestamps) - delta
        frame[name] = target_by_time.reindex(lookup_times).to_numpy(dtype=float)
        names.append(name)
    return names


def _encode_contexts(
    frame: pd.DataFrame,
    *,
    numeric_cols: list[str],
    categorical_cols: list[str],
    dim: int,
) -> np.ndarray:
    """Hash contexts after standardizing from statistics of earlier rows."""

    numeric = (
        frame[numeric_cols]
        .apply(pd.to_numeric, errors="coerce")
        .to_numpy(dtype=float)
    )
    standardized = _online_standardize_matrix(numeric)
    X = np.zeros((len(frame), dim), dtype=float)
    for i in range(len(frame)):
        _add_feature(X[i], "__bias__")
        for j, col in enumerate(numeric_cols):
            if np.isfinite(numeric[i, j]):
                _add_feature(X[i], f"{col}:num", standardized[i, j])
            else:
                _add_feature(X[i], f"{col}=__missing__")
        for col in categorical_cols:
            value = frame.iloc[i][col]
            key = "__missing__" if pd.isna(value) else str(value)
            _add_feature(X[i], f"{col}={key}")
    return _normalize_rows(X)


def _build_stream(
    *,
    name: str,
    frame: pd.DataFrame,
    timestamp_col: str,
    target_col: str,
    numeric_cols: list[str],
    categorical_cols: list[str],
    lag_deltas: Iterable[pd.Timedelta],
    dim: int,
    initialization_fraction: float,
    target_unit: str,
    target_bounds: tuple[float, float] | None,
    description: str,
) -> RegressionStreamData:
    if not 0.0 < initialization_fraction < 0.5:
        raise ValueError("initialization_fraction must lie in (0, 1/2)")
    if len(frame) < 20:
        raise ValueError("regression streams require at least 20 rows")

    frame = frame.copy()
    frame[timestamp_col] = pd.to_datetime(frame[timestamp_col], errors="raise")
    frame = frame.sort_values(timestamp_col, kind="stable").reset_index(drop=True)
    timestamps = frame[timestamp_col]
    raw_targets = pd.to_numeric(frame[target_col], errors="raise").to_numpy(dtype=float)
    if not np.all(np.isfinite(raw_targets)):
        raise ValueError(f"{name} contains nonfinite targets")

    initialization_rounds = max(256, int(math.floor(initialization_fraction * len(frame))))
    initialization_rounds = min(initialization_rounds, len(frame) - 1)
    prefix_targets = raw_targets[:initialization_rounds]
    if target_bounds is None:
        target_min = float(np.min(prefix_targets))
        target_max = float(np.max(prefix_targets))
    else:
        target_min, target_max = map(float, target_bounds)
    if not target_max > target_min:
        raise ValueError(f"{name} has a constant initialization target")
    scale = target_max - target_min
    scaled_targets = np.clip((raw_targets - target_min) / scale, 0.0, 1.0)
    evaluation_raw = raw_targets[initialization_rounds:]
    clipped_fraction = float(
        np.mean((evaluation_raw < target_min) | (evaluation_raw > target_max))
    )

    _calendar_features(frame, timestamps)
    lag_cols = _add_lagged_targets(frame, timestamps, raw_targets, lag_deltas)
    calendar_cols = [
        "calendar_hour_sin",
        "calendar_hour_cos",
        "calendar_weekday_sin",
        "calendar_weekday_cos",
        "calendar_year_sin",
        "calendar_year_cos",
        "calendar_elapsed_days",
    ]
    contexts = _encode_contexts(
        frame,
        numeric_cols=[*numeric_cols, *calendar_cols, *lag_cols],
        categorical_cols=categorical_cols,
        dim=dim,
    )
    return RegressionStreamData(
        name=name,
        X_init=contexts[:initialization_rounds],
        y_init=scaled_targets[:initialization_rounds],
        X=contexts[initialization_rounds:],
        y=scaled_targets[initialization_rounds:],
        raw_y=evaluation_raw,
        target_min=target_min,
        target_max=target_max,
        target_unit=target_unit,
        clipped_fraction=clipped_fraction,
        description=description,
    )


def appliances_energy(
    *,
    max_rows: int | None = None,
    dim: int = 128,
    initialization_fraction: float = 0.1,
) -> RegressionStreamData:
    """Ten-minute household appliance-energy stream in recorded order."""

    cache = PROCESSED_DIR / (
        f"regression_appliances_v2_n{max_rows}_d{dim}_w{initialization_fraction:g}.npz"
    )
    if cache.exists():
        return _stream_from_cache(cache)
    zip_path = RAW_DIR / "appliances_energy_prediction.zip"
    _download(APPLIANCES_URL, zip_path)
    csv_path = _zip_member(zip_path, "energydata_complete.csv")
    frame = pd.read_csv(csv_path)
    if max_rows is not None:
        frame = frame.iloc[:max_rows].copy()
    numeric_cols = [
        col
        for col in frame.columns
        if col not in {"date", "Appliances", "lights", "rv1", "rv2"}
    ]
    stream = _build_stream(
        name="regression_appliances_energy",
        frame=frame,
        timestamp_col="date",
        target_col="Appliances",
        numeric_cols=numeric_cols,
        categorical_cols=[],
        lag_deltas=[
            pd.Timedelta(minutes=10),
            pd.Timedelta(hours=1),
            pd.Timedelta(days=1),
        ],
        dim=dim,
        initialization_fraction=initialization_fraction,
        target_unit="Wh",
        target_bounds=(0.0, 2000.0),
        description=(
            "UCI Appliances Energy Prediction, sampled every ten minutes. "
            "Contexts contain environmental sensors, calendar features, and "
            "past appliance-energy observations; simultaneous light use and "
            "the two random decoy columns are omitted."
        ),
    )
    _save_stream(cache, stream)
    return stream


def bike_demand(
    *,
    max_rows: int | None = None,
    dim: int = 128,
    initialization_fraction: float = 0.1,
) -> RegressionStreamData:
    """Hourly Capital Bikeshare rental-demand stream in recorded order."""

    cache = PROCESSED_DIR / (
        f"regression_bike_v2_n{max_rows}_d{dim}_w{initialization_fraction:g}.npz"
    )
    if cache.exists():
        return _stream_from_cache(cache)
    zip_path = RAW_DIR / "bike_sharing_dataset.zip"
    _download(BIKE_URL, zip_path)
    csv_path = _zip_member(zip_path, "hour.csv")
    frame = pd.read_csv(csv_path)
    frame["timestamp"] = pd.to_datetime(frame["dteday"], errors="raise") + pd.to_timedelta(
        frame["hr"], unit="h"
    )
    if max_rows is not None:
        frame = frame.iloc[:max_rows].copy()
    stream = _build_stream(
        name="regression_bike_demand",
        frame=frame,
        timestamp_col="timestamp",
        target_col="cnt",
        numeric_cols=["temp", "atemp", "hum", "windspeed"],
        categorical_cols=[
            "season",
            "yr",
            "mnth",
            "hr",
            "holiday",
            "weekday",
            "workingday",
            "weathersit",
        ],
        lag_deltas=[
            pd.Timedelta(hours=1),
            pd.Timedelta(days=1),
            pd.Timedelta(days=7),
        ],
        dim=dim,
        initialization_fraction=initialization_fraction,
        target_unit="rentals/hour",
        target_bounds=(0.0, 2000.0),
        description=(
            "UCI Bike Sharing hourly Capital Bikeshare demand in 2011--2012. "
            "Contexts contain calendar and weather variables plus past demand; "
            "the casual and registered counts are omitted because they sum to "
            "the target."
        ),
    )
    _save_stream(cache, stream)
    return stream


def interstate_traffic(
    *,
    max_rows: int | None = None,
    dim: int = 128,
    initialization_fraction: float = 0.1,
) -> RegressionStreamData:
    """Hourly westbound I-94 traffic-volume stream in recorded order."""

    cache = PROCESSED_DIR / (
        f"regression_traffic_v2_n{max_rows}_d{dim}_w{initialization_fraction:g}.npz"
    )
    if cache.exists():
        return _stream_from_cache(cache)
    zip_path = RAW_DIR / "metro_interstate_traffic_volume.zip"
    _download(TRAFFIC_URL, zip_path)
    csv_path = _zip_member(zip_path, "Metro_Interstate_Traffic_Volume.csv.gz")
    frame = pd.read_csv(csv_path)
    frame["date_time"] = pd.to_datetime(frame["date_time"], errors="raise")
    frame = frame.sort_values("date_time", kind="stable")
    # Some hours have duplicate weather reports.  The traffic count is the
    # same within every such group; retaining one observation
    # avoids supplying an immediately repeated target as a lag feature.
    frame = frame.drop_duplicates(subset="date_time", keep="last").reset_index(
        drop=True
    )
    if max_rows is not None:
        frame = frame.iloc[:max_rows].copy()
    stream = _build_stream(
        name="regression_interstate_traffic",
        frame=frame,
        timestamp_col="date_time",
        target_col="traffic_volume",
        numeric_cols=["temp", "rain_1h", "snow_1h", "clouds_all"],
        categorical_cols=["holiday", "weather_main", "weather_description"],
        lag_deltas=[
            pd.Timedelta(hours=1),
            pd.Timedelta(days=1),
            pd.Timedelta(days=7),
        ],
        dim=dim,
        initialization_fraction=initialization_fraction,
        target_unit="vehicles/hour",
        target_bounds=(0.0, 10000.0),
        description=(
            "UCI Metro Interstate Traffic Volume, hourly westbound I-94 counts "
            "from 2012--2018. Contexts contain weather, calendar variables, and "
            "past traffic counts; duplicate weather reports at one timestamp "
            "are collapsed before evaluation."
        ),
    )
    _save_stream(cache, stream)
    return stream


def _save_stream(path: Path, stream: RegressionStreamData) -> None:
    _ensure_dirs()
    np.savez_compressed(
        path,
        name=stream.name,
        X_init=stream.X_init,
        y_init=stream.y_init,
        X=stream.X,
        y=stream.y,
        raw_y=stream.raw_y,
        target_min=stream.target_min,
        target_max=stream.target_max,
        target_unit=stream.target_unit,
        clipped_fraction=stream.clipped_fraction,
        description=stream.description,
    )


def _stream_from_cache(path: Path) -> RegressionStreamData:
    data = np.load(path, allow_pickle=True)
    return RegressionStreamData(
        name=str(data["name"]),
        X_init=np.asarray(data["X_init"], dtype=float),
        y_init=np.asarray(data["y_init"], dtype=float),
        X=np.asarray(data["X"], dtype=float),
        y=np.asarray(data["y"], dtype=float),
        raw_y=np.asarray(data["raw_y"], dtype=float),
        target_min=float(data["target_min"]),
        target_max=float(data["target_max"]),
        target_unit=str(data["target_unit"]),
        clipped_fraction=float(data["clipped_fraction"]),
        description=str(data["description"]),
    )
