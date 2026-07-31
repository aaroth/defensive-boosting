"""Real binary streams for the online boosting experiments."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import time
from pathlib import Path
import requests
import zipfile

import numpy as np
import pandas as pd
from scipy.io import arff

from .streams import StreamData


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

BANK_URL = "https://archive.ics.uci.edu/static/public/222/bank+marketing.zip"
ELEC_URL = "https://master.dl.sourceforge.net/project/moa-datastream/Datasets/Classification/elecNormNew.arff.zip"
AIRLINES_URL = "https://master.dl.sourceforge.net/project/moa-datastream/Datasets/Classification/airlines.arff.zip"
OCCUPANCY_URL = "https://archive.ics.uci.edu/static/public/357/occupancy+detection.zip"

# Public INSECTS URLs maintained by River.  The source paper calls the
# `gradual` variant "incremental-gradual" and the two recurring variants
# "incremental-abrupt-reoccurring" and "incremental-reoccurring".
INSECTS_VARIANTS = {
    "abrupt": {
        "url": "https://drive.google.com/uc?export=download&id=1WQoIuuVgiuXfzv4kvao6XuLQG37V923O&confirm=t",
        "filename": "insects_abrupt_balanced.csv",
        "stream_name": "real_insects_abrupt",
        "description": "five abrupt temperature changes",
    },
    "incremental_gradual": {
        "url": "https://drive.google.com/uc?export=download&id=1fepYkDxwMbuoRUaG_fsymSzkuapS4vJp&confirm=t",
        "filename": "insects_gradual_balanced.csv",
        "stream_name": "real_insects_incremental_gradual",
        "description": "incremental drift with one gradual transition",
    },
    "incremental_abrupt": {
        "url": "https://drive.google.com/uc?export=download&id=1-J5WIBN8_F_tomdcrOaiLCxk9nzxtFsf&confirm=t",
        "filename": "insects_incremental_abrupt_balanced.csv",
        "stream_name": "real_insects_incremental_abrupt",
        "description": "recurring incremental drift separated by abrupt changes",
    },
    "incremental_recurring": {
        "url": "https://drive.google.com/uc?export=download&id=1mSKTSsxzYMjdV005AJqrcMGajuu7dUfW&confirm=t",
        "filename": "insects_incremental_reoccurring_balanced.csv",
        "stream_name": "real_insects_incremental_recurring",
        "description": "recurring increasing and decreasing incremental drift",
    },
    "incremental": {
        "url": "https://drive.google.com/uc?export=download&id=1tKQ2KL4m-ACHCVKUDLFPrM4cyhioiOpu&confirm=t",
        "filename": "insects_incremental_balanced.csv",
        "stream_name": "real_insects_incremental",
        "description": "continuous incremental temperature drift",
    },
}

# The original numeric labels are ordered as Aedes aegypti female/male
# (2, 3), Aedes albopictus female/male (4, 5), and Culex quinquefasciatus
# female/male (11, 12).
INSECTS_ALBOPICTUS_LABELS = frozenset({4, 5})


@dataclass(frozen=True)
class RealStreamSpec:
    name: str
    dim: int = 128
    max_rows: int | None = None


def real_streams(
    *,
    max_rows: int | None = 50_000,
    dim: int = 128,
    progress: bool = False,
) -> list[StreamData]:
    """Return the four real datasets used in the experiments."""

    specs = [
        ("Bank Marketing", bank_marketing, max_rows),
        ("MOA Electricity", moa_electricity, max_rows),
        ("MOA Airlines", moa_airlines, max_rows),
        ("UCI Occupancy Detection", occupancy_detection, max_rows),
    ]
    streams = []
    for label, loader, rows in specs:
        if progress:
            print(f"Loading {label}...", flush=True)
        start = time.perf_counter()
        stream = loader(max_rows=rows, dim=dim)
        streams.append(stream)
        if progress:
            elapsed = time.perf_counter() - start
            print(f"Loaded {label}: {stream.T} rows in {elapsed:.1f}s", flush=True)
    return streams


def insects_drift_streams(
    *,
    max_rows: int | None = None,
    progress: bool = False,
) -> list[StreamData]:
    """Return the five balanced INSECTS drift variants in released order."""

    streams = []
    for variant in INSECTS_VARIANTS:
        if progress:
            print(f"Loading INSECTS {variant.replace('_', '-')}...", flush=True)
        start = time.perf_counter()
        stream = insects_drift(variant=variant, max_rows=max_rows)
        streams.append(stream)
        if progress:
            elapsed = time.perf_counter() - start
            print(f"Loaded {stream.name}: {stream.T} rows in {elapsed:.1f}s", flush=True)
    return streams


def _ensure_dirs() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def _download(url: str, path: Path) -> None:
    _ensure_dirs()
    if path.exists() and path.stat().st_size > 0:
        return
    tmp = path.with_suffix(path.suffix + ".tmp")
    with requests.get(url, stream=True, timeout=120) as response:
        response.raise_for_status()
        with tmp.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)
    tmp.replace(path)


def _zip_member(zip_path: Path, suffix: str) -> Path:
    out_dir = RAW_DIR / zip_path.stem
    marker = out_dir / ".extracted"
    if not marker.exists():
        out_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(out_dir)
        marker.write_text("ok\n")
    matches = [p for p in out_dir.rglob("*") if p.name.endswith(suffix)]
    if matches:
        return matches[0]
    for nested in [p for p in out_dir.rglob("*.zip") if p.is_file()]:
        nested_dir = nested.with_suffix("")
        nested_marker = nested_dir / ".extracted"
        if not nested_marker.exists():
            nested_dir.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(nested) as zf:
                zf.extractall(nested_dir)
            nested_marker.write_text("ok\n")
    matches = [p for p in out_dir.rglob("*") if p.name.endswith(suffix)]
    if not matches:
        raise FileNotFoundError(f"no {suffix} in {zip_path}")
    return matches[0]


def _stable_hash(text: str) -> int:
    return int.from_bytes(hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest(), "little")


def _add_feature(x: np.ndarray, key: str, value: float = 1.0) -> None:
    h = _stable_hash(key)
    idx = h % x.size
    sign = 1.0 if ((h >> 63) & 1) == 0 else -1.0
    x[idx] += sign * float(value)


def _normalize_rows(X: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(norms, 1e-12)


def _online_standardize_matrix(X: np.ndarray) -> np.ndarray:
    """Standardize each row using only statistics from earlier rows."""

    X = np.asarray(X, dtype=float)
    standardized = np.zeros_like(X, dtype=float)
    count = np.zeros(X.shape[1], dtype=float)
    mean = np.zeros(X.shape[1], dtype=float)
    m2 = np.zeros(X.shape[1], dtype=float)
    for i, row in enumerate(X):
        finite = np.isfinite(row)
        ready = finite & (count >= 2.0)
        variance = np.divide(m2, count, out=np.ones_like(m2), where=count > 0.0)
        scale = np.sqrt(np.maximum(variance, 1e-12))
        standardized[i, ready] = np.clip(
            (row[ready] - mean[ready]) / scale[ready],
            -5.0,
            5.0,
        ) / 5.0

        count[finite] += 1.0
        delta = row[finite] - mean[finite]
        mean[finite] += delta / count[finite]
        delta2 = row[finite] - mean[finite]
        m2[finite] += delta * delta2
    return standardized


def _decode_frame(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].map(lambda v: v.decode("utf-8") if isinstance(v, bytes) else v)
    return df


def _binary_labels(values: pd.Series, positive: str | int | float | None = None) -> np.ndarray:
    vals = values.map(lambda v: v.decode("utf-8") if isinstance(v, bytes) else v)
    if positive is None:
        unique = sorted(str(v) for v in pd.unique(vals.dropna()))
        if len(unique) != 2:
            raise ValueError(f"expected binary labels, got {unique}")
        positive = unique[-1]
    return np.where(vals.astype(str) == str(positive), 1, -1).astype(int)


def _hashed_frame(
    df: pd.DataFrame,
    *,
    label_col: str,
    positive: str | int | float | None,
    numeric_cols: list[str] | None,
    dim: int,
    max_rows: int | None,
    drop_cols: set[str] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if max_rows is not None:
        df = df.iloc[:max_rows].copy()
    else:
        df = df.copy()
    df = _decode_frame(df)
    y = _binary_labels(df[label_col], positive=positive)
    drop = {label_col, *(drop_cols or set())}
    feature_cols = [c for c in df.columns if c not in drop]
    if numeric_cols is None:
        numeric_cols = [col for col in feature_cols if pd.api.types.is_numeric_dtype(df[col])]
    numeric_set = set(numeric_cols)

    numeric_values: dict[str, np.ndarray] = {}
    for col in numeric_cols:
        if col not in feature_cols:
            continue
        arr = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
        if np.any(np.isfinite(arr)):
            numeric_values[col] = arr

    numeric_count = {col: 0 for col in numeric_values}
    numeric_mean = {col: 0.0 for col in numeric_values}
    numeric_m2 = {col: 0.0 for col in numeric_values}

    X = np.zeros((len(df), dim), dtype=float)
    for i, (_, row) in enumerate(df.iterrows()):
        _add_feature(X[i], "__bias__", 1.0)
        for col in feature_cols:
            val = row[col]
            if pd.isna(val):
                _add_feature(X[i], f"{col}=__missing__", 1.0)
                continue
            if col in numeric_set and col in numeric_values:
                num = numeric_values[col][i]
                if np.isfinite(num):
                    count = numeric_count[col]
                    if count >= 2:
                        variance = numeric_m2[col] / count
                        std = math.sqrt(max(variance, 1e-12))
                        scaled = float(np.clip((num - numeric_mean[col]) / std, -5.0, 5.0)) / 5.0
                    else:
                        scaled = 0.0
                    _add_feature(X[i], f"{col}:num", scaled)

                    numeric_count[col] = count + 1
                    delta = num - numeric_mean[col]
                    numeric_mean[col] += delta / numeric_count[col]
                    numeric_m2[col] += delta * (num - numeric_mean[col])
                    continue
            _add_feature(X[i], f"{col}={val}", 1.0)
    return _normalize_rows(X), y


def bank_marketing(*, max_rows: int | None = 50_000, dim: int = 128) -> StreamData:
    """Portuguese bank telemarketing stream, preserving the UCI date order."""

    cache = PROCESSED_DIR / f"bank_marketing_online_v2_n{max_rows}_d{dim}.npz"
    if cache.exists():
        data = np.load(cache, allow_pickle=True)
        return _stream_from_cache(data)
    zip_path = RAW_DIR / "bank_marketing.zip"
    _download(BANK_URL, zip_path)
    csv_path = _zip_member(zip_path, "bank-additional-full.csv")
    df = pd.read_csv(csv_path, sep=";")
    numeric_cols = [
        "age",
        "campaign",
        "pdays",
        "previous",
        "emp.var.rate",
        "cons.price.idx",
        "cons.conf.idx",
        "euribor3m",
        "nr.employed",
    ]
    X, y = _hashed_frame(
        df,
        label_col="y",
        positive="yes",
        numeric_cols=numeric_cols,
        dim=dim,
        max_rows=max_rows,
        drop_cols={"duration"},
    )
    stream = StreamData(
        name="real_bank_marketing",
        X=X,
        y=y,
        weak_type="linear",
        gamma_hint=None,
        description=(
            "UCI Bank Marketing, ordered by campaign date. The target is term "
            "deposit subscription; the call duration field is dropped because "
            "it is not available before prediction."
        ),
    )
    _save_stream(cache, stream)
    return stream


def _load_arff_zip(url: str, zip_name: str, suffix: str) -> pd.DataFrame:
    zip_path = RAW_DIR / zip_name
    _download(url, zip_path)
    arff_path = _zip_member(zip_path, suffix)
    data, _ = arff.loadarff(arff_path)
    return _decode_frame(pd.DataFrame(data))


def moa_electricity(*, max_rows: int | None = 50_000, dim: int = 128) -> StreamData:
    """MOA Electricity stream from the NSW electricity market."""

    cache = PROCESSED_DIR / f"moa_electricity_online_v2_n{max_rows}_d{dim}.npz"
    if cache.exists():
        data = np.load(cache, allow_pickle=True)
        return _stream_from_cache(data)
    df = _load_arff_zip(ELEC_URL, "elecNormNew.arff.zip", ".arff")
    label_col = "class" if "class" in df.columns else df.columns[-1]
    numeric_cols = [c for c in df.columns if c != label_col]
    X, y = _hashed_frame(df, label_col=label_col, positive="UP", numeric_cols=numeric_cols, dim=dim, max_rows=max_rows)
    stream = StreamData(
        name="real_moa_electricity",
        X=X,
        y=y,
        weak_type="linear",
        gamma_hint=None,
        description=(
            "MOA Electricity/ELEC stream: NSW electricity-market price movement "
            "relative to a moving average, in chronological market order."
        ),
    )
    _save_stream(cache, stream)
    return stream


def moa_airlines(*, max_rows: int | None = 50_000, dim: int = 128) -> StreamData:
    """MOA Airlines delay stream."""

    cache = PROCESSED_DIR / f"moa_airlines_online_v2_n{max_rows}_d{dim}.npz"
    if cache.exists():
        data = np.load(cache, allow_pickle=True)
        return _stream_from_cache(data)
    df = _load_arff_zip(AIRLINES_URL, "airlines.arff.zip", ".arff")
    label_col = "Delay" if "Delay" in df.columns else df.columns[-1]
    X, y = _hashed_frame(df, label_col=label_col, positive="1", numeric_cols=None, dim=dim, max_rows=max_rows)
    stream = StreamData(
        name="real_moa_airlines",
        X=X,
        y=y,
        weak_type="linear",
        gamma_hint=None,
        description="MOA Airlines stream: predict whether a flight will be delayed from scheduled-flight attributes.",
    )
    _save_stream(cache, stream)
    return stream


def occupancy_detection(*, max_rows: int | None = 50_000, dim: int = 128) -> StreamData:
    """UCI office-occupancy measurements merged in timestamp order."""

    cache = PROCESSED_DIR / f"uci_occupancy_online_v2_n{max_rows}_d{dim}.npz"
    if cache.exists():
        data = np.load(cache, allow_pickle=True)
        return _stream_from_cache(data)

    zip_path = RAW_DIR / "occupancy_detection.zip"
    _download(OCCUPANCY_URL, zip_path)
    frames = []
    for suffix in ("datatest.txt", "datatraining.txt", "datatest2.txt"):
        frames.append(pd.read_csv(_zip_member(zip_path, suffix)))
    df = pd.concat(frames, ignore_index=True)
    df["_timestamp"] = pd.to_datetime(df["date"], errors="raise")
    df = df.sort_values("_timestamp", kind="stable").reset_index(drop=True)
    timestamps = df["_timestamp"]

    minutes = timestamps.dt.hour.to_numpy(dtype=float) * 60.0 + timestamps.dt.minute.to_numpy(dtype=float)
    weekday = timestamps.dt.dayofweek.to_numpy(dtype=float)
    df["time_sin"] = np.sin(2.0 * math.pi * minutes / (24.0 * 60.0))
    df["time_cos"] = np.cos(2.0 * math.pi * minutes / (24.0 * 60.0))
    df["weekday_sin"] = np.sin(2.0 * math.pi * weekday / 7.0)
    df["weekday_cos"] = np.cos(2.0 * math.pi * weekday / 7.0)
    numeric_cols = [
        "Temperature",
        "Humidity",
        "Light",
        "CO2",
        "HumidityRatio",
        "time_sin",
        "time_cos",
        "weekday_sin",
        "weekday_cos",
    ]
    X, y = _hashed_frame(
        df,
        label_col="Occupancy",
        positive=1,
        numeric_cols=numeric_cols,
        dim=dim,
        max_rows=max_rows,
        drop_cols={"date", "_timestamp"},
    )
    stream = StreamData(
        name="real_uci_occupancy",
        X=X,
        y=y,
        weak_type="linear",
        gamma_hint=None,
        description=(
            "UCI Occupancy Detection stream: minute-level office sensor "
            "measurements ordered by their recorded timestamps, with occupancy "
            "labels obtained from timestamped camera images."
        ),
    )
    _save_stream(cache, stream)
    return stream


def insects_drift(*, variant: str, max_rows: int | None = None) -> StreamData:
    """INSECTS optical-sensor stream with a fixed binary species target.

    The public benchmark has six labels: both sexes of three mosquito species.
    We predict whether an observation is Aedes albopictus (either sex), leaving
    the source order unchanged.  This target is fixed across all drift variants.
    """

    if variant not in INSECTS_VARIANTS:
        raise ValueError(f"unknown INSECTS variant: {variant}")
    config = INSECTS_VARIANTS[variant]
    cache = PROCESSED_DIR / f"insects_{variant}_albopictus_online_v2_n{max_rows}.npz"
    if cache.exists():
        data = np.load(cache, allow_pickle=True)
        return _stream_from_cache(data)

    raw_path = RAW_DIR / str(config["filename"])
    _download(str(config["url"]), raw_path)
    frame = pd.read_csv(raw_path, header=None)
    if frame.shape[1] != 34:
        raise ValueError(
            f"expected 33 features and one label in {raw_path}, got {frame.shape[1]} columns"
        )
    if max_rows is not None:
        frame = frame.iloc[:max_rows].copy()

    X = frame.iloc[:, :33].to_numpy(dtype=float)
    labels = frame.iloc[:, 33].to_numpy(dtype=int)
    observed = set(int(value) for value in np.unique(labels))
    expected = {2, 3, 4, 5, 11, 12}
    if not observed <= expected:
        raise ValueError(f"unexpected INSECTS class labels: {sorted(observed)}")
    y = np.where(np.isin(labels, list(INSECTS_ALBOPICTUS_LABELS)), 1, -1).astype(int)

    X = _online_standardize_matrix(X)
    X = np.column_stack([np.ones(X.shape[0], dtype=float), X])
    X = _normalize_rows(X)

    stream = StreamData(
        name=str(config["stream_name"]),
        X=X,
        y=y,
        weak_type="linear",
        gamma_hint=None,
        description=(
            "INSECTS optical-sensor benchmark with "
            f"{config['description']}. The binary target is Aedes albopictus "
            "(either sex) versus Aedes aegypti or Culex quinquefasciatus."
        ),
    )
    _save_stream(cache, stream)
    return stream


def _save_stream(path: Path, stream: StreamData) -> None:
    _ensure_dirs()
    np.savez_compressed(
        path,
        name=stream.name,
        X=stream.X,
        y=stream.y,
        weak_type=stream.weak_type,
        description=stream.description,
        gamma_hint=np.nan if stream.gamma_hint is None else float(stream.gamma_hint),
    )


def _stream_from_cache(data: np.lib.npyio.NpzFile) -> StreamData:
    gamma = float(data["gamma_hint"])
    return StreamData(
        name=str(data["name"]),
        X=np.asarray(data["X"], dtype=float),
        y=np.asarray(data["y"], dtype=int),
        weak_type=str(data["weak_type"]),
        description=str(data["description"]),
        gamma_hint=None if np.isnan(gamma) else gamma,
    )
