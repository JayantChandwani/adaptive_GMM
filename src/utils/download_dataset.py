from __future__ import annotations

from pathlib import Path

import numpy as np
import kagglehub
import pandas as pd
from sklearn.preprocessing import OneHotEncoder


CATEGORICAL_COLS = ["protocol_type", "service", "flag"]
OUTPUT_DIR = Path("data/nsl_kdd")


def _download_raw() -> Path:
    raw_path = Path(kagglehub.dataset_download("hassan06/nslkdd"))
    return raw_path


def _parse_arff(path: Path) -> pd.DataFrame:
    """Parse ARFF manually to avoid scipy's strict nominal-value validation."""
    col_names: list[str] = []
    data_lines: list[str] = []
    in_data = False

    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("%"):
                continue
            if line.lower().startswith("@attribute"):
                # extract column name (second token, strip quotes)
                parts = line.split(None, 2)
                name = parts[1].strip("'\"")
                col_names.append(name)
            elif line.lower().startswith("@data"):
                in_data = True
            elif in_data:
                data_lines.append(line)

    from io import StringIO
    df = pd.read_csv(
        StringIO("\n".join(data_lines)),
        header=None,
        names=col_names,
        quotechar="'",
    )
    # strip stray whitespace from string columns
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].str.strip().str.strip("'\"")
    return df


def _encode(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    label_col = "class"

    y_train = (df_train[label_col] != "normal").astype(int).to_numpy()
    y_test = (df_test[label_col] != "normal").astype(int).to_numpy()

    df_train = df_train.drop(columns=[label_col])
    df_test = df_test.drop(columns=[label_col])

    num_cols = [c for c in df_train.columns if c not in CATEGORICAL_COLS]

    enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    enc.fit(df_train[CATEGORICAL_COLS])

    X_train_cat = enc.transform(df_train[CATEGORICAL_COLS])
    X_test_cat = enc.transform(df_test[CATEGORICAL_COLS])

    X_train_num = df_train[num_cols].to_numpy(dtype=float)
    X_test_num = df_test[num_cols].to_numpy(dtype=float)

    X_train = np.hstack([X_train_num, X_train_cat])
    X_test = np.hstack([X_test_num, X_test_cat])

    return X_train, y_train, X_test, y_test


def load_nsl_kdd(
    output_dir: str | Path = OUTPUT_DIR,
    force_redownload: bool = False,
) -> dict[str, np.ndarray]:
    output_dir = Path(output_dir)
    npz_path = output_dir / "nsl_kdd.npz"

    if npz_path.exists() and not force_redownload:
        with np.load(npz_path) as f:
            return {k: f[k] for k in f.files}

    raw_path = _download_raw()

    train_arff = next(Path(raw_path).glob("KDDTrain+.arff"))
    test_arff = next(Path(raw_path).glob("KDDTest+.arff"))

    df_train = _parse_arff(train_arff)
    df_test = _parse_arff(test_arff)

    X_train, y_train, X_test, y_test = _encode(df_train, df_test)

    # NSL-KDD has no explicit timestamp column. The ARFF row order is the
    # arrival order of the network records, so we expose it as t_train / t_test
    # and the streaming evaluator sorts batches by these values.
    t_train = np.arange(len(X_train), dtype=np.int64)
    t_test = np.arange(len(X_test), dtype=np.int64)

    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        npz_path,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        t_train=t_train,
        t_test=t_test,
    )

    print(f"Saved NSL-KDD to {npz_path}")
    print(f"  X_train: {X_train.shape}, anomalies in train: {y_train.sum()}")
    print(f"  X_test:  {X_test.shape},  anomalies in test:  {y_test.sum()}")

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
        "t_train": t_train,
        "t_test": t_test,
    }


if __name__ == "__main__":
    load_nsl_kdd()