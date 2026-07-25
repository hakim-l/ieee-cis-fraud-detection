"""Compute TabTransformer encoder features and append to processed parquet datasets.

This script:
- loads preprocessing metadata and the trained TabTransformer model
  from models/tab_transformer/
- discovers parquet files under data/processed/ (recursively)
- for each parquet file, maps categorical features to integer indices using
  the saved vocabularies, prepares numeric feature arrays, runs the encoder,
  mean-pools encoder token embeddings to a single vector per row, and
  appends embedding columns to the table.
- writes augmented parquet files to data/processed_with_encoder_features/
  preserving the input directory structure.

Usage: run as a module or call main() from CLI.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import tensorflow as tf

from src.config import MODELS_DIR, PROCESSED_DATA_DIR
from src.utils.file_utils import list_files


DEFAULT_OUTPUT_DIR = Path("data/processed_with_encoder_features")


def _load_preprocessing(model_dir: Path) -> Dict:
    preprocessing_path = model_dir / "preprocessing.json"
    if not preprocessing_path.exists():
        raise FileNotFoundError(f"Missing preprocessing metadata: {preprocessing_path}")
    return json.loads(preprocessing_path.read_text(encoding="utf-8"))


def _load_training_config(model_dir: Path) -> Dict:
    """Load training hyperparameters from training_config.json."""
    training_config_path = model_dir / "training_config.json"
    if not training_config_path.exists():
        raise FileNotFoundError(
            f"Missing training_config.json: {training_config_path}. "
            "Please train the model first or ensure training_config.json was saved."
        )
    return json.loads(training_config_path.read_text(encoding="utf-8"))


def _map_categorical_column(series: pd.Series, vocabulary: List[str], unk_index: int = 0) -> np.ndarray:
    """Map a pandas Series to integer indices using the provided vocabulary.
    Unknowns and NA map to unk_index. Values are converted to strings before lookup.
    """
    vocab_index = {token: idx for idx, token in enumerate(vocabulary)}

    def lookup(val):
        if pd.isna(val):
            return unk_index
        token = str(val)
        return vocab_index.get(token, unk_index)

    return series.map(lookup).to_numpy(dtype=np.int32)


def _prepare_model_inputs(
    df: pd.DataFrame, categorical_names: List[str], numeric_names: List[str], vocabularies: Dict[str, List[str]]
) -> Dict[str, np.ndarray]:
    # Build categorical matrix (n_rows, n_cat)
    categorical_arrays = []
    for name in categorical_names:
        if name in df.columns:
            vocab = vocabularies.get(name, ["[UNK]"])
            arr = _map_categorical_column(df[name], vocab, unk_index=0)
        else:
            # If missing column, fill with unk
            arr = np.full(len(df), 0, dtype=np.int32)
        categorical_arrays.append(arr)

    if categorical_arrays:
        categorical_matrix = np.stack(categorical_arrays, axis=1)  # shape (n_rows, n_cat)
    else:
        categorical_matrix = np.zeros((len(df), 0), dtype=np.int32)

    # Numeric matrix (n_rows, n_numeric)
    numeric_arrays = []
    for name in numeric_names:
        if name in df.columns:
            numeric_arrays.append(pd.to_numeric(df[name], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32))
        else:
            numeric_arrays.append(np.zeros(len(df), dtype=np.float32))

    if numeric_arrays:
        numeric_matrix = np.stack(numeric_arrays, axis=1)  # shape (n_rows, n_numeric)
    else:
        numeric_matrix = np.zeros((len(df), 0), dtype=np.float32)

    return {
        "categorical_features": categorical_matrix,
        "numeric_features": numeric_matrix,
    }


def process_parquet_file(
    parquet_path: Path,
    encoder: tf.keras.Model,
    categorical_names: List[str],
    numeric_names: List[str],
    vocabularies: Dict[str, List[str]],
    output_base: Path,
    compression: str = "snappy",
) -> Path:
    df = pd.read_parquet(parquet_path)
    if df.empty:
        # write empty file to preserve structure
        out_path = output_base / parquet_path.relative_to(PROCESSED_DATA_DIR)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out_path, index=False, compression=compression)
        return out_path

    inputs = _prepare_model_inputs(df, categorical_names, numeric_names, vocabularies)

    # Run encoder in batches to avoid OOM
    batch_size = 4096
    n = len(df)
    pooled_embeddings = []
    for start in range(0, n, batch_size):
        end = min(n, start + batch_size)
        batch_inputs = {
            "categorical_features": inputs["categorical_features"][start:end],
            "numeric_features": inputs["numeric_features"][start:end],
        }
        # encoder output shape: (batch, num_tokens, embedding_dim)
        encoder_output = encoder(batch_inputs, training=False)
        # mean pool across token axis (axis=1)
        pooled = np.mean(encoder_output, axis=1)
        pooled_embeddings.append(pooled)

    pooled_embeddings = np.vstack(pooled_embeddings)  # (n_rows, embedding_dim)
    emb_dim = pooled_embeddings.shape[1]

    # Append embedding columns
    for i in range(emb_dim):
        col_name = f"encoder_emb_{i}"
        df[col_name] = pooled_embeddings[:, i]

    # Write augmented parquet preserving relative path
    out_path = output_base / parquet_path.relative_to(PROCESSED_DATA_DIR)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False, compression=compression)
    return out_path


def run_encoder_feature_pipeline(
    source_dir: Path = PROCESSED_DATA_DIR,
    model_dir: Path = MODELS_DIR / "tab_transformer",
    output_dir: Path | None = None,
) -> List[Path]:
    if output_dir is None:
        output_dir = DEFAULT_OUTPUT_DIR
    source_dir = Path(source_dir)
    model_dir = Path(model_dir)
    output_dir = Path(output_dir)

    if not source_dir.exists():
        raise FileNotFoundError(f"Source processed data directory not found: {source_dir}")

    # Load preprocessing metadata and model
    preprocessing = _load_preprocessing(model_dir)
    categorical_names = preprocessing.get("categorical_feature_names", [])
    numeric_names = preprocessing.get("numeric_feature_names", [])
    vocabularies = preprocessing.get("vocabularies", {})

    model_path = model_dir / "model.keras"
    if not model_path.exists():
        # fallback to best_model.keras
        model_path = model_dir / "best_model.keras"
    if not model_path.exists():
        raise FileNotFoundError(f"TabTransformer model not found in {model_dir}")

    # Load training hyperparameters from training_config.json
    training_config = _load_training_config(model_dir)
    embedding_dim = training_config["embedding_dim"]
    num_heads = training_config["num_heads"]
    num_transformer_blocks = training_config["num_transformer_blocks"]
    feedforward_dim = training_config["feedforward_dim"]
    mlp_hidden_units = tuple(training_config["mlp_hidden_units"])
    dropout_rate = training_config["dropout_rate"]

    # categorical cardinalities from preprocessing
    categorical_cardinalities = preprocessing.get("categorical_cardinalities")
    if not categorical_cardinalities:
        # derive from vocabularies
        categorical_cardinalities = [len(vocab) for vocab in vocabularies.values()]

    # Build TabTransformer instance and load weights
    from src.models.encoder.tab_transformer import TabTransformer

    model_instance = TabTransformer(
        categorical_cardinalities=categorical_cardinalities,
        num_numeric_features=len(numeric_names),
        embedding_dim=embedding_dim,
        num_heads=num_heads,
        num_transformer_blocks=num_transformer_blocks,
        feedforward_dim=feedforward_dim,
        mlp_hidden_units=mlp_hidden_units,
        dropout_rate=dropout_rate,
    )

    # Extract weights from .keras archive
    import zipfile
    import tempfile

    with zipfile.ZipFile(str(model_path), "r") as z:
        # Extract weights to a temporary file
        weight_member = None
        for name in z.namelist():
            if name.endswith(".h5") or name.endswith(".weights.h5"):
                weight_member = name
                break
        if weight_member is None:
            raise FileNotFoundError(f"No weights (.h5) file found inside {model_path}")

        tmpdir = tempfile.mkdtemp()
        z.extract(weight_member, path=tmpdir)
        weights_path = Path(tmpdir) / Path(weight_member).name

    # Load HDF5 weights
    try:
        model_instance.model.load_weights(str(weights_path))
    except Exception as exc:
        raise RuntimeError(
            "Failed to load model weights. Ensure the weights file inside the .keras archive is present."
        ) from exc

    encoder = model_instance.encoder

    # Find all parquet files under source_dir recursively
    parquet_files = list_files(source_dir, extension=".parquet", recursive=True)
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found under {source_dir}")

    written_files = []
    for parquet_file in parquet_files:
        print(f"Processing {parquet_file}...")
        out_path = process_parquet_file(
            parquet_file,
            encoder=encoder,
            categorical_names=categorical_names,
            numeric_names=numeric_names,
            vocabularies=vocabularies,
            output_base=output_dir,
            compression="snappy",
        )
        written_files.append(out_path)
        print(f"Wrote augmented parquet to {out_path}")

    return written_files


def main() -> None:
    written = run_encoder_feature_pipeline()
    print(f"Completed. Wrote {len(written)} files.")


if __name__ == "__main__":
    main()
