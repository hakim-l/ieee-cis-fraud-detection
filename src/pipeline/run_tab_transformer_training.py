"""Train the TabTransformer model from processed TFRecord datasets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import tensorflow as tf

from src.config import (
    CATEGORICAL_FEATURES_LIST,
    MODELS_DIR,
    PROCESSED_TF_RECORD_DIR,
    TARGET_COLUMN,
    USED_FEATURE_NAMES,
)
from src.models.encoder.tab_transformer import TabTransformer


AUTOTUNE = tf.data.AUTOTUNE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train the TabTransformer model from TFRecord files generated in "
            "data/processed_tf_record."
        )
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from existing weights/checkpoint in model-dir.",
    )
    parser.add_argument(
        "--no-checkpoints",
        action="store_true",
        help="Disable saving epoch checkpoints (saved to model-dir/checkpoints by default).",
    )

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=PROCESSED_TF_RECORD_DIR,
        help="Base directory containing processed TFRecord splits.",
    )
    parser.add_argument(
        "--train-split",
        default="train",
        help="Split name to use for training. Defaults to 'train'.",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=MODELS_DIR / "tab_transformer",
        help="Directory where model artifacts will be written.",
    )
    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=0.2,
        help="Fraction of the training split reserved for validation.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=512,
        help="Training batch size.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Maximum number of training epochs.",
    )
    parser.add_argument(
        "--shuffle-buffer-size",
        type=int,
        default=8192,
        help="Shuffle buffer size for the training dataset.",
    )
    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=32,
        help="Embedding dimension for categorical and numeric tokens.",
    )
    parser.add_argument(
        "--num-heads",
        type=int,
        default=4,
        help="Number of attention heads.",
    )
    parser.add_argument(
        "--num-transformer-blocks",
        type=int,
        default=2,
        help="Number of transformer encoder blocks.",
    )
    parser.add_argument(
        "--feedforward-dim",
        type=int,
        default=128,
        help="Hidden size of the transformer feedforward network.",
    )
    parser.add_argument(
        "--mlp-hidden-units",
        type=int,
        nargs="+",
        default=(128, 64),
        help="Hidden units for the prediction head MLP.",
    )
    parser.add_argument(
        "--dropout-rate",
        type=float,
        default=0.1,
        help="Dropout rate used across the model.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="Adam learning rate.",
    )
    parser.add_argument(
        "--loss",
        choices=("binary_crossentropy", "focal_loss"),
        default="binary_crossentropy",
        help="Loss function used to train the model.",
    )
    parser.add_argument(
        "--focal-loss-alpha",
        type=float,
        default=0.25,
        help="Alpha parameter when focal loss is enabled.",
    )
    parser.add_argument(
        "--focal-loss-gamma",
        type=float,
        default=2.0,
        help="Gamma parameter when focal loss is enabled.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for deterministic shuffling.",
    )
    parser.add_argument(
        "--cache-dataset",
        action="store_true",
        help="Cache train and validation datasets in memory after preprocessing.",
    )
    return parser.parse_args()


class TabTransformerTFRecordTrainer:
    """Train TabTransformer from serialized TFRecord features."""

    def __init__(
        self,
        data_dir: str | Path = PROCESSED_TF_RECORD_DIR,
        train_split: str = "train",
        model_dir: str | Path = MODELS_DIR / "tab_transformer",
        validation_fraction: float = 0.2,
        batch_size: int = 512,
        epochs: int = 20,
        shuffle_buffer_size: int = 8192,
        embedding_dim: int = 8,
        num_heads: int = 4,
        num_transformer_blocks: int = 2,
        feedforward_dim: int = 16,
        mlp_hidden_units: tuple[int, ...] = (16, 8),
        dropout_rate: float = 0.1,
        learning_rate: float = 1e-3,
        loss: str = "binary_crossentropy",
        focal_loss_alpha: float = 0.25,
        focal_loss_gamma: float = 2.0,
        random_seed: int = 42,
        cache_dataset: bool = False,
    ,
        resume: bool = False,
        save_checkpoints: bool = True,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.train_split = train_split
        self.model_dir = Path(model_dir)
        self.validation_fraction = validation_fraction
        self.batch_size = batch_size
        self.epochs = epochs
        self.shuffle_buffer_size = shuffle_buffer_size
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.num_transformer_blocks = num_transformer_blocks
        self.feedforward_dim = feedforward_dim
        self.mlp_hidden_units = tuple(mlp_hidden_units)
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.loss = loss
        self.focal_loss_alpha = focal_loss_alpha
        self.focal_loss_gamma = focal_loss_gamma
        self.random_seed = random_seed
        self.cache_dataset = cache_dataset
        self.resume = resume
        self.save_checkpoints = save_checkpoints

        self.split_dir = self.data_dir / self.train_split
        self.metadata = self._load_metadata()
        self.metadata_features_by_name = {
            feature["name"]: feature for feature in self.metadata["features"]
        }
        self.feature_spec = self._build_feature_spec()
        self.categorical_feature_names = self._resolve_categorical_feature_names()
        self.numeric_feature_names = self._resolve_numeric_feature_names()
        self.lookup_layers = self._build_lookup_layers()
        self.categorical_cardinalities = self._resolve_categorical_cardinalities()

    def train(self) -> tuple[tf.keras.callbacks.History, dict[str, float]]:
        self._validate_configuration()
        tf.keras.utils.set_random_seed(self.random_seed)
        self.model_dir.mkdir(parents=True, exist_ok=True)

        train_dataset, validation_dataset = self._build_training_datasets()

        model = TabTransformer(
            categorical_cardinalities=self.categorical_cardinalities,
            num_numeric_features=len(self.numeric_feature_names),
            embedding_dim=self.embedding_dim,
            num_heads=self.num_heads,
            num_transformer_blocks=self.num_transformer_blocks,
            feedforward_dim=self.feedforward_dim,
            mlp_hidden_units=self.mlp_hidden_units,
            dropout_rate=self.dropout_rate,
            loss=self.loss,
            focal_loss_alpha=self.focal_loss_alpha,
            focal_loss_gamma=self.focal_loss_gamma,
        )

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            metrics=[
                tf.keras.metrics.BinaryAccuracy(name="accuracy"),
                tf.keras.metrics.AUC(name="auc"),
                tf.keras.metrics.Precision(name="precision"),
                tf.keras.metrics.Recall(name="recall"),
            ],
        )

        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_auc",
                mode="max",
                patience=3,
                restore_best_weights=True,
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=str(self.model_dir / "best_model.keras"),
                monitor="val_auc",
                mode="max",
                save_best_only=True,
            ),
        ]

        # Add per-epoch weights-only checkpointing (legacy) if requested
        if getattr(self, "save_checkpoints", True):
            legacy_checkpoint_dir = self.model_dir / "checkpoints_weights"
            legacy_checkpoint_dir.mkdir(parents=True, exist_ok=True)
            callbacks.append(
                tf.keras.callbacks.ModelCheckpoint(
                    filepath=str(legacy_checkpoint_dir / "epoch-{epoch:03d}.weights.h5"),
                    save_weights_only=True,
                    save_freq="epoch",
                )
            )

        # --- tf.train.Checkpoint manager to save model + optimizer + epoch for full resume ---
        checkpoint_dir = self.model_dir / "tf_checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        # epoch variable to store progress
        checkpoint_epoch = tf.Variable(0, dtype=tf.int64)
        ckpt = tf.train.Checkpoint(optimizer=model.model.optimizer, model=model.model, epoch=checkpoint_epoch)
        manager = tf.train.CheckpointManager(ckpt, directory=str(checkpoint_dir), max_to_keep=5)

        initial_epoch = 0
        if getattr(self, "resume", False):
            latest = manager.latest_checkpoint
            if latest:
                try:
                    ckpt.restore(latest).expect_partial()
                    initial_epoch = int(checkpoint_epoch.numpy())
                    print(f"Restored tf checkpoint from {latest}, resuming at epoch {initial_epoch}")
                except Exception as exc:
                    print(f"Failed to restore checkpoint {latest}: {exc}. Training will continue from scratch.")

        class CheckpointManagerCallback(tf.keras.callbacks.Callback):
            def __init__(self, ckpt_obj: tf.train.Checkpoint, mgr: tf.train.CheckpointManager, epoch_var: tf.Variable):
                super().__init__()
                self.ckpt_obj = ckpt_obj
                self.mgr = mgr
                self.epoch_var = epoch_var

            def on_epoch_end(self, epoch, logs=None):
                # epoch is 0-based, store completed epochs
                self.epoch_var.assign(epoch + 1)
                try:
                    path = self.mgr.save()
                    print(f"Saved tf checkpoint: {path}")
                except Exception as exc:
                    print(f"Failed to save tf checkpoint: {exc}")

        # attach checkpoint callback
        callbacks.append(CheckpointManagerCallback(ckpt, manager, checkpoint_epoch))

        # Use initial_epoch to resume training from the correct step
        history = model.fit(
            train_dataset,
            validation_data=validation_dataset,
            epochs=self.epochs,
            initial_epoch=initial_epoch,
            callbacks=callbacks,
            verbose=2,
        )
        evaluation_values = model.model.evaluate(validation_dataset, verbose=0)
        evaluation = {
            metric_name: float(metric_value)
            for metric_name, metric_value in zip(model.model.metrics_names, evaluation_values)
        }

        # Save final model (full archive) and weights separately for easy resume
        model.save(self.model_dir / "model.keras")
        try:
            model.model.save_weights(str(self.model_dir / "model.weights.h5"))
        except Exception:
            # best-effort; warn but don't fail
            print("Warning: failed to save model weights to model.weights.h5")

        self._write_artifacts(history, evaluation)
        return history, evaluation

    def _validate_configuration(self) -> None:
        if not self.split_dir.exists():
            raise FileNotFoundError(f"Training split directory not found: {self.split_dir}")
        if not 0.0 < self.validation_fraction < 1.0:
            raise ValueError("validation_fraction must be between 0 and 1")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be greater than 0")
        if self.epochs <= 0:
            raise ValueError("epochs must be greater than 0")
        if self.loss not in {"binary_crossentropy", "focal_loss"}:
            raise ValueError("loss must be 'binary_crossentropy' or 'focal_loss'")
        if self.focal_loss_alpha <= 0:
            raise ValueError("focal_loss_alpha must be greater than 0")
        if self.focal_loss_gamma < 0:
            raise ValueError("focal_loss_gamma must be greater than or equal to 0")
        if not self.categorical_feature_names:
            raise ValueError("No categorical features were found in the TFRecord metadata")

    def _load_metadata(self) -> dict[str, Any]:
        metadata_path = self.split_dir / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Missing TFRecord metadata file: {metadata_path}")
        with metadata_path.open("r", encoding="utf-8") as file:
            return json.load(file)

    def _build_feature_spec(self) -> dict[str, tf.io.FixedLenFeature]:
        feature_spec: dict[str, tf.io.FixedLenFeature] = {}
        for feature in self.metadata["features"]:
            name = feature["name"]
            tfrecord_type = feature["tfrecord_type"]
            if tfrecord_type == "bytes":
                feature_spec[name] = tf.io.FixedLenFeature([], tf.string, default_value="")
            elif tfrecord_type == "float":
                feature_spec[name] = tf.io.FixedLenFeature([], tf.float32, default_value=0.0)
            elif tfrecord_type == "int64":
                feature_spec[name] = tf.io.FixedLenFeature([], tf.int64, default_value=0)
            else:
                raise TypeError(f"Unsupported TFRecord feature type: {tfrecord_type}")
        return feature_spec

    def _resolve_categorical_feature_names(self) -> list[str]:
        available_features = set(self.metadata_features_by_name)
        return [
            feature_name
            for feature_name in USED_FEATURE_NAMES
            if feature_name in available_features and feature_name in CATEGORICAL_FEATURES_LIST
        ]

    def _resolve_numeric_feature_names(self) -> list[str]:
        available_features = set(self.metadata_features_by_name)
        categorical_features = set(self.categorical_feature_names)
        return [
            feature_name
            for feature_name in USED_FEATURE_NAMES
            if feature_name in available_features and feature_name not in categorical_features
        ]

    def _build_lookup_layers(self) -> dict[str, tf.keras.layers.StringLookup]:
        lookup_layers: dict[str, tf.keras.layers.StringLookup] = {}
        for feature_name in self.categorical_feature_names:
            lookup = tf.keras.layers.StringLookup(
                mask_token=None,
                num_oov_indices=1,
                output_mode="int",
                name=f"{feature_name}_lookup",
            )
            lookup.adapt(self._categorical_value_dataset(feature_name))
            lookup_layers[feature_name] = lookup
        return lookup_layers

    def _categorical_value_dataset(self, feature_name: str) -> tf.data.Dataset:
        return (
            self._base_dataset(training_only=True)
            .map(lambda features, _: features[feature_name], num_parallel_calls=AUTOTUNE)
            .batch(4096)
            .prefetch(AUTOTUNE)
        )

    def _resolve_categorical_cardinalities(self) -> tuple[int, ...]:
        return tuple(
            int(self.lookup_layers[feature_name].vocabulary_size())
            for feature_name in self.categorical_feature_names
        )

    def _build_training_datasets(self) -> tuple[tf.data.Dataset, tf.data.Dataset]:
        train_dataset = self._encoded_dataset(training_only=True)
        validation_dataset = self._encoded_dataset(training_only=False)

        if self.cache_dataset:
            train_dataset = train_dataset.cache()
            validation_dataset = validation_dataset.cache()

        train_dataset = (
            train_dataset.shuffle(
                buffer_size=self.shuffle_buffer_size,
                seed=self.random_seed,
                reshuffle_each_iteration=True,
            )
            .batch(self.batch_size)
            .prefetch(AUTOTUNE)
        )
        validation_dataset = validation_dataset.batch(self.batch_size).prefetch(AUTOTUNE)
        return train_dataset, validation_dataset

    def _encoded_dataset(self, training_only: bool) -> tf.data.Dataset:
        return self._base_dataset(training_only=training_only).map(
            self._encode_example,
            num_parallel_calls=AUTOTUNE,
        )

    def _base_dataset(self, training_only: bool) -> tf.data.Dataset:
        tfrecord_files = sorted(self.split_dir.glob("*.tfrecord"))
        if not tfrecord_files:
            raise FileNotFoundError(f"No TFRecord shards found in {self.split_dir}")

        dataset = tf.data.TFRecordDataset([str(path) for path in tfrecord_files])
        dataset = dataset.map(self._parse_example, num_parallel_calls=AUTOTUNE)
        dataset = dataset.enumerate()
        validation_period = self._validation_period()

        if training_only:
            dataset = dataset.filter(
                lambda index, _: tf.not_equal(tf.math.floormod(index, validation_period), 0)
            )
        else:
            dataset = dataset.filter(
                lambda index, _: tf.equal(tf.math.floormod(index, validation_period), 0)
            )

        return dataset.map(lambda _, example: example, num_parallel_calls=AUTOTUNE)

    def _validation_period(self) -> int:
        return max(2, int(round(1.0 / self.validation_fraction)))

    def _parse_example(self, serialized_example: tf.Tensor) -> tuple[dict[str, tf.Tensor], tf.Tensor]:
        parsed = tf.io.parse_single_example(serialized_example, self.feature_spec)
        label = parsed.pop(TARGET_COLUMN)
        if label.dtype == tf.string:
            label = tf.strings.to_number(label, out_type=tf.float32)
        else:
            label = tf.cast(label, tf.float32)
        return parsed, tf.reshape(label, (1,))

    def _encode_example(
        self,
        features: dict[str, tf.Tensor],
        label: tf.Tensor,
    ) -> tuple[dict[str, tf.Tensor], tf.Tensor]:
        categorical_tensors = [
            tf.cast(self.lookup_layers[feature_name](features[feature_name]), tf.int32)
            for feature_name in self.categorical_feature_names
        ]
        numeric_tensors = [
            tf.cast(features[feature_name], tf.float32)
            for feature_name in self.numeric_feature_names
        ]
        if numeric_tensors:
            numeric_features = tf.stack(numeric_tensors, axis=0)
        else:
            numeric_features = tf.constant([], dtype=tf.float32)

        encoded_features = {
            "categorical_features": tf.stack(categorical_tensors, axis=0),
            "numeric_features": numeric_features,
        }
        return encoded_features, label

    def _write_artifacts(
        self,
        history: tf.keras.callbacks.History,
        evaluation: dict[str, float],
    ) -> None:
        history_path = self.model_dir / "history.json"
        evaluation_path = self.model_dir / "evaluation.json"
        preprocessing_path = self.model_dir / "preprocessing.json"

        history_payload = {
            metric_name: [float(value) for value in values]
            for metric_name, values in history.history.items()
        }
        preprocessing_payload = {
            "data_dir": str(self.data_dir),
            "train_split": self.train_split,
            "categorical_feature_names": self.categorical_feature_names,
            "numeric_feature_names": self.numeric_feature_names,
            "categorical_cardinalities": list(self.categorical_cardinalities),
            "loss": self.loss,
            "focal_loss_alpha": self.focal_loss_alpha,
            "focal_loss_gamma": self.focal_loss_gamma,
            "vocabularies": {
                feature_name: self.lookup_layers[feature_name].get_vocabulary()
                for feature_name in self.categorical_feature_names
            },
        }

        history_path.write_text(json.dumps(history_payload, indent=2), encoding="utf-8")
        evaluation_path.write_text(json.dumps(evaluation, indent=2), encoding="utf-8")
        preprocessing_path.write_text(json.dumps(preprocessing_payload, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    trainer = TabTransformerTFRecordTrainer(
        data_dir=args.data_dir,
        train_split=args.train_split,
        model_dir=args.model_dir,
        validation_fraction=args.validation_fraction,
        batch_size=args.batch_size,
        epochs=args.epochs,
        shuffle_buffer_size=args.shuffle_buffer_size,
        embedding_dim=args.embedding_dim,
        num_heads=args.num_heads,
        num_transformer_blocks=args.num_transformer_blocks,
        feedforward_dim=args.feedforward_dim,
        mlp_hidden_units=tuple(args.mlp_hidden_units),
        dropout_rate=args.dropout_rate,
        learning_rate=args.learning_rate,
        loss=args.loss,
        focal_loss_alpha=args.focal_loss_alpha,
        focal_loss_gamma=args.focal_loss_gamma,
        random_seed=args.random_seed,
        cache_dataset=args.cache_dataset,
        resume=args.resume,
        save_checkpoints=(not args.no_checkpoints),
    )
    _, evaluation = trainer.train()
    print("Validation metrics:")
    print(json.dumps(evaluation, indent=2))


if __name__ == "__main__":
    main()
