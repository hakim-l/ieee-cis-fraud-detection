from __future__ import annotations

from pathlib import Path
from typing import Iterable

import tensorflow as tf

from src.models.base import BaseModel


@tf.keras.utils.register_keras_serializable(package="ieee_cis_fraud_detection")
class TransformerEncoderBlock(tf.keras.layers.Layer):
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        feedforward_dim: int,
        dropout_rate: float = 0.1,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.feedforward_dim = feedforward_dim
        self.dropout_rate = dropout_rate

        self.attention = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embedding_dim // num_heads,
            dropout=dropout_rate,
            name="self_attention",
        )
        self.attention_dropout = tf.keras.layers.Dropout(dropout_rate, name="attention_dropout")
        self.attention_norm = tf.keras.layers.LayerNormalization(
            epsilon=1e-6,
            name="attention_norm",
        )
        self.feedforward = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(feedforward_dim, activation="gelu", name="ffn_dense_1"),
                tf.keras.layers.Dropout(dropout_rate, name="ffn_dropout"),
                tf.keras.layers.Dense(embedding_dim, name="ffn_dense_2"),
            ],
            name="feedforward_network",
        )
        self.feedforward_dropout = tf.keras.layers.Dropout(
            dropout_rate,
            name="feedforward_output_dropout",
        )
        self.feedforward_norm = tf.keras.layers.LayerNormalization(
            epsilon=1e-6,
            name="feedforward_norm",
        )

    def call(self, inputs: tf.Tensor, training: bool | None = None) -> tf.Tensor:
        attention_output = self.attention(inputs, inputs, training=training)
        attention_output = self.attention_dropout(attention_output, training=training)
        attention_residual = self.attention_norm(inputs + attention_output)

        feedforward_output = self.feedforward(attention_residual, training=training)
        feedforward_output = self.feedforward_dropout(feedforward_output, training=training)
        return self.feedforward_norm(attention_residual + feedforward_output)

    def get_config(self) -> dict:
        config = super().get_config()
        config.update(
            {
                "embedding_dim": self.embedding_dim,
                "num_heads": self.num_heads,
                "feedforward_dim": self.feedforward_dim,
                "dropout_rate": self.dropout_rate,
            }
        )
        return config


class TabTransformer(BaseModel):
    def __init__(
        self,
        categorical_cardinalities: Iterable[int],
        num_numeric_features: int = 0,
        embedding_dim: int = 32,
        num_heads: int = 4,
        num_transformer_blocks: int = 2,
        feedforward_dim: int = 128,
        mlp_hidden_units: Iterable[int] = (64, 32),
        dropout_rate: float = 0.1,
    ) -> None:
        self.categorical_cardinalities = tuple(categorical_cardinalities)
        self.num_numeric_features = num_numeric_features
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.num_transformer_blocks = num_transformer_blocks
        self.feedforward_dim = feedforward_dim
        self.mlp_hidden_units = tuple(mlp_hidden_units)
        self.dropout_rate = dropout_rate

        self._validate_configuration()
        self.encoder, self.prediction_head, self.model = self._build_models()

    def _validate_configuration(self) -> None:
        if not self.categorical_cardinalities:
            raise ValueError("categorical_cardinalities must contain at least one categorical feature")
        if any(cardinality <= 1 for cardinality in self.categorical_cardinalities):
            raise ValueError("categorical_cardinalities values must be greater than 1")
        if self.num_numeric_features < 0:
            raise ValueError("num_numeric_features must be greater than or equal to 0")
        if self.embedding_dim <= 0:
            raise ValueError("embedding_dim must be greater than 0")
        if self.num_heads <= 0:
            raise ValueError("num_heads must be greater than 0")
        if self.embedding_dim % self.num_heads != 0:
            raise ValueError("embedding_dim must be divisible by num_heads")
        if self.num_transformer_blocks <= 0:
            raise ValueError("num_transformer_blocks must be greater than 0")
        if self.feedforward_dim <= 0:
            raise ValueError("feedforward_dim must be greater than 0")

    def _build_models(self) -> tuple[tf.keras.Model, tf.keras.Model, tf.keras.Model]:
        categorical_inputs = tf.keras.Input(
            shape=(len(self.categorical_cardinalities),),
            dtype=tf.int32,
            name="categorical_features",
        )
        numeric_inputs = tf.keras.Input(
            shape=(self.num_numeric_features,),
            dtype=tf.float32,
            name="numeric_features",
        )

        combined_tokens = self._build_input_representation(categorical_inputs, numeric_inputs)
        encoder_outputs = combined_tokens

        for block_index in range(self.num_transformer_blocks):
            encoder_outputs = TransformerEncoderBlock(
                embedding_dim=self.embedding_dim,
                num_heads=self.num_heads,
                feedforward_dim=self.feedforward_dim,
                dropout_rate=self.dropout_rate,
                name=f"transformer_encoder_block_{block_index + 1}",
            )(encoder_outputs)

        encoder = tf.keras.Model(
            inputs={
                "categorical_features": categorical_inputs,
                "numeric_features": numeric_inputs,
            },
            outputs=encoder_outputs,
            name="tab_transformer_encoder",
        )

        prediction_inputs = tf.keras.Input(
            shape=(len(self.categorical_cardinalities) + self.num_numeric_features, self.embedding_dim),
            dtype=tf.float32,
            name="contextual_tokens",
        )
        prediction_outputs = self._build_prediction_head(prediction_inputs)
        prediction_head = tf.keras.Model(
            inputs=prediction_inputs,
            outputs=prediction_outputs,
            name="tab_transformer_prediction_head",
        )

        full_outputs = prediction_head(encoder.output)
        model = tf.keras.Model(
            inputs=encoder.inputs,
            outputs=full_outputs,
            name="tab_transformer",
        )

        return encoder, prediction_head, model

    def _build_input_representation(
        self,
        categorical_inputs: tf.keras.KerasTensor,
        numeric_inputs: tf.keras.KerasTensor,
    ) -> tf.Tensor:
        categorical_tokens = []
        total_columns = len(self.categorical_cardinalities) + self.num_numeric_features

        for index, cardinality in enumerate(self.categorical_cardinalities):
            categorical_feature = tf.keras.layers.Lambda(
                lambda x, col=index: x[:, col],
                output_shape=(),
                name=f"categorical_feature_{index}_slice",
            )(categorical_inputs)
            categorical_embedding = tf.keras.layers.Embedding(
                input_dim=cardinality,
                output_dim=self.embedding_dim,
                name=f"categorical_embedding_{index}",
            )(categorical_feature)
            column_embedding = tf.keras.layers.Embedding(
                input_dim=total_columns,
                output_dim=self.embedding_dim,
                name=f"categorical_column_embedding_{index}",
            )(tf.constant([self.num_numeric_features + index], dtype=tf.int32))
            combined_embedding = tf.keras.layers.Add(name=f"categorical_token_with_column_{index}")(
                [categorical_embedding, column_embedding]
            )
            categorical_tokens.append(combined_embedding)

        categorical_token_tensor = tf.keras.layers.Lambda(
            lambda tensors: tf.stack(tensors, axis=1),
            output_shape=(len(self.categorical_cardinalities), self.embedding_dim),
            name="categorical_tokens",
        )(categorical_tokens)

        if self.num_numeric_features == 0:
            return categorical_token_tensor

        numeric_tokens = []
        for index in range(self.num_numeric_features):
            numeric_feature = tf.keras.layers.Lambda(
                lambda x, col=index: x[:, col : col + 1],
                output_shape=(1,),
                name=f"numeric_feature_{index}_slice",
            )(numeric_inputs)
            projected_numeric = tf.keras.Sequential(
                [
                    tf.keras.layers.Dense(self.embedding_dim, name=f"numeric_projection_{index}_dense"),
                    tf.keras.layers.LayerNormalization(
                        epsilon=1e-6,
                        name=f"numeric_projection_{index}_norm",
                    ),
                ],
                name=f"numeric_projection_{index}",
            )(numeric_feature)
            column_embedding = tf.keras.layers.Embedding(
                input_dim=total_columns,
                output_dim=self.embedding_dim,
                name=f"numeric_column_embedding_{index}",
            )(tf.constant([index], dtype=tf.int32))
            combined_projection = tf.keras.layers.Add(name=f"numeric_token_with_column_{index}")(
                [projected_numeric, column_embedding]
            )
            numeric_tokens.append(combined_projection)

        numeric_token_tensor = tf.keras.layers.Lambda(
            lambda tensors: tf.stack(tensors, axis=1),
            output_shape=(self.num_numeric_features, self.embedding_dim),
            name="numeric_tokens",
        )(numeric_tokens)

        return tf.keras.layers.Concatenate(axis=1, name="tabular_tokens")(
            [numeric_token_tensor, categorical_token_tensor]
        )

    def _build_prediction_head(self, contextual_tokens: tf.keras.KerasTensor) -> tf.Tensor:
        pooled_tokens = tf.keras.layers.GlobalAveragePooling1D(name="mean_pooling")(contextual_tokens)
        x = pooled_tokens

        for layer_index, hidden_units in enumerate(self.mlp_hidden_units):
            x = tf.keras.layers.Dense(
                hidden_units,
                activation="gelu",
                name=f"prediction_dense_{layer_index + 1}",
            )(x)
            x = tf.keras.layers.Dropout(
                self.dropout_rate,
                name=f"prediction_dropout_{layer_index + 1}",
            )(x)

        return tf.keras.layers.Dense(1, activation="sigmoid", name="fraud_probability")(x)

    def compile(self, **kwargs) -> None:
        if "optimizer" not in kwargs:
            kwargs["optimizer"] = tf.keras.optimizers.Adam()
        if "loss" not in kwargs:
            kwargs["loss"] = tf.keras.losses.BinaryCrossentropy()
        self.model.compile(**kwargs)

    def fit(self, X, y, **kwargs):
        if not self._is_compiled():
            self.compile(metrics=["accuracy"])
        return self.model.fit(X, y, **kwargs)

    def predict(self, X, **kwargs):
        return self.model.predict(X, **kwargs)

    def evaluate(self, X, y, **kwargs):
        if not self._is_compiled():
            self.compile(metrics=["accuracy"])
        return self.model.evaluate(X, y, **kwargs)

    def save(self, file_path) -> None:
        self.model.save(self._normalize_save_path(file_path))

    def save_encoder(self, file_path) -> None:
        self.encoder.save(self._normalize_save_path(file_path))

    def save_prediction_head(self, file_path) -> None:
        self.prediction_head.save(self._normalize_save_path(file_path))

    def load(self, file_path):
        loaded_model = tf.keras.models.load_model(self._normalize_save_path(file_path))
        self._attach_loaded_models(loaded_model)
        return self

    @classmethod
    def load_full_model(cls, file_path) -> "TabTransformer":
        loaded_model = tf.keras.models.load_model(cls._normalize_save_path(file_path))
        instance = cls.__new__(cls)
        instance._attach_loaded_models(loaded_model)
        return instance

    @classmethod
    def load_encoder_model(cls, file_path) -> tf.keras.Model:
        return tf.keras.models.load_model(cls._normalize_save_path(file_path))

    def _attach_loaded_models(self, loaded_model: tf.keras.Model) -> None:
        self.model = loaded_model
        self.encoder = loaded_model.get_layer("tab_transformer_encoder")
        self.prediction_head = loaded_model.get_layer("tab_transformer_prediction_head")

        self.categorical_cardinalities = tuple(
            layer.input_dim
            for layer in self.encoder.layers
            if isinstance(layer, tf.keras.layers.Embedding)
            and layer.name.startswith("categorical_embedding_")
        )
        numeric_input = next(
            tensor for tensor in self.encoder.inputs if tensor.name.startswith("numeric_features")
        )
        self.num_numeric_features = int(numeric_input.shape[-1] or 0)
        self.embedding_dim = int(self.encoder.output.shape[-1])
        self.num_heads = self._infer_num_heads()
        self.num_transformer_blocks = len(
            [
                layer
                for layer in self.encoder.layers
                if isinstance(layer, TransformerEncoderBlock)
            ]
        )
        self.feedforward_dim = self._infer_feedforward_dim()
        self.mlp_hidden_units = tuple(
            layer.units
            for layer in self.prediction_head.layers
            if isinstance(layer, tf.keras.layers.Dense) and layer.name.startswith("prediction_dense_")
        )
        self.dropout_rate = self._infer_dropout_rate()

        # Access the inputs to preserve the public input names after load.
        # _ = categorical_input, numeric_input

    def _infer_num_heads(self) -> int:
        for layer in self.encoder.layers:
            if isinstance(layer, TransformerEncoderBlock):
                return layer.num_heads
        return 0

    def _infer_feedforward_dim(self) -> int:
        for layer in self.encoder.layers:
            if isinstance(layer, TransformerEncoderBlock):
                return layer.feedforward_dim
        return 0

    def _infer_dropout_rate(self) -> float:
        for layer in self.prediction_head.layers:
            if isinstance(layer, tf.keras.layers.Dropout):
                return float(layer.rate)
        for layer in self.encoder.layers:
            if isinstance(layer, TransformerEncoderBlock):
                return float(layer.dropout_rate)
        return 0.0

    def _is_compiled(self) -> bool:
        return self.model.optimizer is not None

    @staticmethod
    def _normalize_save_path(file_path) -> str:
        return str(Path(file_path))
