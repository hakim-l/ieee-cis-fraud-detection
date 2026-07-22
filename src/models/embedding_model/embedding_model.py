from src.models.base_model import BaseModel
from tensorflow.keras.layers import Embedding, Dense, Input
from tensorflow.keras.models import Model


class EmbeddingModel(BaseModel):
    def __init__(self, vocab_size, embedding_dim):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.embedding_model, self.model= self.build_model()

    def _build_model(self):
        input_layer = Input(
            shape=(None,), 
            dtype="int32", 
            name= 'input_layer'
            )
        embedding_layer = Embedding(input_dim=self.vocab_size, output_dim=self.embedding_dim)(input_layer)
        output_layer = Dense(1, activation="sigmoid")(embedding_layer)
        embedding_model= Model(
            inputs= input_layer,
            outputs= embedding_layer
        )

        model_with_head= Model(
            inputs= input_layer,
            outputs= output_layer
        )

        return embedding_model, model_with_head


    def fit(self, X, y, epochs=10, batch_size=32):
        self.model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
        self.model.fit(X, y, epochs=epochs, batch_size=batch_size)

    def predict(self, X):
        return self.model.predict(X)

    def evaluate(self, X, y):
        return self.model.evaluate(X, y)

    def save(self, file_path):
        self.model.save(file_path)

    def load(self, file_path):
        self.model = tf.keras.models.load_model(file_path)