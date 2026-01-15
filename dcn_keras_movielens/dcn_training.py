# DCN (Deep & Cross Network) Training with Keras + Kubetorch
# Distributed training demo with 2 replicas
# Based on: https://keras.io/keras_rs/examples/dcn/

import os

os.environ["KERAS_BACKEND"] = "tensorflow"


import keras

import kubetorch as kt
import numpy as np
import tensorflow as tf

CONFIG = {
    "int_features": ["movie_id", "user_id", "user_gender", "bucketized_user_age"],
    "str_features": ["user_zip_code", "user_occupation_text"],
    "embedding_dim": 8,
    "deep_units": [192, 192],
    "projection_dim": 8,
    "learning_rate": 1e-2,
    "epochs": 8,
    "batch_size": 8192,
}


class DCNTrainer:
    def __init__(self):
        self.vocabularies = None
        self.lookup_layers = None
        self.train_ds = None
        self.test_ds = None
        self.history = None

    def prepare_data(self, num_replicas=1):
        """Load and prepare MovieLens dataset."""
        import tensorflow_datasets as tfds

        print("Loading MovieLens dataset...")
        ratings_ds = tfds.load("movielens/100k-ratings", split="train")
        ratings_ds = ratings_ds.map(
            lambda x: (
                {
                    "movie_id": int(x["movie_id"]),
                    "user_id": int(x["user_id"]),
                    "user_gender": int(x["user_gender"]),
                    "user_zip_code": x["user_zip_code"],
                    "user_occupation_text": x["user_occupation_text"],
                    "bucketized_user_age": int(x["bucketized_user_age"]),
                },
                x["user_rating"],
            )
        )

        # Build vocabularies
        print("Building vocabularies...")
        self.vocabularies = {}
        for feature in CONFIG["int_features"] + CONFIG["str_features"]:
            vocab = ratings_ds.batch(10_000).map(lambda x, y: x[feature])
            self.vocabularies[feature] = np.unique(np.concatenate(list(vocab)))

        # Create lookup layers
        self.lookup_layers = {}
        for feature in CONFIG["int_features"]:
            self.lookup_layers[feature] = keras.layers.IntegerLookup(
                vocabulary=self.vocabularies[feature]
            )
        for feature in CONFIG["str_features"]:
            self.lookup_layers[feature] = keras.layers.StringLookup(
                vocabulary=self.vocabularies[feature]
            )

        # Apply lookups
        ratings_ds = ratings_ds.map(
            lambda x, y: (
                {f: self.lookup_layers[f](x[f]) for f in self.vocabularies},
                y,
            )
        )

        # Train/test split
        ratings_ds = ratings_ds.shuffle(100_000, seed=42)
        batch_size = CONFIG["batch_size"] * num_replicas

        self.train_ds = (
            ratings_ds.take(80_000).batch(batch_size).cache().prefetch(tf.data.AUTOTUNE)
        )
        self.test_ds = (
            ratings_ds.skip(80_000).batch(batch_size).cache().prefetch(tf.data.AUTOTUNE)
        )

        return self.train_ds, self.test_ds

    def train(self, use_cross_layer=True):
        """Train DCN model with optional distributed strategy."""

        # Setup distributed strategy
        if "TF_CONFIG" in os.environ:
            strategy = tf.distribute.MultiWorkerMirroredStrategy()
            print(
                f"Running distributed training with {strategy.num_replicas_in_sync} replicas"
            )
        else:
            strategy = tf.distribute.get_strategy()
            print("Running single-worker training")

        # Prepare data
        train_ds, test_ds = self.prepare_data(
            num_replicas=strategy.num_replicas_in_sync
        )

        # Build and train model
        with strategy.scope():
            model = self._build_model(use_cross_layer)
            model.compile(
                optimizer=keras.optimizers.AdamW(learning_rate=CONFIG["learning_rate"]),
                loss=keras.losses.MeanSquaredError(),
                metrics=[keras.metrics.RootMeanSquaredError()],
            )

        print(f"Model params: {model.count_params():,}")
        print(f"Training for {CONFIG['epochs']} epochs...")

        self.history = model.fit(
            train_ds,
            epochs=CONFIG["epochs"],
            validation_data=test_ds,
            verbose=1,
        )

        # Evaluate
        results = model.evaluate(test_ds, return_dict=True, verbose=0)
        rmse = results["root_mean_squared_error"]
        print(f"Final RMSE: {rmse:.4f}")

        return {
            "rmse": float(rmse),
            "params": model.count_params(),
            "epochs": CONFIG["epochs"],
            "use_cross_layer": use_cross_layer,
        }

    def _build_model(self, use_cross_layer):
        """Build DCN model."""
        import keras_rs

        inputs = {
            f: keras.Input(shape=(), dtype="int64", name=f) for f in self.vocabularies
        }

        # Embeddings
        embeddings = []
        for feature in self.vocabularies:
            emb = keras.layers.Embedding(
                input_dim=len(self.vocabularies[feature]) + 1,
                output_dim=CONFIG["embedding_dim"],
            )(inputs[feature])
            embeddings.append(emb)

        x = keras.layers.Concatenate()(embeddings)

        # Cross layer
        if use_cross_layer:
            x = keras_rs.layers.FeatureCross(projection_dim=CONFIG["projection_dim"])(x)

        # Deep layers
        for units in CONFIG["deep_units"]:
            x = keras.layers.Dense(units, activation="relu")(x)

        output = keras.layers.Dense(1)(x)

        return keras.Model(inputs=inputs, outputs=output)


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--local", action="store_true", help="Run locally without kubetorch"
    )
    parser.add_argument("--no-cross", action="store_true", help="Disable cross layer")
    args = parser.parse_args()

    if args.local:
        trainer = DCNTrainer()
        result = trainer.train(use_cross_layer=not args.no_cross)
        print(f"Result: {result}")
    else:
        # Distributed training with kubetorch
        compute = kt.Compute(
            cpus=4,
            memory="8Gi",
            image=kt.Image().pip_install(
                [
                    "tensorflow",
                    "tensorflow-datasets",
                    "keras>=3.0",
                    "keras-rs",
                ]
            ),
        ).distribute(
            distribution="tensorflow",
            workers=2,
        )

        trainer = kt.cls(DCNTrainer).to(compute)
        result = trainer.train(use_cross_layer=not args.no_cross)
        print(f"Result: {result}")


if __name__ == "__main__":
    main()
