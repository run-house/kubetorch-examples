# # XGB Trainer class and Kubetorch launch
# In this file, you will see a Trainer class that encapsulates XGB training,
# with a dummy load_data() method that just grabs the fashion MNIST classification
# exercise, a method to do training, a method to test the model, and a method to save.
# This is easily adaptable to your training.
#
# If we run locally, with `python trainer.py`, then, can see we will locally run the helper
# function launch_training, which will launch Kubetorch compute, send our trainer there, and
# then call methods on that remote instance of the trainer.
#
# The beauty is that now all we need to make this run as orchestrated by Temporal is
# to import and call that same function from within Temporal. This is 100% reproducing
# across Temporal and local (making it easy to debug production as well), and
# all the underlying training code is regular code.
class Trainer:
    """XGBoost trainer that runs on remote GPU compute."""

    def __init__(self):
        self.model = None
        self.dtrain = None
        self.dtest = None
        self.dval = None

    def load_data(self):
        import tensorflow as tf
        import xgboost as xgb
        from sklearn.model_selection import train_test_split

        fashion_mnist = tf.keras.datasets.fashion_mnist
        (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

        # Preprocess features
        X_train = self._preprocess_features(X_train)
        X_test = self._preprocess_features(X_test)

        # Split test into validation and test
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42
        )

        self.dtrain = xgb.DMatrix(X_train, label=y_train)
        self.dtest = xgb.DMatrix(X_test, label=y_test)
        self.dval = xgb.DMatrix(X_val, label=y_val)

    @staticmethod
    def _preprocess_features(X):
        import numpy as np

        X = X.reshape(X.shape[0], -1).astype(np.float32)
        return X / 255.0

    def train_model(self, params, num_rounds):
        import xgboost as xgb

        if not all([self.dtrain, self.dval]):
            raise ValueError("Data not loaded. Call load_data() first.")

        evals = [(self.dtrain, "train"), (self.dval, "val")]
        self.model = xgb.train(
            params,
            self.dtrain,
            num_boost_round=num_rounds,
            evals=evals,
            early_stopping_rounds=10,
            verbose_eval=True,
        )

    def test_model(self):
        from sklearn.metrics import accuracy_score, classification_report

        preds = self.model.predict(self.dtest)
        accuracy = accuracy_score(self.dtest.get_label(), preds)
        print(f"Test accuracy: {accuracy:.4f}")
        print(
            "\nClassification Report:\n",
            classification_report(self.dtest.get_label(), preds),
        )
        return accuracy

    def save_model(self, path):
        self.model.save_model(path)


# Teardown is set to false during local iteration and to true during production
# execution so it tears down immediately.
def launch_training(config, logger=None, teardown=False):
    """Launch GPU training with KubeTorch."""
    import kubetorch as kt

    if not logger:
        import logging

        logger = logging.getLogger(__name__)

    logger.info(f"Starting GPU training with config: {config}")

    # Define GPU compute configuration
    img = kt.Image(image_id="nvcr.io/nvidia/pytorch:23.10-py3").run_bash(
        'uv pip install --system -U "numpy>2.0.0" "xgboost>3.0.0" scipy scikit-learn pandas tensorflow pyarrow'
    )

    compute = kt.Compute(
        gpus="1",
        image=img,
        launch_timeout=600,
    )

    logger.info("Dispatching Trainer class to remote GPU compute")
    remote_trainer = kt.cls(Trainer).to(compute)

    logger.info("Loading data on remote compute")
    remote_trainer.load_data()

    logger.info(f"Training model for {config.num_rounds} rounds")
    remote_trainer.train_model(config.train_params, num_rounds=config.num_rounds)

    logger.info("Testing model")
    accuracy = remote_trainer.test_model()

    model_path = "fashion_mnist.model"
    logger.info(f"Saving model to {model_path}")
    remote_trainer.save_model(model_path)

    logger.info("Tearing down remote compute")
    remote_trainer.teardown()

    return accuracy, model_path


if __name__ == "__main__":
    from dataclasses import dataclass

    @dataclass
    class TrainingConfig:
        num_rounds: int
        train_params: dict

    # XGBoost parameters optimized for GPU
    train_params = {
        "objective": "multi:softmax",
        "num_class": 10,
        "eval_metric": ["mlogloss", "merror"],
        "max_depth": 6,
        "eta": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "tree_method": "hist",
        "predictor": "gpu_predictor",
        "device": "cuda",  # GPU-accelerated training
        "seed": 42,
        "n_jobs": -1,
    }

    config = TrainingConfig(num_rounds=100, train_params=train_params)
    accuracy, model_path = launch_training(config)
    print(f"Training complete! Accuracy: {accuracy:.4f}, Model: {model_path}")
