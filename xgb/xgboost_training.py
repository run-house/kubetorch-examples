import kubetorch as kt
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

# ## First we encapsulate XGB training in a class
# We will send this training class to a remote instance with a GPU with Runhouse
class Trainer:
    def __init__(self):
        self.model = None
        self.dtrain = None
        self.dtest = None
        self.dval = None

    def load_data(self):
        import tensorflow as tf  # Imports in the function are only required on remote Image, not local env

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
        X = X.reshape(X.shape[0], -1).astype(np.float32)
        return X / 255.0

    def train_model(self, params, num_rounds):
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
        preds = self.model.predict(self.dtest)
        accuracy = accuracy_score(self.dtest.get_label(), preds)
        print(f"Test accuracy: {accuracy:.4f}")
        print(
            "\nClassification Report:\n",
            classification_report(self.dtest.get_label(), preds),
        )
        return preds.tolist(), accuracy

    def predict(self, X):
        if not isinstance(X, xgb.DMatrix):
            X = xgb.DMatrix(X)
        return self.model.predict(X)

    def save_model(self, path):
        self.model.save_model(path)

    def load_model(self, path):
        self.model = xgb.Booster()
        self.model.load_model(path)


# ## Set up Runhouse primitives
#
# Now, we define the main function that will run locally when we run this script and set up
# our Runhouse module on a remote cluster. First, we create a cluster with the desired instance type and provider.
if __name__ == "__main__":
    img = kt.Image(image_id="nvcr.io/nvidia/pytorch:23.10-py3").pip_install(
        ["numpy > 2.0.0", "xgboost", "pandas", "scikit-learn", "tensorflow"]
    )

    cluster = kt.Compute(
        gpus="1",
        cpus=3,
        memory="12Gi",
        image=img,
        inactivity_ttl="20m",
        launch_timeout=600,
    )

    train_params = {
        "objective": "multi:softmax",
        "num_class": 10,
        "eval_metric": ["mlogloss", "merror"],
        "max_depth": 6,
        "eta": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "tree_method": "gpu_hist",  # Using a GPU here reduces training time by ~99%
        "predictor": "gpu_predictor",
        "seed": 42,
        "n_jobs": -1,
    }

    # Now we send the training class to the remote cluster and invoke the training
    remote_trainer = kt.cls(Trainer).to(cluster)
    remote_trainer.load_data()
    remote_trainer.train_model(train_params, num_rounds=100)
    remote_trainer.test_model()
    remote_trainer.save_model("fashion_mnist.model")
