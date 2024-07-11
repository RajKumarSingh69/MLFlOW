import mlflow
from mlflow.models import infer_signature
from mlflow_utils import get_mlflow_experiment

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

import pandas as pd

if __name__ == "__main__":
    #specifying the run_id for that model for accesing the model and do prediction
    run_id = "36d9ac276b914a0ea9e4f0f0fbfed345"

    X, y = make_classification(n_samples=1000, n_features=10, n_informative=5, n_redundant=5, random_state=42)
    X = pd.DataFrame(X, columns=["feature_{}".format(i) for i in range(10)])
    y = pd.DataFrame(y, columns=["target"])

    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=43)

    # Load model
    #here is the model path with run id
    #model_uri=f"file:///F:/DATA_SCIENCE/MLOPS/MLFLOW/mlruns/386416466345023223/36d9ac276b914a0ea9e4f0f0fbfed345/artifacts/random_forest_classifier"abs
    #below is the second way
    model_uri = f"runs:/{run_id}/random_forest_classifier"
    rfc = mlflow.sklearn.load_model(model_uri=model_uri)

    y_pred = rfc.predict(X_test)
    y_pred = pd.DataFrame(y_pred, columns=["prediction"])

    print(y_pred.head())
