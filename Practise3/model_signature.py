# The modle signature is an object that allows you to 
#specify the data type and the data shape of the data that the model can work with

import mlflow
from mlflow_utils import create_mlflow_experiment
from mlflow.models.signature import ModelSignature
from mlflow.models.signature import infer_signature
from mlflow.types.schema import Schema
from mlflow.types.schema import ColSpec
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
import pandas as pd
from typing import Tuple


def get_train_data() -> Tuple[pd.DataFrame]:
    """
    Generate train and test data.

    :return: x_train,y_train
    """
    x, y = make_classification()
    features = [f"feature_{i+1}" for i in range(x.shape[1])]
    df = pd.DataFrame(x, columns=features)
    df["label"] = y

    return df[features], df["label"]


if __name__ == "__main__":
    x_train, y_train = get_train_data()
    cols_spec = []
    data_map = {
        'int64': 'integer',
        'float64': 'double',
        'bool': 'boolean',
        'object': 'string',
        'datetime64[ns]': 'datetime'
    }

    for name, dtype in x_train.dtypes.to_dict().items():
        cols_spec.append(ColSpec(type=data_map[str(dtype)], name=name))

    input_schema = Schema(cols_spec)
    output_schema = Schema([ColSpec(type="integer", name="label")])

    model_signature = ModelSignature(inputs=input_schema, outputs=output_schema)
    print("MODEL SIGNATURE")
    print(model_signature.to_dict())

    model_signature = infer_signature(x_train, y_train)
    print("MODEL SIGNATURE (inferred)")
    print(model_signature.to_dict())

    experiment_id = create_mlflow_experiment(
        experiment_name="Model Signature",
        artifact_location="model_signature_artifacts",
        tags={"purpose": "learning"},
    )

    with mlflow.start_run(run_name="model_signature_run") as run:
        mlflow.sklearn.log_model(
            sk_model=RandomForestClassifier(),
            artifact_path="model_signature",
            signature=model_signature,
        )
