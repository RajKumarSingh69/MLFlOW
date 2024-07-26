import mlflow
from mlflow.models.signature import infer_signature, ModelSignature
from mlflow.types import ColSpec, Schema
import numpy as np
import pandas as pd

class CustomModel(mlflow.pyfunc.PythonModel):
    def predict_model1(self, model_input):
        # do some processing for model 1
        return 0 * model_input

    def predict_model2(self, model_input):
        # do some processing for model 2
        return model_input

    def predict_model3(self, model_input):
        # do some processing for model 3
        return 2 * model_input

    def predict(self, context, model_input):
        params = context.artifacts["params"]
        if params["model_name"] == "model_1":
            return self.predict_model1(model_input=model_input)
        elif params["model_name"] == "model_2":
            return self.predict_model2(model_input=model_input)
        elif params["model_name"] == "model_3":
            return self.predict_model3(model_input=model_input)
        else:
            raise Exception("Model Not Found!")

def create_mlflow_experiment(experiment_name, artifact_location, tags):
    # Dummy implementation
    experiment_id = "0"
    return experiment_id

if __name__ == "__main__":
    experiment_id = create_mlflow_experiment(
        experiment_name="Serving Multiple Models",
        artifact_location="serving_multiple_models",
        tags={"purpose": "learning"},
    )
    input_schema = Schema([ColSpec(type="integer", name="input")])
    output_schema = Schema([ColSpec(type="integer", name="output")])
    model_signature = ModelSignature(inputs=input_schema, outputs=output_schema)

    with mlflow.start_run(run_name="multiple_models", experiment_id=experiment_id) as run:
        params = {"model_name": "model_1"}
        mlflow.log_dict(params, "params.json")
        mlflow.pyfunc.log_model(
            artifact_path="model", 
            python_model=CustomModel(), 
            signature=model_signature, 
            artifacts={"params": "params.json"}
        )

        model_uri = f"runs:/{run.info.run_id}/model"
        loaded_model = mlflow.pyfunc.load_model(model_uri=model_uri, dst_path="./")

        for n in range(3):
            params = {"model_name": f"model_{n+1}"}
            with open("params.json", "w") as f:
                json.dump(params, f)
            context = mlflow.pyfunc.PythonModelContext(artifacts={"params": "params.json"})
            print(f"PREDICTION FROM MODEL {n+1}")
            print(loaded_model.predict(model_input=np.array([[10]]), context=context))
            print("\n")

        print(f"RUN_ID: {run.info.run_id}")
