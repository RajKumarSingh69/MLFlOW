import mlflow
from mlflow_utils import get_mlflow_experiment

if __name__=="__main__":
    #retrieve the mlflow experiment
    #experiment=get_mlflow_experiment(experiment_name="testing_mlflow")
    experiment=get_mlflow_experiment(experiment_id='906387291137381899')

    print("Name: {}".format(experiment.name))
    print("Experiment_id: {}".format(experiment.experiment_id))
    print("Artifact Location: {}".format(experiment.artifact_location))
    print("Tage: {}".format(experiment.tags))
    print("Lifecycle_stage: {}".format(experiment.lifecycle_stage))
    print("Creation timestamp: {}".format(experiment.creation_time))