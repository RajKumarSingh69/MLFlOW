import mlflow

if __name__=="__main__":
    #start a new mlflow run
    mlflow.start_run()

    # our machie learning code 
    mlflow.log_param("learning_rate",0.01)

    #end the mlflow run
    mlflow.end_run()