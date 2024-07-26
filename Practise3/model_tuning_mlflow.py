# In this tutorial we are going to learn about the model
#tuning with Hyperpot and mlflow
from mlflow_utils import create_dataset
from hyperopt import hp
from hyperopt import fmin
from hyperopt import tpe
from hyperopt import Trials

def objective_function(params):
    y=(params["x"]-1)**2+2

    return y

search_space={
    "x":hp.uniform("x",-10,10)
}
trails=Trials()

best=fmin(
    fn=objective_function,
    space=search_space,
    algo=tpe.suggest,
    max_evals=100,
    trials=trails
)
print(best)