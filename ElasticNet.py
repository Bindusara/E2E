import os
import warnings
import sys
import dagshub

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from urllib.parse import urlparse
import mlflow
import mlflow.sklearn

from mlflow.models import infer_signature
from itertools import product
import logging

# os.environ['MLFLOW_TRACKING_USERNAME'] = 'bindusara007'
# os.environ['MLFLOW_TRACKING_PASSWORD'] = 'e8b66976bf29ade7fc08e9ebb6c3b0bda784edd4'

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

dagshub.init(repo_owner='bindusara007', repo_name='E2E', mlflow=True)
mlflow.set_tracking_uri('https://dagshub.com/bindusara007/E2E.mlflow')

def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    # Read the wine-quality csv file from the URL
    csv_url = (
        "https://raw.githubusercontent.com/mlflow/mlflow/master/tests/datasets/winequality-red.csv"
    )
    try:
        data = pd.read_csv(csv_url, sep=";")
    except Exception as e:
        logger.exception(
            "Unable to download training & test CSV, check your internet connection. Error: %s", e
        )

    # Split the data into training and test sets. (0.75, 0.25) split.
    train, test = train_test_split(data)

    # The predicted column is "quality" which is a scalar from [3, 9]
    train_x = train.drop(["quality"], axis=1)
    test_x = test.drop(["quality"], axis=1)
    train_y = train[["quality"]]
    test_y = test[["quality"]]

    alpha = [0.2, 0.6, 0.8]
    l1_ratio = [0.5, 0.7]
    
    mlflow.autolog()

    with mlflow.start_run(run_name="Hyperparameter_Tuning") as parent_run:
        mlflow.log_param("tuning_method", "grid_search")
        
        param_combinations = list(product(alpha, l1_ratio))
        
        for alpha, l1_ratio in param_combinations:
            with mlflow.start_run(nested=True) as child_run:
                lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
                lr.fit(train_x, train_y)

                predicted_qualities = lr.predict(test_x)

                (rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)

                print("Elasticnet model (alpha={:f}, l1_ratio={:f}):".format(alpha, l1_ratio))
                print("  RMSE: %s" % rmse)
                print("  MAE: %s" % mae)
                print("  R2: %s" % r2)

                mlflow.log_param("alpha", alpha)
                mlflow.log_param("l1_ratio", l1_ratio)
                mlflow.log_metric("rmse", rmse)
                mlflow.log_metric("r2", r2)
                mlflow.log_metric("mae", mae)

        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        # Model registry does not work with file store
        if tracking_url_type_store != "file":
            # Register the model
            # There are other ways to use the Model Registry, which depends on the use case,
            # please refer to the doc for more information:
            # https://mlflow.org/docs/latest/model-registry.html#api-workflow
            mlflow.sklearn.log_model(lr, "model", registered_model_name="ElasticnetWineModel")
        else:
            mlflow.sklearn.log_model(lr, "model")