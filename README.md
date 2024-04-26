# E2E


# For DagsHub

MLFLOW_TRACKING_URI=https://dagshub.com/bindusara007/E2E.mlflow \
MLFLOW_TRACKING_USERNAME=bindusara007 \
MLFLOW_TRACKING_PASSWORD=e8b66976bf29ade7fc08e9ebb6c3b0bda784edd4 \
python script.py


```bash

export MLFLOW_TRACKING_URI=https://dagshub.com/bindusara007/E2E.mlflow

export MLFLOW_TRACKING_USERNAME=bindusara007

export MLFLOW_TRACKING_PASSWORD=e8b66976bf29ade7fc08e9ebb6c3b0bda784edd4

```

# MLflow on AWS



Run the following command on EC2 machine

```bash



#Bucket_Name
mlflow server -h 0.0.0.0 --default-artifact-root s3://mlfb-e2e
export MLFLOW_TRACKING_URI=http://ec2-3-88-101-45.compute-1.amazonaws.com:5000/

