import json
import requests
import pandas as pd
import logging
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class ServingClient:
    def __init__(self, ip: str = "0.0.0.0", port: int = 8000, features=None):
        self.base_url = f"http://{ip}:{port}"
        logger.info(f"Initializing client; base URL: {self.base_url}")

        if features is None:
            features = ["distance"]
        self.features = features

        # any other potential initialization

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Formats the inputs into an appropriate payload for a POST request, and queries the
        prediction service. Retrieves the response from the server, and processes it back into a
        dataframe that corresponds index-wise to the input dataframe.
        
        Args:
            X (Dataframe): Input dataframe to submit to the prediction service.
        """
        # logger.info(f"Intializing POST request predictions, query the predictions")
        # json_data = json.loads(X.to_json())
        # response = requests.post(f"{self.base_url}/predict", json=json_data)
        # logger.info(f"Query the predictions with success")
        # body = response.json()
        # df = pd.DataFrame.from_records(body)
        # return df
        # X = X[['DistanceToGoal', 'ShootingAngle']]
        # X[['DistanceToGoal', 'ShootingAngle']] = StandardScaler().fit_transform(X[['DistanceToGoal', 'ShootingAngle']])
        json_data = json.loads(X.to_json())
        request = requests.post(
            f"{self.base_url}/predict",
            json=json_data
        )
        response = request.json()
        return response

    def logs(self) -> dict:
        """Get server logs"""

        logger.info(f"Initializing POST request to get server logs")
        response = requests.get(f"{self.base_url}/logs")
        logger.info(f"Getting server logs with success")
        logs = response.json()
        return logs

    def download_registry_model(self, workspace: str, model: str, version: str) -> dict:
        """
        Triggers a "model swap" in the service; the workspace, model, and model version are
        specified and the service looks for this model in the model registry and tries to
        download it. 

        See more here:

            https://www.comet.ml/docs/python-sdk/API/#apidownload_registry_model
        
        Args:
            workspace (str): The Comet ML workspace
            model (str): The model in the Comet ML registry to download
            version (str): The model version to download
        """

        # logger.info(f"Initializing request to download the comet model")
        # request_dict = {"workspace": workspace, "model": model, "version": version}
        # data = json.dumps(request_dict)
        # response = requests.post(f"{self.base_url}/download_registry_model",json=data)
        # return response.json()
        response = requests.post(self.base_url + '/download_registry_model',
                                 json={'workspace': workspace, 'model': model, 'version': version})
        logger.info("SUCCESS: Model downloaded!")
        return response.json()
        
        
        
