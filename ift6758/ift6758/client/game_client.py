from ..tidy_data import game_client
import pandas as pd
import numpy as np
import requests

class GameClient:
    def get_game_data(game_id: int):
        url_game = f'https://statsapi.web.nhl.com/api/v1/game/{int(game_id)}/feed/live/'
        response = requests.get(url_game)

        if response.status_code != 200:
            return None
        return response.json()

    def ping_game(game_id: int) -> pd.DataFrame:
        # feature engineering, clean, transform from json to df
        df = get_game_data(game_id: int)

        if data is None:
            return pd.DataFrame()
        
        return df