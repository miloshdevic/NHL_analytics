from ...tidy_data import generate_game_client_df
import pandas as pd
import numpy as np
import requests

class GameClient:
    def get_game_data(game_id: int):
        url_game = f'https://api-web.nhle.com/v1/gamecenter/{int(game_id)}/play-by-play/'
        response = requests.get(url_game)

        if response.status_code == 200:
            data = response.json()
            with open(url_game) as file:
                return json.dumps(data, file)

    def ping_game(game_id: int) -> pd.DataFrame:
        # feature engineering, clean, transform from json to df
        df = get_game_data(game_id)

        if df is None:
            return pd.DataFrame()
        
        return df

# #Testing
game_id_to_test = 2022020451
client = GameClient()
result_df = client.ping_game(game_id_to_test)

print(result_df)