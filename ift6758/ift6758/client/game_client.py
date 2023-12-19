from tidy_data import generate_game_client_df
import pandas as pd
import numpy as np
import requests
import json
import os

class GameClient:
    def ping_game(self, game_id: int) -> pd.DataFrame:
        url_game = f'https://api-web.nhle.com/v1/gamecenter/{int(game_id)}/play-by-play/'
        response = requests.get(url_game)

        id = str(game_id)

        # Move up three levels in the directory structure
        parent_directory = os.path.abspath(os.path.join(os.getcwd(), os.pardir, os.pardir, os.pardir))

        file_name = os.path.join(parent_directory, f'nhl_play_by_play_{id[:4]}_{game_id}.json')

        if response.status_code == 200:
            data = response.json()
            with open(file_name, 'w') as file:
                json.dump(data, file)

        # feature engineering, clean, transform from json to df
        df = generate_game_client_df(f'{file_name}')

        if df is None:
            return pd.DataFrame()
        
        return df

# #Testing
game_id_to_test = 2022020451
client = GameClient()
result_df = client.ping_game(game_id_to_test)

print(result_df)