import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
from serving_client import *
from game_client import *
#from ift6758.client.game_client import *


st.title("NHL Live Games Visualization App")
sc = ServingClient(ip='127.0.0.1',port = 8000)
gc = GameClient()

# Convert int64 values to Python int for JSON serialization
def convert_to_python_types(obj):
    if isinstance(obj, np.int64):
        return int(obj)
    return obj

with st.sidebar:
    # Add input for the sidebar
    workspace = st.selectbox(
        "Work space",
        ("nhl-analytics-milestone-2", "")
    )
    model = st.selectbox(
        "Model",
        ("logisticregressiondistancetogoal", "logisticregressionshootingangle",
         "logisticregressiondistancetogoal_shootingangle")
    )
    version = st.selectbox(
        "Version",
        ("1.1.0", "")
    )
    download_button = st.button("Download Model")
    if download_button:
        # if model changed, tracker will be cleared, so no interference of seen data between models
        with open('tracker.json', 'w') as outfile:
            data = {}
            json.dump(data, outfile)
        sc.download_registry_model(workspace=workspace, model=model, version=version)
        st.write('Model Downloaded')

with st.container():
    # Add Game ID input
    game_id = st.text_input("Game ID", "")
    ping_button = st.button("Ping Game")

with st.container():
    # Add Game info and predictions
    if ping_button:
        with st.container():
            home_xG = 0
            away_xG = 0
            # r = requests.post(
            #     f"http://127.0.0.1:8000/predict",
            #     json=json.loads(df_for_pred.to_json())
            # )
            # predictions = r.json()

            # Display game information and expected goals
            df_for_pred, live, period, timeLeft, home_team, away_team, home_score, away_score = gc.ping_game(int(game_id))
            st.subheader(f"Game {game_id}: {home_team} (Home) vs {away_team} (Away)")
            if live:
                st.write(f"Period: {period} - {timeLeft} left")
            else:
                st.write(f'The game ended with the score: {home_team} {home_score} - {away_score} {away_team}')
            if len(df_for_pred) != 0:
                # df_og = df_for_pred.copy()
                y = sc.predict(df_for_pred)
                y = list(y.values())
                y = [round(value) for value in y]  # [1 if value > 0.58 else 0 for value in y] #
                df_y = pd.DataFrame(y)
                df_for_pred['xG'] = df_y.values

                home_xG = df_for_pred[home_team == df_for_pred['Team']]['xG'].sum()
                away_xG = df_for_pred[away_team == df_for_pred['Team']]['xG'].sum()

            f = open('tracker.json')
            data = json.load(f)
            if 'home_xG' in data[str(game_id)]:
                st.write(f'if')
                temp_home_xG = data[str(game_id)]['home_xG']
                temp_away_xG = data[str(game_id)]['away_xG']
                data[str(game_id)]['home_xG'] = temp_home_xG + home_xG
                data[str(game_id)]['away_xG'] = temp_away_xG + away_xG
            else:
                st.write(f'else')
                data[str(game_id)]['home_xG'] = home_xG
                data[str(game_id)]['away_xG'] = away_xG

            home_xG = data[str(game_id)]['home_xG']
            away_xG = data[str(game_id)]['away_xG']
            cols = st.columns(2)
            cols[0].metric(label=home_team + ' xG (actual)', value=str(home_xG) + " (" + str(home_score) + ')',
                           delta=int(home_score) - int(home_xG))
            cols[1].metric(label=away_team + ' xG (actual)', value=str(away_xG) + " (" + str(away_score) + ')',
                           delta=int(away_score) - int(away_xG))

            with open('tracker.json', 'w') as outfile:
                json.dump(data, outfile, default=convert_to_python_types)

            # df = df_for_pred.reset_index()
            if len(df_for_pred) != 0:
                st.subheader("Data used for predictions:")
                st.dataframe(df_for_pred)
            else:
                st.write("We have seen all the events for", game_id)

            # st.write(f"Current Score: {predictions['current_score']}")
            #
            # # NOT DONE YET, NEED TO BE MODIFIED
            # st.write(f"Sum of Expected Goals (xG) - Home: {predictions['sum_xG_home']}")
            # st.write(f"Sum of Expected Goals (xG) - Away: {predictions['sum_xG_away']}")
            #
            # # NOT DONE YET, NEED TO BE MODIFIED
            # st.metric(label=f"Goals Scored - {home_team}", value=home_score,
            #           delta=predictions['score']-predictions['sum_xG_home'])
            #
            # st.metric(label=f"Goals Scored - {away_team}", value=away_score,
            #           delta=predictions['score']-predictions['sum_xG_home'])

            # Add data used for predictions
