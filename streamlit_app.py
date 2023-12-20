import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
from ift6758.client import serving_client
from ift6758.client import game_client


st.title("NHL Live Games Visualization App")
sc = serving_client.ServingClient(ip='serving',port = 8000)
gc = game_client.Game_Client()

with st.sidebar:
    # Add input for the sidebar
    workspace = st.text_input("Workspace", "")
    model = st.text_input("Model", "")
    version = st.text_input("Version", "")
    download_button = st.button("Download Model")
    if st.button('Download Model'):
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
        # TODO: Use your GameClient to ping the game and obtain relevant information
        # Replace the following code with your actual implementation
        df_for_pred, live, period, timeLeft, home_team, away_team, home_score, away_score = game_client.ping_game(game_id)
        r = requests.post(
            f"http://127.0.0.1:8000/predict",
            json=json.loads(df_for_pred.to_json())
        )
        predictions = r.json()

        # Display game information and expected goals
        st.subheader(f"Game {game_id}: {home_team} (Home) vs {away_team} (Away)")

        st.write(f"Period: {period} - {timeLeft} left")

        st.write(f"Current Score: {predictions['current_score']}")

        # NOT DONE YET, NEED TO BE MODIFIED
        st.write(f"Sum of Expected Goals (xG) - Home: {predictions['sum_xG_home']}")
        st.write(f"Sum of Expected Goals (xG) - Away: {predictions['sum_xG_away']}")

        # NOT DONE YET, NEED TO BE MODIFIED
        st.metric(label=f"Goals Scored - {home_team}", value=home_score,
                  delta=predictions['score']-predictions['sum_xG_home'])

        st.metric(label=f"Goals Scored - {away_team}", value=away_score,
                  delta=predictions['score']-predictions['sum_xG_home'])

        # Add data used for predictions
        st.write("Data used for the predictions:")
        df = None
        st.dataframe(df_for_pred.style.highlight_max(axis=0, color='lightgreen'))



# with st.container():
#     # Add data used for predictions
#     st.write(f"Data used for the predictions:")
#     df = None
#     st.dataframe(df.style.highlight_max(axis=0, color='lightgreen'))