import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from serving_client import *
from game_client import *


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
        # if model changed, tracker will be cleared, so no interference on seen data between models
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
            df_for_pred, game_state, period, timeLeft, home_team, away_team, home_score, away_score = gc.ping_game(int(game_id))
            st.subheader(f"Game {game_id}: {home_team} (Home) vs {away_team} (Away)")

            if game_state == "FUT":
                st.write(f"This game is scheduled for {period}.")
            elif game_state == "PRE":
                st.write(f"This game will start shortly.")
            else:

                if game_state == 'LIVE' or game_state == "CRIT":
                    st.write(f"Current Score: {home_team} {home_score} - {away_score} {away_team}")
                    st.write(f"Period: {period} - {timeLeft} left")
                else:
                    st.write(f'The game ended with the final score: {home_team} {home_score} - {away_score} {away_team}')

                if len(df_for_pred) != 0:
                    y = sc.predict(df_for_pred)
                    y = list(y.values())
                    y = [round(value, 1) for value in y]
                    df_y = pd.DataFrame(y)

                    df_for_pred['xG'] = df_y.values

                    home_xG = df_for_pred[home_team == df_for_pred['Team']]['xG'].sum()
                    away_xG = df_for_pred[away_team == df_for_pred['Team']]['xG'].sum()

                    # different columns kept depending on the model
                    if model == "logisticregressiondistancetogoal":
                        df_for_pred = df_for_pred[['DistanceToGoal', 'Team']]
                    elif model == "logisticregressionshootingangle":
                        df_for_pred = df_for_pred[['ShootingAngle', 'Team']]

                f = open('tracker.json')
                data = json.load(f)
                if 'home_xG' in data[str(game_id)]: # str(game_id) in data: #
                    temp_home_xG = data[str(game_id)]['home_xG']
                    temp_away_xG = data[str(game_id)]['away_xG']
                    data[str(game_id)]['home_xG'] = temp_home_xG + home_xG
                    data[str(game_id)]['away_xG'] = temp_away_xG + away_xG
                else:
                    data[str(game_id)]['home_xG'] = home_xG
                    data[str(game_id)]['away_xG'] = away_xG

                home_xG = data[str(game_id)]['home_xG']
                away_xG = data[str(game_id)]['away_xG']
                cols = st.columns(2)
                cols[0].metric(label=home_team + ' xG (actual)', value=str(round(home_xG, 1)) + " (" + str(home_score) + ')',
                               delta=round(float(home_score) - float(home_xG), 1))
                cols[1].metric(label=away_team + ' xG (actual)', value=str(round(away_xG, 1)) + " (" + str(away_score) + ')',
                               delta=round(float(away_score) - float(away_xG), 1))

                with open('tracker.json', 'w') as outfile:
                    json.dump(data, outfile, default=convert_to_python_types)

                # df = df_for_pred.reset_index()
                if len(df_for_pred) != 0:
                    # Add data used for predictions
                    st.subheader("Data used for predictions:")
                    st.dataframe(df_for_pred)
                else:
                    if game_state == 'LIVE' or game_state == "CRIT":
                        st.write("We have seen all the events for this game so far.")
                    else:
                        st.write("We have seen all the events for this game.")

# Bonus
with st.container():
    if ping_button:
        df = gc.ping_game_bonus(int(game_id))

        if game_state == "LIVE" or game_state == "CRIT":
            st.subheader('Map of Shots and Goals')

            fig, ax = plt.subplots(figsize=(10, 5), dpi=100)
            img = mpimg.imread("nhl_rink.png")
            ax.imshow(img, extent=[-100.0, 100.0, -42.5, 42.5])

            ax.set_xlabel('feet')
            ax.set_ylabel('feet')
            ax.set_title(
                f"Period: {period} - {timeLeft} left, Current Score: {home_team} {home_score} - {away_score} {away_team}")
        elif game_state == "FINAL" or game_state == "OFF":
            st.subheader('Map of Shots and Goals')

            fig, ax = plt.subplots(figsize=(10, 5), dpi=100)
            img = mpimg.imread("nhl_rink.png")
            ax.imshow(img, extent=[-100.0, 100.0, -42.5, 42.5])

            ax.set_xlabel('feet')
            ax.set_ylabel('feet')
            ax.set_title(f"The game ended with the final score: {home_team} {home_score} - {away_score} {away_team}")

        if game_state == "LIVE" or game_state == "FINAL" or game_state == "CRIT" or game_state == "OFF":

            try:
                # Separate shots and goals for each team
                home_shots = df[(df['Event'] == 'shot-on-goal') & (df['Team'] == home_team)]
                away_shots = df[(df['Event'] == 'shot-on-goal') & (df['Team'] == away_team)]

                home_goals = df[(df['Event'] == 'goal') & (df['Team'] == home_team)]
                away_goals = df[(df['Event'] == 'goal') & (df['Team'] == away_team)]

                # Change the sign of positive XCoord values for the home team
                home_shots.loc[home_shots['XCoord'] > 0, 'XCoord'] *= -1
                home_goals.loc[home_goals['XCoord'] > 0, 'XCoord'] *= -1

                # Change the sign of negative XCoord values for the away team
                away_shots.loc[away_shots['XCoord'] < 0, 'XCoord'] *= -1
                away_goals.loc[away_goals['XCoord'] < 0, 'XCoord'] *= -1

                # Flip the sign of YCoord for the coordinates that have been changed
                home_shots.loc[home_shots['XCoord'] < 0, 'YCoord'] *= -1
                home_goals.loc[home_goals['XCoord'] < 0, 'YCoord'] *= -1
                away_shots.loc[away_shots['XCoord'] > 0, 'YCoord'] *= -1
                away_goals.loc[away_goals['XCoord'] > 0, 'YCoord'] *= -1

                # Plot shots and goals with different colors for each team
                ax.scatter(home_shots['XCoord'], home_shots['YCoord'], marker='o', color='blue', label=f'{home_team} Shots')
                ax.scatter(home_goals['XCoord'], home_goals['YCoord'], marker='x', color='red', label=f'{home_team} Goals')

                ax.scatter(away_shots['XCoord'], away_shots['YCoord'], marker='o', color='orange',
                           label=f'{away_team} Shots')
                ax.scatter(away_goals['XCoord'], away_goals['YCoord'], marker='x', color='green', label=f'{away_team} Goals')

                ax.legend()
            except Exception as e:
                logging.error(f"An error occurred: {str(e)}")

            # Display the figure in Streamlit
            st.pyplot(fig)

            st.write(
                "In this version of the Shots and Goals Map, we've introduced team-specific coordinates to provide a clearer "
                "visualization of shots and goals for each team. Positive X coordinates for the home team have been transformed "
                "to negative, and negative X coordinates for the away team have been transformed to positive. Additionally, the "
                "Y coordinates corresponding to the changed X coordinates have been flipped, ensuring a consistent representation "
                "of the rink for both teams. This enhancement improves the interpretability of the map, allowing users to easily "
                "distinguish between shots and goals from the home and away teams. The technical challenge involved conditional "
                "data manipulation and thoughtful coordination of visual elements to create a more informative and engaging map.")

            st.write("We also handle the cases when a game will start shortly (pre-game) and when a game wasn't played yet. "
                     "In the second case, we indicate when the game is scheduled to be played.")
