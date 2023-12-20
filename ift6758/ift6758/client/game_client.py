import pandas as pd
import numpy as np
import requests
import json
import os
import logging
from feature_extraction import *

logger = logging.getLogger(__name__)

# tracker file to not repeat events
# with open('tracker.json', 'w') as outfile:
#     data = {}
#     json.dump(data, outfile)

class GameClient:
    def __init__(self):
        self.processed_events = set()
        self.tracker = self.load_tracker()
        logger.info(f"Initializing ClientGame; base URL: ")

    def load_tracker(self):
        try:
            with open('tracker.json', 'r') as tracker_file:
                return json.load(tracker_file)
        except FileNotFoundError:
            return {}

    def extract_features(self, play_by_play):
        home_team_id = play_by_play['homeTeam']['id']
        away_team_id = play_by_play['awayTeam']['id']
        teams = {home_team_id: play_by_play['homeTeam']['name']['default'],
                 away_team_id: play_by_play['awayTeam']['name']['default']}

        # Initialize lists to store information
        game_time = []
        period = []
        time_left = []
        game_id_list = []
        team = []
        event_type = []
        x_coord = []
        y_coord = []
        shot_type = []
        isGoal = []
        isEmptyNet = []
        strength = []
        rinkSide = []
        season = []
        last_ping = []
        s = play_by_play['season']
        id = play_by_play['id']

        for event in play_by_play['plays']:
            if event['periodDescriptor']['periodType'] == "SO":
                continue
            game_id_list.append(id)
            season.append(s)
            if 'eventOwnerTeamId' in event:
                team.append(teams[event['eventOwnerTeamId']])
            elif 'details' in event and 'eventOwnerTeamId' in event['details']:
                team.append(teams[event['details']['eventOwnerTeamId']])
            else:
                team.append('')

            event_type.append(event['typeDescKey'])
            if event['typeDescKey'] == "shot-on-goal" and 'shotType' in event[
                'details']:  # sometimes API is missing values
                shot_type.append(event['details']['shotType'])
            else:
                shot_type.append('')

            isGoal.append(int(event['typeDescKey'] == 'goal'))
            game_time.append(event['timeInPeriod'])
            period.append(event['period'])
            time_left.append(event['timeRemaining'])

            x_coord.append(event['details']['xCoord'] if 'details' in event and 'xCoord' in event['details'] else None)
            y_coord.append(event['details']['yCoord'] if 'details' in event and 'yCoord' in event['details'] else None)

            if 'eventOwnerTeamId' in event:
                rinkSide.append(event['homeTeamDefendingSide'] if event['eventOwnerTeamId'] == home_team_id
                                else opposite(event['homeTeamDefendingSide']))
            elif 'details' in event and 'eventOwnerTeamId' in event['details']:
                rinkSide.append(event['homeTeamDefendingSide'] if event['details']['eventOwnerTeamId'] == home_team_id
                                else opposite(event['homeTeamDefendingSide']))
            else:
                rinkSide.append('')

            # Get information about skaters/goaltenders on the ice
            if 'situationCode' in event:
                situationCode = str(event['situationCode'])

                # Get the first digit
                away_goalie = int(situationCode[0])
                away_skaters = int(situationCode[1])
                home_skaters = int(situationCode[2])
                home_goalie = int(situationCode[3])

                if int(away_skaters) == int(home_skaters):
                    strength.append('Even')
                else:
                    strength.append('')

                if int(away_goalie) == 0 and int(event['typeDescKey'] == 'goal') and event['details'][
                    'eventOwnerTeamId'] == home_team_id:
                    isEmptyNet.append(1)
                elif int(home_goalie) == 0 and int(event['typeDescKey'] == 'goal') and event['details'][
                    'eventOwnerTeamId'] == away_team_id:
                    isEmptyNet.append(1)
                else:
                    isEmptyNet.append(0)
            else:
                strength.append('')
                isEmptyNet.append(0)

            last_ping.append('')
        last_ping[-1] = 'ping'

        # Create a DataFrame from the extracted information
        df = pd.DataFrame({
            'GameTime': game_time,
            'Period': period,
            'TimeLeft': time_left,
            'GameID': game_id_list,
            'Team': team,
            'Event': event_type,
            'XCoord': x_coord,
            'YCoord': y_coord,
            'ShotType': shot_type,
            'isGoal': isGoal,
            'isEmptyNet': isEmptyNet,
            'Strength': strength,
            'RinkSide': rinkSide,
            'Season': season,
            'LastPing': last_ping
        })

        return df

    def ping_game(self, game_id: int):
        live = True
        url_game = f'https://api-web.nhle.com/v1/gamecenter/{int(game_id)}/play-by-play/'
        response = requests.get(url_game)

        id = str(game_id)

        file_name = f'nhl_play_by_play_{id[:4]}_{game_id}.json'

        if response.status_code == 200:
            data = response.json()
            with open(file_name, 'w') as file:
                json.dump(data, file)

            with open(file_name, 'r') as file:
                play_by_play = json.load(file)

            # check if the game is live
            if play_by_play['gameState'] == 'OFF':
                live = False

            # feature engineering, clean, transform from json to df
            df_for_pred = generate_game_client_df(f'{file_name}')
            df = self.extract_features(play_by_play)
            last_row = df.tail(1)
            ping = last_row['LastPing'].values[0]

            period = last_row['Period'].values[0]
            timeLeft = last_row['TimeLeft'].values[0]
            home_team = play_by_play['homeTeam']['name']['default']
            away_team = play_by_play['awayTeam']['name']['default']
            home_score = play_by_play['homeTeam']['score']
            away_score = play_by_play['awayTeam']['score']

            previous_idx = 0
            if game_id in self.tracker:
                previous_idx = self.tracker[str(game_id)].get('idx', 0)
                self.tracker[str(game_id)]['idx'] = len(df_for_pred)
            else:
                self.tracker[str(game_id)] = {'idx': len(df_for_pred)}

            with open('tracker.json', 'w') as outfile:
                json.dump(self.tracker, outfile)

            df_for_pred = df_for_pred.reset_index().drop('index', axis=1)[previous_idx:]

            return df_for_pred, live, period, timeLeft, home_team, away_team, home_score, away_score
        else:
            print(f"Failed to retrieve live game data for game_id {game_id}")
            return None

# Testing
game_id_to_test = 2019020003 # 2022020451 #
client = GameClient()
result_df, live, period, timeLEft, home, away, h_score, a_score = client.ping_game(game_id_to_test)

print(result_df)
print(live)
print(period)
print(timeLEft)
print(home)
print(away)
print(h_score)
print(a_score)