import pandas as pd
import numpy as np
import requests
import json
import os
import logging

logger = logging.getLogger(__name__)

class GameClient:
    def __init__(self):
        self.tracker = self.load_tracker()
        logger.info(f"Initializing ClientGame; base URL: ")

    def load_tracker(self):
        try:
            with open('tracker.json', 'r') as tracker_file:
                return json.load(tracker_file)
        except FileNotFoundError:
            # If tracker file doesn't exist, create it
            with open('tracker.json', 'w') as outfile:
                data = {}
                json.dump(data, outfile)
            return {}

    # returns the opposite of left or right
    def opposite(self, direction):
        if direction == 'right':
            return 'left'
        elif direction == 'left':
            return 'right'

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
                                else self.opposite(event['homeTeamDefendingSide']))
            elif 'details' in event and 'eventOwnerTeamId' in event['details']:
                rinkSide.append(event['homeTeamDefendingSide'] if event['details']['eventOwnerTeamId'] == home_team_id
                                else self.opposite(event['homeTeamDefendingSide']))
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
            'Season': season
        })

        return df

    # Calculates the distance between a shot/goal and the net, rounded to the nearest number
    # the column 'distance_to_goal' is added to the df
    def add_distance(self, df: pd.DataFrame) -> pd.DataFrame:
        right_goal = [89, 0]
        left_goal = [-89, 0]
        distance_to_goal = np.zeros(df.shape[0])

        i = 0
        for j, row in df.iterrows():
            if row['Event'] in ['goal', 'shot-on-goal']:
                if row['RinkSide'] == 'right':
                    distance_to_goal[i] = np.sqrt(
                        (row['XCoord'] - left_goal[0]) ** 2 + (row['YCoord'] - left_goal[1]) ** 2).round()

                elif row['RinkSide'] == 'left':
                    distance_to_goal[i] = np.sqrt(
                        (row['XCoord'] - right_goal[0]) ** 2 + (row['YCoord'] - right_goal[1]) ** 2).round()

                else:
                    distance_to_goal[
                        i] = None  # some games didn't have the information for which side the team was defending
            else:
                distance_to_goal[i] = None
            i += 1

        # add the column with its values
        df['DistanceToGoal'] = distance_to_goal
        return df

    # Calculates the angle between a shot/goal and the net
    # the column 'shooting_angle' is added to the df
    def add_angle(self, df: pd.DataFrame) -> pd.DataFrame:
        right_goal = [89, 0]
        left_goal = [-89, 0]
        shooting_angle = np.zeros(df.shape[0])

        i = 0
        for j, row in df.iterrows():
            if row['Event'] in ['goal', 'shot-on-goal']:
                if row['RinkSide'] == 'right':
                    # if (left_goal[0] - row['XCoord']) != 0:
                    if row['YCoord'] != 0:
                        shooting_angle[i] = (
                                np.arctan((left_goal[0] - row['XCoord']) / row['YCoord']) * (180 / np.pi)).round()
                    else:
                        shooting_angle[i] = 0

                elif row['RinkSide'] == 'left':
                    # if (-right_goal[0] - row['XCoord']) != 0:
                    if row['YCoord'] != 0:
                        shooting_angle[i] = (
                                np.arctan((right_goal[0] - row['XCoord']) / row['YCoord']) * (180 / np.pi)).round()
                    else:
                        shooting_angle[i] = 0

                else:
                    shooting_angle[
                        i] = None  # some games didn't have the information for which side the team was defending
            else:
                shooting_angle[i] = None
            i += 1

        # add the column with its values
        df['ShootingAngle'] = shooting_angle
        return df

    def generate_game_client_df(self, file_path) -> pd.DataFrame:
        with open(file_path, 'r') as file:
            play_by_play = json.load(file)
        df = self.extract_features(play_by_play)

        # keep only shots and goals
        df_sng = df[df['Event'].isin(['shot-on-goal', 'goal'])]

        # add distance and angle columns
        df_sng = self.add_distance(df_sng)
        df_sng = self.add_angle(df_sng)

        # keep only the selected columns
        df_sng = df_sng[['DistanceToGoal', 'ShootingAngle', 'Team']]

        return df_sng

    def generate_bonus_df(self, file_path) -> pd.DataFrame:
        with open(file_path, 'r') as file:
            play_by_play = json.load(file)
        df = self.extract_features(play_by_play)

        # keep only shots and goals
        df_sng = df[df['Event'].isin(['shot-on-goal', 'goal'])]

        return df_sng

    def ping_game(self, game_id: int):
        live = False
        url_game = f'https://api-web.nhle.com/v1/gamecenter/{int(game_id)}/play-by-play/'
        response = requests.get(url_game)

        id = str(game_id)

        file_name = f'nhl_play_by_play_{id[:4]}_{game_id}.json'

        data = response.json()
        with open(file_name, 'w') as file:
            json.dump(data, file)

        with open(file_name, 'r') as file:
            play_by_play = json.load(file)

        # check if the game is live
        if play_by_play['gameState'] == 'LIVE':
            live = True

        # feature engineering, clean, transform from json to df
        df_for_pred = self.generate_game_client_df(f'{file_name}')
        df = self.extract_features(play_by_play)
        last_row = df.tail(1)

        period = last_row['Period'].values[0]
        timeLeft = last_row['TimeLeft'].values[0]
        home_team = play_by_play['homeTeam']['name']['default']
        away_team = play_by_play['awayTeam']['name']['default']
        home_score = play_by_play['homeTeam']['score']
        away_score = play_by_play['awayTeam']['score']

        with open('tracker.json', 'r') as t:
            tracker = json.load(t)

        previous_idx = 0
        if str(game_id) in tracker:
            previous_idx = tracker.get(str(game_id), {}).get("idx") # [str(game_id)]['idx']
            tracker[str(game_id)]['idx'] = len(df_for_pred)
        else:
            tracker[str(game_id)] = {}
            tracker[str(game_id)]['idx'] = len(df_for_pred)
        with open('tracker.json', 'w') as outfile:
            json.dump(tracker, outfile)

        df_for_pred = df_for_pred.reset_index().drop('index', axis=1)[previous_idx:]

        # previous_idx = 0
        # if game_id in self.tracker:
        #     previous_idx = self.tracker[str(game_id)].get('idx', 0)
        #     self.tracker[str(game_id)]['idx'] = len(df_for_pred)
        # else:
        #     self.tracker[str(game_id)] = {'idx': len(df_for_pred)}
        #
        # with open('tracker.json', 'w') as outfile:
        #     json.dump(self.tracker, outfile)
        #
        # df_for_pred = df_for_pred.reset_index().drop('index', axis=1)[previous_idx:]

        return df_for_pred, live, period, timeLeft, home_team, away_team, home_score, away_score

    def ping_game_bonus(self, game_id: int):
        url_game = f'https://api-web.nhle.com/v1/gamecenter/{int(game_id)}/play-by-play/'
        response = requests.get(url_game)

        id = str(game_id)

        file_name = f'nhl_play_by_play_{id[:4]}_{game_id}.json'

        data = response.json()
        with open(file_name, 'w') as file:
            json.dump(data, file)

        with open(file_name, 'r') as file:
            play_by_play = json.load(file)

        # feature engineering, clean, transform from json to df
        df_for_pred = self.generate_game_client_df(f'{file_name}')
        df = self.extract_features(play_by_play)

        # previous_idx = 0
        # if game_id in self.tracker:
        #     previous_idx = self.tracker[str(game_id)].get('idx', 0)
        #     self.tracker[str(game_id)]['idx'] = len(df_for_pred)
        # else:
        #     self.tracker[str(game_id)] = {'idx': len(df_for_pred)}
        #
        # with open('tracker.json', 'w') as outfile:
        #     json.dump(self.tracker, outfile)
        #
        # df_for_pred = df_for_pred.reset_index().drop('index', axis=1)[previous_idx:]

        return df
