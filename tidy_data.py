import pandas as pd
import numpy as np
import json
import os as os
from datetime import datetime, time, date


def create_event_dataframe(file_path):
    # Load .json File
    with open(file_path, 'r') as file:
        playByplay = json.load(file)
    # print(playByplay['gamePk'])
    # Initialize lists to store information
    game_time = []
    period = []
    game_id_list = []
    team = []
    event_type = []
    x_coord = []
    y_coord = []
    player_name = []
    shooter = []
    goalie = []
    shot_type = []
    isGoal = []
    isEmptyNet = []
    strength = []
    rinkSide = []
    season = []
    # for rinkSide information classification
    home_team = playByplay['gameData']['teams']['home']['name']
    # Extract relevant information for each event
    for event in playByplay['liveData']['plays']['allPlays']:
        if 'result' in event and 'eventTypeId' in event['result']:
            season.append(
                playByplay['gameData']['game']['season'][0:4] + '-' + playByplay['gameData']['game']['season'][4:8])
            # game time not date/time
            game_time.append(event['about']['periodTime'])
            period.append(event['about']['period'])
            game_id_list.append(playByplay['gamePk'])
            team.append(event['team']['name'] if 'team' in event else None)
            home_away = 'home' if team[-1] == home_team else 'away'
            if event['about']['period'] < 5:  # no rinkside for shootout
                # no rinkSide information for some games for example 2018020666 and 2020020177 between 'Vegas Golden
                # Knights' and 'Los Angeles Kings'
                if 'rinkSide' not in playByplay['liveData']['linescore']['periods'][period[-1] - 1][home_away]:
                    rinkSide.append(None)
                else:
                    rinkSide.append(
                        playByplay['liveData']['linescore']['periods'][period[-1] - 1][home_away]['rinkSide'] if
                        team[-1] is not None else None)
            else:
                rinkSide.append(None)
            event_type.append(event['result']['eventTypeId'])
            isGoal.append(int(event['result']['eventTypeId'] == 'GOAL') if 'secondaryType' in event['result'] else 0)
            x_coord.append(event['coordinates']['x'] if 'x' in event['coordinates'] else None)
            y_coord.append(event['coordinates']['y'] if 'y' in event['coordinates'] else None)
            # to get all palyer names for that event
            players = event.get('players', [])
            player_names = [player['player']['fullName'] for player in players]
            player_name.append(', '.join(player_names) if player_names else None)
            # shot and goals
            if event['result']['eventTypeId'] in ['SHOT', 'GOAL']:
                for player in players:
                    if player['playerType'] in ['Shooter', 'Scorer']:
                        shooter.append(player['player']['fullName'])
                    elif player['playerType'] in ['Goalie']:
                        goalie.append(player['player']['fullName'])
                    else:
                        ''
                shot_type.append(event['result']['secondaryType'] if 'secondaryType' in event['result'] else None)

                if event['result']['eventTypeId'] in ['GOAL']:
                    # only GOALs in regular time and OT has 'emptyNet' character
                    if event['about']['period'] < 5:
                        # isEmptyNet.append(event['result']['emptyNet'])
                        isEmptyNet.append(event['result']['emptyNet'])
                        # correcting for empty net goals
                        if event['result']['emptyNet'] == True:
                            goalie.append(None)
                        elif not any('Goalie' in player['playerType'] for player in players):
                            goalie.append(None)
                    else:
                        # Shootout goals are False by default
                        isEmptyNet.append(0)
                    strength.append(event['result']['strength']['name'])
                else:  # if no Goalie for a shot
                    if not any('Goalie' in player['playerType'] for player in players):
                        goalie.append(None)
                    # to have equal length
                    isEmptyNet.append(0)
                    strength.append(None)
            else:
                # to have equal length
                shooter.append(None)
                goalie.append(None)
                shot_type.append(None)
                isEmptyNet.append(0)
                strength.append(None)
    # Create a DataFrame from the extracted information
    df = pd.DataFrame({
        'GameTime': game_time,
        'Period': period,
        'GameID': game_id_list,
        'Team': team,
        'Event': event_type,
        'XCoord': x_coord,
        'YCoord': y_coord,
        'PlayerName': player_name,
        'Shooter': shooter,
        'Goalie': goalie,
        'ShotType': shot_type,
        'isGoal': isGoal,
        'isEmptyNet': [int(x) for x in isEmptyNet],
        'Strength': strength,
        'RinkSide': rinkSide,
        'Season': season
    })
    return df


def pivot_for_shots_and_goals(df):
    # extract SHOT and GOAL events
    sng_df = df[df['Event'].isin(['SHOT', 'GOAL', 'FACEOFF', 'HIT', 'GIVEAWAY', 'MISSED_SHOT',
                                  'BLOCKED_SHOT', 'PENALTY', 'TAKEAWAY', 'PENALTY', 'FIGHT'])]
    # pivot the shots and goal(sng) DataFrame
    sng_df = sng_df.pivot_table(index=['GameID', 'Season', 'Event', 'Period', 'GameTime', 'Team'], columns=None,
                                values=['XCoord', 'YCoord', 'Shooter', 'Goalie', 'ShotType', 'isGoal', 'isEmptyNet',
                                        'Strength', 'RinkSide'], aggfunc='first')
    return sng_df


# Calculates the distance between a shot/goal and the net, rounded to the nearest number
# the column 'distance_to_goal' is added to the df
def add_distance(df: pd.DataFrame) -> pd.DataFrame:
    right_goal = [89, 0]
    left_goal = [-89, 0]
    distance_to_goal = np.zeros(df.shape[0])

    i = 0
    for j, row in df.iterrows():
        if row['Event'] in ['GOAL', 'SHOT']:
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
def add_angle(df: pd.DataFrame) -> pd.DataFrame:
    right_goal = [89, 0]
    left_goal = [-89, 0]
    shooting_angle = np.zeros(df.shape[0])

    i = 0
    for j, row in df.iterrows():
        if row['Event'] in ['GOAL', 'SHOT']:
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
                shooting_angle[i] = None  # some games didn't have the information for which side the team was defending
        else:
            shooting_angle[i] = None
        i += 1

    # add the column with its values
    df['ShootingAngle'] = shooting_angle
    return df


# adds information from the previous events to each shot
def add_previous_events(df: pd.DataFrame) -> pd.DataFrame:
    last_event = np.full((df.shape[0]), None)
    last_event_XCoord = np.zeros(df.shape[0])
    last_event_YCoord = np.zeros(df.shape[0])
    time_last_event = np.zeros(df.shape[0])
    distance_last_event = np.zeros(df.shape[0])

    last_event[0] = None
    last_event_XCoord[0] = None
    last_event_YCoord[0] = None
    time_last_event[0] = None
    distance_last_event[0] = None

    previous_row = None
    i = 0
    for j, row in df.iterrows():
        if i == 0:
            previous_row = row
            i += 1
            continue

        # if it is a different game
        if row['GameID'] != previous_row['GameID']:
            last_event[i] = None
            last_event_XCoord[i] = None
            last_event_YCoord[i] = None
            time_last_event[i] = None
            distance_last_event[i] = None
            previous_row = row
            i += 1
            continue

        # update new columns
        last_event[i] = previous_row['Event']
        last_event_XCoord[i] = previous_row['XCoord']
        last_event_YCoord[i] = previous_row['YCoord']
        distance_last_event[i] = np.sqrt(
            (row['XCoord'] - previous_row['XCoord']) ** 2 + (row['YCoord'] - previous_row['YCoord']) ** 2).round()
        time_last_event[i] = row['GameTime'] - previous_row['GameTime']

        # update variables
        previous_row = row
        i += 1

    df['LastEvent'] = last_event
    df['LastEvent_XCoord'] = last_event_XCoord
    df['LastEvent_YCoord'] = last_event_YCoord
    df['TimeLastEvent'] = time_last_event
    df['DistanceLastEvent'] = distance_last_event
    return df


# adds rebound column
# True if the last event was also a shot, otherwise False
def add_rebound(df) -> pd.DataFrame:
    rebound = np.full((df.shape[0]), False)

    previous_row = None
    i = 0
    for j, row in df.iterrows():
        if i == 0:
            previous_row = row
            i += 1
            continue

        if row['Event'] == 'SHOT' and previous_row['Event'] == 'SHOT':
            rebound[i] = True

        previous_row = row
        i += 1

    df['Rebound'] = rebound
    return df


# adds angle change
# only include if the shot is a rebound, otherwise 0
def angle_change(df) -> pd.DataFrame:
    angle_diff = np.zeros(df.shape[0])

    previous_row = None
    i = 0
    for j, row in df.iterrows():
        if i == 0:
            previous_row = row
            i += 1
            continue

        if row['Rebound']:
            if (row['ShootingAngle'] < 0 and previous_row['ShootingAngle'] > 0) or (
                    row['ShootingAngle'] > 0 and previous_row['ShootingAngle'] < 0):
                angle_diff[i] = np.abs(row['ShootingAngle']) + np.abs(previous_row['ShootingAngle'])
            else:
                angle_diff[i] = np.abs(row['ShootingAngle'] - previous_row['ShootingAngle'])

        previous_row = row
        i += 1

    df['AngleChange'] = angle_diff
    return df


# adds speed column
# defined as the distance from the previous event, divided by the time since the previous event
def add_speed(df) -> pd.DataFrame:
    speed = np.zeros(df.shape[0])

    i = 0
    for j, row in df.iterrows():
        if i == 0:
            i += 1
            continue

        if row['TimeLastEvent'] != 0:
            speed[i] = row['DistanceLastEvent'] / row['TimeLastEvent']
        i += 1

    df['Speed'] = speed
    return df


# modifies 'GameTime' from datetime to total number of seconds elapsed in the game (float)
def add_game_seconds(df):
    # transform data type into a datetime.time object
    df['GameTime'] = df['GameTime'].apply(lambda x: datetime.strptime(x, '%M:%S').time())

    seconds = np.zeros(df.shape[0])
    previous_row = None

    i = 0
    for j, row in df.iterrows():
        if i == 0:
            seconds[i] = 0
            previous_row = row
            i += 1
            continue

        if row['GameID'] != previous_row['GameID']:
            seconds[i] = 0
            previous_row = row
            i += 1
            continue

        if row['Period'] == 1:
            seconds[i] = (datetime.combine(date.today(), row['GameTime']) -
                          datetime.combine(date.today(), time(0, 0, 0))).total_seconds()

        elif row['Period'] == 2:
            time1 = (datetime.combine(date.today(), row['GameTime']) -
                     datetime.combine(date.today(), time(0, 0, 0))).total_seconds()
            time2 = (datetime.combine(date.today(), time(0, 20, 0)) -
                     datetime.combine(date.today(), time(0, 0, 0))).total_seconds()
            seconds[i] = time1 + time2

        elif row['Period'] == 3:
            time1 = (datetime.combine(date.today(), row['GameTime']) -
                     datetime.combine(date.today(), time(0, 0, 0))).total_seconds()
            time2 = (datetime.combine(date.today(), time(0, 40, 0)) -
                     datetime.combine(date.today(), time(0, 0, 0))).total_seconds()
            seconds[i] = time1 + time2

        previous_row = row
        i += 1

    df['GameTime'] = seconds
    return df


# get raw data into a tidied csv file
def run_tidy_data(folder):
    fileList = [f for f in os.listdir(folder)]
    all_sng_df = pd.DataFrame()
    for file in fileList:
        if file.endswith('.json'):
            # get all events, then pivot for shots and goals
            tmp_df = pivot_for_shots_and_goals(create_event_dataframe(f'{folder}/{file}'))
            # tmp_df = create_event_dataframe(f'{folder}/{file}')

            # stacking all dataframes
            all_sng_df = pd.concat([all_sng_df, tmp_df])

    # sort by GameID
    asngsorted_df = all_sng_df.sort_values(by=['GameID', 'Period', 'GameTime'])
    asngsorted_df.reset_index()
    asngsorted_df.to_csv(f'{folder}.csv')
    # asngsorted_df['GameTime'] = asngsorted_df['GameTime'].apply(lambda x: datetime.strptime(x, '%M:%S').time())


if __name__ == '__main__':
    folder_train = 'nhl_data_train'
    folder_test = 'nhl_data_test'
    run_tidy_data(folder_train)
    run_tidy_data(folder_test)
