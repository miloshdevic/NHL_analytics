import pandas as pd
import numpy as np
import json
import os as os
from datetime import datetime, time, date

# keys:
# ['id', 'season', 'gameType', 'gameDate', 'venue', 'startTimeUTC', 'easternUTCOffset', 'venueUTCOffset',
# 'tvBroadcasts', 'gameState', 'gameScheduleState', 'period', 'periodDescriptor', 'awayTeam', 'homeTeam', 'clock',
# 'rosterSpots', 'displayPeriod', 'gameOutcome', 'plays'])
def print_keys(data, indent=0):
    if isinstance(data, dict):
        for key, value in data.items():
            print('  ' * indent + str(key))
            print_keys(value, indent + 1)
    elif isinstance(data, list):
        for item in data:
            print_keys(item, indent)
    else:
        pass  # Handle other types as needed

def opposite(direction):
    if direction == 'right':
        return 'left'
    elif direction == 'left':
        return 'right'


def create_event_dataframe(file_path):
    # Load .json File
    with open(file_path, 'r') as file:
        playByplay = json.load(file)

    home_team_id = playByplay['homeTeam']['id']
    away_team_id = playByplay['awayTeam']['id']
    teams = {home_team_id: playByplay['homeTeam']['name']['default'],
             away_team_id: playByplay['awayTeam']['name']['default']}

    # Initialize lists to store information
    game_time = []
    period = []
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
    s = playByplay['season']
    id = playByplay['id']
    print(id)

    # if id != 2019020451:
    #     df = pd.DataFrame({
    #         'GameTime': game_time,  # done
    #         'Period': period,  # done
    #         'GameID': game_id_list,  # done
    #         'Team': team,  # done
    #         'Event': event_type,  # done
    #         'XCoord': x_coord,  # done
    #         'YCoord': y_coord,  # done
    #         'ShotType': shot_type,  # done
    #         'isGoal': isGoal,  # done
    #         'isEmptyNet': isEmptyNet,  # done
    #         'Strength': strength,  # done
    #         'RinkSide': rinkSide,  # done
    #         'Season': season  # done
    #     })
    #     return df

    for event in playByplay['plays']:
        print(event['eventId'])
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
        if event['typeDescKey'] == "shot-on-goal" and 'shotType' in event['details']:  # sometimes API is missing values
            shot_type.append(event['details']['shotType'])
        else:
            shot_type.append('')

        isGoal.append(int(event['typeDescKey'] == 'goal'))
        game_time.append(event['timeInPeriod'])
        period.append(event['period'])

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

            if int(away_goalie) == 0 and int(event['typeDescKey'] == 'goal') and event['details']['eventOwnerTeamId'] == home_team_id:
                isEmptyNet.append(1)
            elif int(home_goalie) == 0 and int(event['typeDescKey'] == 'goal') and event['details']['eventOwnerTeamId'] == away_team_id:
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


def pivot_for_shots_and_goals(df: pd.DataFrame) -> pd.DataFrame:
    # extract specific events
    sng_df = df[df['Event'].isin(['shot-on-goal', 'goal', 'faceoff', 'hit', 'giveaway', 'missed-shot',
                                  'blocked-shot', 'takeaway', 'penalty', 'fight'])]
    # pivot the DataFrame
    sng_df = sng_df.pivot_table(index=['GameID', 'Season', 'Event', 'Period', 'GameTime', 'Team'], columns=None,
                                values=['XCoord', 'YCoord', 'ShotType', 'isGoal', 'isEmptyNet',
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
        if row['Event'] in ['goal', 'shot']:
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
        if row['Event'] in ['goal', 'shot']:
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
def add_rebound(df: pd.DataFrame) -> pd.DataFrame:
    rebound = np.full((df.shape[0]), False)

    previous_row = None
    i = 0
    for j, row in df.iterrows():
        if i == 0:
            previous_row = row
            i += 1
            continue

        if row['Event'] == 'shot' and previous_row['Event'] == 'shot':
            rebound[i] = True

        previous_row = row
        i += 1

    df['Rebound'] = rebound
    return df


# adds angle change
# only include if the shot is a rebound, otherwise 0
def angle_change(df: pd.DataFrame) -> pd.DataFrame:
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
def add_speed(df: pd.DataFrame) -> pd.DataFrame:
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
def add_game_seconds(df: pd.DataFrame) -> pd.DataFrame:
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


def game_client(file_path) -> pd.DataFrame:
    df = create_event_dataframe(file_path)

    # keep only shots and goals
    df_sng = df[df['Event'].isin(['shot', 'goal'])]

    # add distance and angle columns
    df_sng = add_distance(df_sng)
    df_sng = add_angle(df_sng)

    # keep only the selected columns
    df_sng = df_sng[['isEmptyNet', 'isGoal', 'DistanceToGoal', 'ShootingAngle']]

    return df_sng


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
    # run_tidy_data(folder_test)
