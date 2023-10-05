import pandas as pd
import json
import os as os

def create_event_dataframe(file_path):
    # Load .json File
    with open(file_path, 'r') as file:
        playByplay=json.load(file)
    print(playByplay['gamePk'])
    # Initialize lists to store information
    game_time=[]
    period=[]
    game_id_list=[]
    team=[]
    event_type=[]
    x_coord=[]
    y_coord=[]
    player_name=[]
    shooter=[]
    goalie=[]
    shot_type=[]
    isEmptyNet=[]
    strength=[]
    home_away=[]
    # for home_away classification
    home_team=playByplay['gameData']['teams']['home']['name']
    # Extract relevant information for each event
    for event in playByplay['liveData']['plays']['allPlays']:
        if 'result' in event and 'eventTypeId' in event['result']:
            # game time not date/time
            game_time.append(event['about']['periodTime'])
            period.append(event['about']['period'])
            game_id_list.append(playByplay['gamePk'])
            team.append(event['team']['name'] if 'team' in event else None)
            home_away.append('home' if team[-1]==home_team else 'away')
            event_type.append(event['result']['eventTypeId'])
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
                    else: ''
                shot_type.append(event['result']['secondaryType'] if 'secondaryType' in event['result'] else None)
                if event['result']['eventTypeId'] in ['GOAL']:
                    # only GOALs in regular time and OT has 'emptyNet' character
                    if event['about']['period']<5:
                        isEmptyNet.append(event['result']['emptyNet'])
                        # correcting for empty net goals
                        if event['result']['emptyNet']==True:
                            goalie.append(None)
                        elif not any('Goalie' in player['playerType'] for player in players): 
                            goalie.append(None)
                    else:
                        # Shootout goals are False by default
                        isEmptyNet.append(False)
                    strength.append(event['result']['strength']['name'])
                else: 
                    if not any('Goalie' in player['playerType'] for player in players): 
                        goalie.append(None)
                    # to have equal length
                    isEmptyNet.append(False)
                    strength.append(None)
            else: 
                # to have equal length
                shooter.append(None)
                goalie.append(None)
                shot_type.append(None)
                isEmptyNet.append(False)
                strength.append(None)
    # Create a DataFrame from the extracted information
    df=pd.DataFrame({
        'GameTime': game_time,
        'Period': period,
        'GameID': game_id_list,
        'Team': team,
        'Event': event_type,
        'XCoord': x_coord,
        'YCoord': y_coord,
        'PlayerName': player_name,
        'Shooter/Scorer': shooter,
        'Goalie': goalie,
        'ShotType': shot_type,
        'IsEmptyNet': isEmptyNet,
        'Strength': strength,
        'Home/Away': home_away
    })
    return df


def pivot_for_shots_and_goals(df):
    # extract SHOT and GOAL events
    sng_df=df[df['Event'].isin(['SHOT', 'GOAL'])]
    # pivot the shots and goal(sng) DataFrame
    chrono_ordered=sng_df.pivot_table(index=['GameID', 'Period', 'GameTime', 'Team', 'Home/Away'], columns='Event', values=['XCoord', 'YCoord', 'Shooter/Scorer', 'Goalie', 'ShotType', 'IsEmptyNet', 'Strength'], aggfunc='first')
    sng_df=sng_df.pivot_table(index=['GameID', 'Event', 'Period', 'GameTime', 'Team', 'Home/Away'], columns=None, values=['XCoord', 'YCoord', 'Shooter/Scorer', 'Goalie', 'ShotType', 'IsEmptyNet', 'Strength'], aggfunc='first')
    return sng_df


if __name__ == '__main__':
    fileList=[f for f in os.listdir('nhl_data')]
    all_sng_df=pd.DataFrame()
    for file in fileList:
        if file.endswith('.json'):
            # get all events, then pivot for shots and goals
            tmp_df=pivot_for_shots_and_goals(create_event_dataframe('nhl_data/'+file))
            # stacking all dataframes
            all_sng_df=pd.concat([all_sng_df, tmp_df])
    # sort by GameID
    asngsorted_df=all_sng_df.sort_values(by=['GameID'])
    asngsorted_df.to_csv(os.getcwd()+'/tidied_nhl.csv')

