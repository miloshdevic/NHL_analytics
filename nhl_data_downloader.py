import requests
import numpy as np
import pandas as pd
import os
import json


class NHLDataDownloader:
    def __init__(self, season, data_dir='data'):
        self._season = None
        self.season = season
        self.data_dir = data_dir

        # codes for types of games
        self.pre_season = '01'
        self.reg_season = '02'
        self.playoffs = '03'
        self.all_star = '04'

        # urls
        self.base_url = 'https://statsapi.web.nhl.com/api/v1/game/'
        self.end_base_url = '/feed/live/'
        self.cache_file = os.path.join(self.data_dir, f'nhl_play_by_play_{self.season}.json')

    @property
    def season(self):
        return self._season

    @season.setter
    def season(self, value):
        self._season = value

        # Update nb_games_reg_season based on the new season value
        if int(self.season) < 2017:
            self.nb_games_reg_season = 1230
        elif 2017 <= int(self.season) <= 2020:
            self.nb_games_reg_season = 1271
        else:
            self.nb_games_reg_season = 1353

    def download_nhl_data(self, game_id: str):
        """
        Creates and saves a .json file of the play-by-play of a game (defined by the game ID, aka game_id) in the
        specified directory. If the file has already been downloaded there before, it skips it and continues on to
        the next one.

        Some playoff games have a .json file even though that game never took place (example: games 6 and 7 of a series
        that finished 4-1 have existing files but without data in it). This function filters them out by checking the
        nested value of 'status'. If the value is "Final", the game happened.
        """
        if os.path.exists(f'nhl_play_by_play_{self.season}_{game_id}.json'):
            with open(self.cache_file, 'r') as file:
                return True  # json.load(file)

        os.makedirs(self.data_dir, exist_ok=True)
        url = f'{self.base_url}{game_id}{self.end_base_url}'
        response = requests.get(url)

        if response.status_code == 200:
            data = response.json()
            detailed_state = data.get('gameData', {}).get('status', {}).get('detailedState', '')
            if detailed_state == 'Final':
                with open(f'{self.data_dir}/nhl_play_by_play_{self.season}_{game_id}.json', 'w') as file:
                    json.dump(data, file)
                return True  # data
            else:
                print(f"Skipping download for game ID {game_id} as detailedState is not 'Final'")
                return False
        else:
            print(f"Failed to download data for game ID {game_id}")
            return False

    def get_nhl_data_season(self):
        """
        Loops through all the games of a given regular season and creates the appropriate game ID in each loop.
        """
        for i in range(1, self.nb_games_reg_season + 1):
            game_id = f'{self.season}{self.reg_season}{i:04}'
            self.download_nhl_data(game_id)

    def get_nhl_data_playoffs(self):
        """
        Loops through all the games of a given playoff season and creates the appropriate game ID in each loop.

        It checks if a game is possible and updates the value accordingly (example: game 8 of matchup 2 in round 1 isn't
        possible, or game 1 of matchup 10 in round 1 isn't possible, or game 1 of matchup 1 of round 5 also isn't
        possible).
        """
        round = 1
        matchup = 1
        game = 1

        for i in range(105):  # maximum number of possible playoff games is 105
            game_id = f'{self.season}{self.playoffs}0{round}{matchup}{game}'
            downloaded = self.download_nhl_data(game_id)
            game += 1

            # the following 'if' statements make sure that we don't iterate impossible scenarios
            # (ex: round 5, matchup 10, game 8)
            if game > 7 or not downloaded:
                matchup += 1
                game = 1

            if matchup > 8 and round == 1 or \
                    matchup > 4 and round == 2 or \
                    matchup > 2 and round == 3 or \
                    matchup > 1 and round == 4:
                round += 1
                matchup = 1
                game = 1

            if round > 4:
                return


if __name__ == '__main__':
    # Set the seasons and data directory
    nhl_season_start = 2016  # USER INPUT HERE FOR STARTING YEAR
    nhl_season_end = 2020  # USER INPUT HERE FOR ENDING YEAR
    data_directory = 'nhl_data'  # USER INPUT HERE FOR FOLDER WHERE DATA WILL BE SAVED

    # Create an instance of NHLPlayByPlayDownloader
    nhl_downloader = NHLDataDownloader(str(nhl_season_start), data_directory)

    # download NHL play-by-play data for the specified seasons
    for i in range(nhl_season_start, nhl_season_end + 1):
        nhl_downloader.season = str(i)
        nhl_downloader.get_nhl_data_season()
        nhl_downloader.get_nhl_data_playoffs()
