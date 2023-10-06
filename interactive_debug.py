import ipywidgets as widgets
from IPython.display import display
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd

def get_game_output(game_id):
    game_id = str(game_id)
    file_path = "nhl_data/nhl_play_by_play_"+game_id[:4]+'_'+game_id+".json"
    df = pd.read_json(file_path)

    away_team = df["liveData"]["boxscore"]["teams"]["away"]["team"]["abbreviation"]
    away_goals = df["liveData"]["boxscore"]["teams"]["away"]["teamStats"]["teamSkaterStats"]["goals"]
    home_team = df["liveData"]["boxscore"]["teams"]["home"]["team"]["abbreviation"]
    home_goals = df["liveData"]["boxscore"]["teams"]["home"]["teamStats"]["teamSkaterStats"]["goals"]
    df = pd.DataFrame({'Home': [home_team, home_goals],'Away': [away_team, away_goals]})
    
    return df

def get_event_output(game_id, event_id):
    game_id = str(game_id)
    file_path = "nhl_data/nhl_play_by_play_"+game_id[:4]+'_'+game_id+".json"
    with open(file_path, 'r') as f:
      data = json.load(f)
    one_event_data = data["liveData"]["plays"]["allPlays"][event_id]
    del one_event_data["about"]["goals"]
    return one_event_data

def plot_ice_rink_with_events(game_id, event_id):
    # Load the local ice rink image
    rink_image_path = 'nhl_data/nhl_rink.png'
    img = mpimg.imread(rink_image_path)
    plt.axis('off')  # Hide axis numbers and ticks
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(img, extent=[-100, 100, -42.5, 42.5])
    event_coordinates = get_event_output(game_id, event_id)["coordinates"]
    if event_coordinates != {}:
        x, y = event_coordinates["x"], event_coordinates["y"]
        plt.plot(x, y, 'ro')
    
    plt.show()
    

# Function to update the game based on the selected game ID
def update_game(game_id, event_id):
    print(f"Loading game {game_id} with event {event_id}...")
    
    # Simulate getting event data for the game (replace with actual data)
    print(get_game_output(game_id))
    plot_ice_rink_with_events(game_id, event_id)
    print(get_event_output(game_id, event_id))

# Widget to select the game ID
game_id_text = widgets.IntText(
    value=0,
    description='Game ID:',
    disabled=False
)
event_id_text = widgets.IntText(
    value=0,
    description='Event ID:',
    disabled=False
)

# Button to load the selected game
load_button = widgets.Button(
    description='Load Game',
    button_style='success',
    tooltip='Load the selected game',
)

def load_game(button):
    game_id = game_id_text.value
    event_id = event_id_text.value
    update_game(game_id, event_id)


## For testing

# def get_game_output(game_id):
#     game_id = str(game_id)
#     file_path = "nhl_data/nhl_play_by_play_"+game_id[:4]+'_'+game_id+".json"
#     df = pd.read_json(file_path)

#     away_team = df["liveData"]["boxscore"]["teams"]["away"]["team"]["abbreviation"]
#     away_goals = df["liveData"]["boxscore"]["teams"]["away"]["teamStats"]["teamSkaterStats"]["goals"]
#     home_team = df["liveData"]["boxscore"]["teams"]["home"]["team"]["abbreviation"]
#     home_goals = df["liveData"]["boxscore"]["teams"]["home"]["teamStats"]["teamSkaterStats"]["goals"]
#     df = pd.DataFrame({'Home': [home_team, home_goals],'Away': [away_team, away_goals]})
    
#     return df
    
# game_id = 2016020001
# print(get_game_output(game_id))


if __name__ == '__main__':
    load_button.on_click(load_game)

	# Display the widgets
	display(game_id_text, event_id_text, load_button)

	#game_id: 2016020001
	#event_id: 8

