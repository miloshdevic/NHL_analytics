---
layout: post
title:  "NHL Data Analysis Milestone 1"
date:  2023-10-15
tags: [Download data from API, data cleaning]
---
## Introduction

In this project, we utilized the NHL Stats API to retrieve comprehensive "play-by-play" data for a specific time period. Our primary objective was to generate insightful plots and interactive visualizations that provide users with a deeper understanding of a particular game, various events within the game, shot types, their success rates relative to the distance from the net, and shot rates with respect to distance using heat maps. Furthermore, we aimed to illustrate how frequently a team shoots from specific spots compared to the NHL average team in the same season using a shot map. In addition to gathering and preprocessing the data, using tools for data retrieval and manipulation, we engaged in in-depth discussions about the various plots and interactive visualizations we created. All these visual representations will be showcased on a static web page using Jekyll.

## 1. Data Acquisition

### How to Download NHL Play-by-Play Data

If you are looking to download NHL but don’t know how, here is a quick tutorial on how to do it! We will walk you step by step through the process by using the Python code provided.

<!--more-->

### 1.1 Requirements

Before you start, make sure you have the following:

1.	Python installed on your computer.
2.	The required Python libraries installed (they can be found in the file "requirements.txt") for the virtual environment.
3.	Access to the internet to download the NHL data from the API.

### Step 1: Setting Up the Environment

Start by importing the necessary libraries and creating a class that will help us download the data. In the provided code, the ‘NHLDataDownloader’ class handles the data download process.

{% include image_full.html imageurl="/images/milestone1/download_step1.png" caption="setting up the environment" %}

### Step 2: Create an Instance of the Downloader

Now, create an instance of the ‘NHLDataDownloader’ class. All you need to do is to specify the starting and ending season where it is indicated in our code. You only need to put the first year of a given season (e.g., put 2016 if you want the 2016-17 season, or 2020 if you want the 2020-21 season). It will download the data for all the regular season games and playoff games for the selected interval (inclusively). You also need to also specify the path of the folder where the downloaded data will be stored. 

Here is an example:

{% include image_full.html imageurl="/images/milestone1/download_step2.png" %}

### Step 3: Download the Data

Now, it’s time to download the NHL play-by-play data for the specified seasons. The code provided does this in two steps:
* First, it downloads the data for the regular season games using the ‘get_nhl_data_season’ method.
* Then, it downloads data for the playoff games using the ‘get_nhl_data_playoffs’ method.

The downloader will ensure that it skips games that didn’t take place, such as games that were not played due to a series ending early, or the regular season being shortened unexpectedly (like the 2019-20 season due to Covid). The code prints out a message when it skips the download of a file for a game or if it fails to download the file for a game and it prints out the ID of the game in question.

**Note**: Downloading the data may take a while so keep that in mind. However, we have coded these methods to allow you to download the data in several shots. If you cannot wait for the dataset to be downloaded completely all at once, you can stop the process at any time and continue later (with the same inputs as before) and it will continue downloading the data where it stopped!

{% include image_full.html imageurl="/images/milestone1/download_step3.png" %}

### Conclusion
You now have the guide to download the NHL play-by-play data using the provided Python code. Simply replace the starting and ending years as well as the data directory with your preferences and you are good to go! Good Luck!



## 2. Interactive Debugging Tool

### The guide on what the interactive debugging tool is for and how to use this tool.
### Question 1
The code is included in the interactive_debug.ipynb in the NHL_analytics repository. Here the interactive debugging tool allows the user to to flip through all of the events, for
every game of a given season, with the ability to switch between the regular
season and playoffs. WIth the inputs (game ID and event ID) entered by the user, the description of the game will be printed. The description compares the goals, shots on goals, and so on of the home team and away team. Then a rink figure with the description of the event is shown. THe blue point represents the location of the event. Lastly, a detailed description of the event is printed, including the time, location, players’ names, result, and so on.

As you can see in our example, after the user inputs (game ID and event ID) are provided, the game information between OTT and TOR is printed, and the location of the event is shown on the figure, the title of the figure tells us “Mark Stone blocked shot from Jake Gardiner at 01:10 period 1”. Then the event details are printed below. And the user could change the inputs to retrieve the information of any game and event we have.


{% include image_full.html imageurl="/images/milestone1/debug.png" %}

## 4. Tidy Data

### Question 1

{% include image_full.html imageurl="/images/milestone1/tidy.png" caption="head of tidied dataframe" %}


### Question 2

We would need a “Penalty” feature plus the name of the team that got penalized and the name of the team that got the advantage.
Then we go row by row to label shot events of the team that has the advantage to ‘power play’ or ‘5 on 4’, until we run into 1 of the 3 possibilities:
1. “Goal” by the team that has the advantage – minor penalty would end (not major)
2. Penalty time is done by using the “Period Time” column, so we only label shot events within the penalty time.
3. A new penalty event occurs:
  * If the penalty is for the same team, then it would be ‘5 on 3’ for strength, until the first penalty is up or until any of the other possibilities happen.
  *  If the penalty is for a different team, then strength would be ‘4 on 4’ (aka “even strength”) until the first penalty is up or until any of the other possibilities happen.



### Question 3

1. For rebound shots: A rebound is any attempt made within 3 seconds of another blocked, missed or saved shot attempt without a stoppage in play in between.

   We could check right before a specific goal/shots, whether there is another shot event (within 3 seconds, hence ‘rebound’). Plus, there should be no other events (for example ‘stoppage’) in between those two shot events. Therefore, we could look for these based on period time and classify by yes or no whether there is a shot from the same team right before the current event.
2. For shots off the rush:

   It could be characterized by changing of possession right before the shots event

3. The third feature could be “Assist’

   Since we would try to quantify the MVP of a hockey season, the number of assists is a very useful feature for the model.

## 5. Simple Visualizations

### Question 1: Figure Analysis: Shot Types and Their Impact

In our analysis of the different shot types and their impact on the game, we examined the data from a selected NHL season (in our case, the 2016-17 season) to gain insights into which shot is the most dangerous and which one is the most common. We plotted a stacked bar graph that overlays the number of goals over the number of shots for each shot type.

The wrist shot appears to be the most dangerous shot as it is the shot that contributed to the greatest number of goals in the chosen season. It is also the most common shot by far, which is quite understandable given that it is arguably the shot with the most accuracy, quick release, and goal-scoring ability, so obviously players tend to use it way more than the other types of shots.

We chose to plot a stacked bar graph because we believe it is the best way to visualize which shot is the most common as well as most dangerous. It provides a clear and visually compelling way to compare these shot types. By looking at the graph, we can know right away the number of goals and shots taken for each shot type and we can compare it with the other shot types quickly since they are all represented in the same graph. By doing such, we can identify patterns and trends in player behavior which enables us to draw conclusions on how effective and common a shot is.

{% include image_full.html imageurl="/images/milestone1/shot_types_2016-17.png" %}

### Question 2: Figure Analysis: Shot Distance vs Goal Probability

In our analysis of the distance from which a shot is taken and the probability it is a goal, we visually represented that data for the 2018-19 to 2020-21 NHL seasons. We created one line plot for each season to understand the relation between the distance of a shot and its probability of being a goal. We have also plotted one graph with all the line plots of each season to facilitate an easier comparison between these three seasons.

We have noticed that the shots taken from a distance of 0 feet to about 40 feet from the net had the highest probability of scoring. We see that the further the shot is taken, the smaller the chance is for a goal to be scored. There is however a small but noticeable spike for the bigger distances (75 feet and more) which can be explained. These shots from very far distances (often from the side of the rink players defend) are usually taken when the opposing team remove their goaltender and leave their net empty to have an extra player in offense in the hopes of scoring a goal to bring the score to a draw.

{% include image_full.html imageurl="/images/milestone1/shot_goal_ratio_vs_distance_2018-19.png" %}
{% include image_full.html imageurl="/images/milestone1/shot_goal_ratio_vs_distance_2019-20.png" %}
{% include image_full.html imageurl="/images/milestone1/shot_goal_ratio_vs_distance_2020-21.png" %}


But what is noteworthy about these line plots is their striking similarity across the three seasons (it can be very well seen in the last graph). This led us to the following key takeaway from these line plots, which is the remarkable consistency in the relationship between the shot distance and the goal-scoring probability. Despite all the external factors, varying strategies over the seasons, player rosters changing very often, the data suggests that the shot distance is still a very important factor in the prediction of a shot being converted to a goal.

{% include image_full.html imageurl="/images/milestone1/shot_goal_ratio_vs_distance_combined.png" %}


We chose this type of figure because these plots allow us to analyze the probability of a goal as shots are taken from different distances. By examining multiple seasons, we can identify trends and assess whether there have been significant changes in goal-scoring patterns according to the distance over the past three seasons which enabled us to draw the conclusion we presented.




### Question 3: Figure Analysis: Percentage of Goals per Shots by Distance Category and Shot Type

By looking at our heatmap, we can say that the deflected shot might be one of the most dangerous shots as it has one of the highest percentage of goals per shot for short distances (between 0-10 ft). It has a nearly 70% success rate for deflections that happened between 0 and 5 ft from the net. This can be explained by the following: a deflection happens when a player changes the direction of a shot (whether it is intentional or not) as the puck is being shot at the net at a very high speed. The goaltender sees the shot and expects it to go in some direction and then at the last minute it changes due to a deflection and if it is close enough it doesn’t give him the chance to readjust.

It is worth noting that for all the shot types, the closer they are to the net, the higher their percentage of being converted to a goal is.


{% include image_full.html imageurl="/images/milestone1/heatmap_goal_percentage_2016-17.png" %}





## 6. Advanced Visualizations: Shot Maps



### Question 1

Here are the interactive shot maps for the NHL teams for 5 different seasons. At the beginning, all the teams' shot maps are displayed together. Once you select a team, only their shot map will be displayed.


[2016-2017 Season NHL shot maps]({% link shotMaps/2016-2017_season_nhl_shot_map.html %}){:target="_blank"}

[2017-2018 Season NHL shot maps]({% link shotMaps/2017-2018_season_nhl_shot_map.html %}){:target="_blank"}

[2018-2019 Season NHL shot maps]({% link shotMaps/2018-2019_season_nhl_shot_map.html %}){:target="_blank"}

[2019-2020 Season NHL shot maps]({% link shotMaps/2019-2020_season_nhl_shot_map.html %}){:target="_blank"}

[2020-2021 Season NHL shot maps]({% link shotMaps/2020-2021_season_nhl_shot_map.html %}){:target="_blank"}


### Question 2

With these plots, we can get an idea of how effective a team’s offense is. The plots show us for each location of the offensive zone how much a team shoots compared to the average of the league. If it is red, they shoot above the average (the darker the higher) and if it is blue they shoot below the average (the darker the lower). 

From looking at these plots, we can determine if a team’s offense can dominate the opponent when setting up their attack and how dangerous their attack is. This can be seen if the plot for a team shows more red in the center of the offensive zone and near the goal crease. This also allows us to see if a team’s offense lacks quality shots and drive for the net. This will often manifest into having a plot for these teams where they either mostly have blue areas in most parts of the zone, or they have blue ares in the dangerous parts of the zone (where the scoring opportunities are the best).




### Question 3

We can see in the shot map of the Colorado Avalanche from the 2016-2017 season that in most parts of the offensive zone, they shoot way less than the average NHL teams do, especially near the goal crease, where they are way below the average. 

They exceed the league’s average only in 3 areas of the offensive zone:
- Near the blue line (aka offside line) on the left corner and in the middle
- On the top right part of the right-hand faceoff circle of the offensive zone

All 3 of these areas either have not much of a good angle for a shot, or are pretty far from the net. We have shown in our analysis in the previous section that the further the shots are from the goal, the smaller their chances of scoring is. We can say that the team overall didn’t shoot enough towards the net which led to less scoring chances therefore less goals so in the end less wins. This can explain why they finished last that season (21 points behind the second worst team).


{% include image_full.html imageurl="/images/milestone1/colorado_avalanche_sm_2016-17.png" %}


In the 2020-2021 season, the Colorado Avalanche have been shooting much more than all the other teams. They had a much higher shot average than the other teams in almost every section of the offensive zone which would also mean they shot more often than the others. Shooting more often and closer creates more quality scoring opportunities, therefore more goals. They finished 1st overall that season.

We can conclude that the amount of shots and the location of the shots are a big contributing factor that will inevitably help you or not in scoring more goals and winning more games as we can see with the case of the Colorado Avalanche. This does indeed make sense. As Wayne Gretzky, the greatest hockey player on the planet, once said: “You miss 100% of the shots you don’t take”. To score more goals, one must shoot more often.


{% include image_full.html imageurl="/images/milestone1/colorado_avalanche_sm_2020-21.png" %}



### Question 4

We can see for the Tampa Bay Lightning that in all 3 seasons, they have been dominating the league in terms of shots taken in the middle of the offensive zone. This shows they’re capable of controlling the offensive play and setting up their attack properly in order to shoot from the most dangerous areas of the ice. This obviously contributed to their success in recent years. 

{% include image_full.html imageurl="/images/milestone1/tampabay_lightning_sm_2018-19.png" %}
{% include image_full.html imageurl="/images/milestone1/tampabay_lightning_sm_2019-20.png" %}
{% include image_full.html imageurl="/images/milestone1/tampabay_lightning_sm_2020-21.png" %}


On the other hand, we can see that the Buffalo Sabres have been struggling to shoot in the same mentioned areas, falling way behind the league average. They have been however taking more shots on the extremities of the offensive zone. This shows that the opposing team’s defense was able to push off their attack and not allow them to get in the slot and have better scoring opportunities (which resulted in them taking more shots on the outside which has less angle, more distance and more traffic therefore less chance of being a goal).



{% include image_full.html imageurl="/images/milestone1/buffalo_sabres_sm_2018-19.png" %}
{% include image_full.html imageurl="/images/milestone1/buffalo_sabres_sm_2019-20.png" %}
{% include image_full.html imageurl="/images/milestone1/buffalo_sabres_sm_2020-21.png" %}


These graphs obviously don’t paint a complete picture because many other aspects need to be considered when trying to explain a team’s success or struggle, one of them being the amount of shots allowed compared to the league’s average (indeed, scoring many goals doesn’t mean much if on the other end they concede goals as easily). However, this still gives us a good picture of how well the offense performs.


Authors:
  
Name:              Kaiqi Cheng (20272810)
email:             kaiqi.cheng@umontreal.ca

Name:              Milosh Devic (20158232)
email:             milosh.devic@umontreal.ca

Name:              Yonglin Zhu (20257137)
email:             yonglin.zhu@umontreal.ca

[Github](git)


created using [Jekyll](https://jekyllrb.com/)

Just the [Docs][jekyll-docs] or Better [GitHub][jekyll-gh]

[jekyll-docs]: https://jekyllrb.com/docs/home
[jekyll-gh]:   https://github.com/jekyll/jekyll
[2016-2017]: "/_includes/2016-2017_season_nhl_shot_map.html"
[2017-2018]: "/_includes/2017-2018_season_nhl_shot_map.html"
[2018-2019]: "/_includes/2018-2019_season_nhl_shot_map.html"
[2019-2020]: "/_includes/2019-2020_season_nhl_shot_map.html"
[2020-2021]: "/_includes/2020-2021_season_nhl_shot_map.html"
[git]: https://github.com/miloshdevic/NHL_analytics.git
