---
layout: post
title:  "NHL Data Analysis Milestone 2"
description: "some Introduction."
date:  2023-11-14
feature_image: # maybe?
tags: [Download data from API, data cleaning]
---
## Introduction

## 2. Feature Engineering I

### 2.1 Question 1

{% include image_full.html imageurl="/images/milestone2/hist_shot_count_by_distance.png" caption="Histogram of the Shot Count by Distance" %}

We observe that there are much more shots taken from a closer distance to the net than from a further distance. In fact, nearly 80% of the shots are taken inside the offensive zone (less than 65 feet from the net) and that’s where almost all of the goals are scored. We can barely see the goals count at greater distances (although present), and these goals usually happen when there’s an empty net.

{% include image_full.html imageurl="/images/milestone2/hist_shot_count_by_angle.png" caption="Histogram of the Shot Count by Shooting Angle" %}

We decided to keep negative values for the angles because it shows from which side the shot was taken because we believe this could be useful if we wanted to predict the probability of a player scoring a goal depending on his location (especially considering if he is lefty or righty). To be clear, the angle is determined by the relative position to the goal line, so an angle of 90 degrees means the player was facing the net. We can observe that the greater the shooting angle, the more there are shots taken and goals scored. In fact, most of the goals were scored at an angle between 75 and 90 (and -75 to -90) degrees from the goal line. This is normal because that’s where the goaltender can cover the net the least, in comparison to a shot that doesn’t have a good angle. We can also observe that there’s an unusually high goal rate for very small angles (between 0 and 10). This could be explained by the puck being almost in the net and a player just tapping it in (could be confirmed if the distance for these shots are taken in account) or simply luck however the amount is too high for such deduction.

{% include image_full.html imageurl="/images/milestone2/2D_hist_distance_vs_angle.png" caption="2D Histogram of the Distance vs the Shooting Angle" %}

This graph allows us to see the shot distance and the shooting angle in one same graph instead of 2 separate ones. We can observe the same things we stated previously: the greater the distance, the less shots are taken and the less goals are scored; and the greater the shooting angle, the more the shots are taken and goals are scored. The “anomaly” with the small shooting angle is still visible.


### 2.2 Question 2

{% include image_full.html imageurl="/images/milestone2/goal_rate_vs_distance.png" caption="Goal Rate vs Distance" %}

The goal rate increases as the distance is smaller with the highest goal rate (between 70-80%) being when the shot is within 10 feet of the net. We can also observe an increase of the goal rate starting from a distance of around 90 feet and more (so from the other side of the rink). As mentioned previously, this can be explained by the fact that teams pull out their goaltender at the end to attempt to tie the game if they’re losing. Usually, players will not shoot from their defensive zone but in these situations, they will attempt from a big distance and often succeed because no one is tending the net, hence the increase in the goal rate.

{% include image_full.html imageurl="/images/milestone2/goal_rate_vs_angle.png" caption="Goal Rate vs Shooting Angle" %}

The goal rate increases as the shooting angle gets better (bigger). The highest goal rate can be seen for shots taken at an angle between 75 and 90 (and -75 to -90) degrees from the goal line. This is normal because that’s where the goaltender can cover the net the least, in comparison to a shot that doesn’t have a good angle. As mentioned before, we can observe that there’s an unusual spike for the goal rate for very small angles (between 0 and 10).

### 2.3 Question 3

{% include image_full.html imageurl="/images/milestone2/hist_goal_distance_empty_nets.png" caption="Histogram of Goals by Distance (Empty Net vs Non-Empty Net)" %}

We can see that most of the goals scored are in fact non-empty net goals. Once again, most of the goals are scored within the offensive zone (less than 65 feet from the net). We can also see that most of the empty net goals are actually scored within 100 feet of the net. Players will attempt shots at an empty net at greater distances since there isn’t a goaltender.


## 3. Baseline Models

### 3.1 Question 1

By using a basic logistic regression model, we observed an accuracy of 0.91. However, upon deeper data exploration, we identified a significant class imbalance issue within the dataset. The number of data instances with an "isGoal" value of 0 outnumbers those with an "isGoal" value of 1 by approximately a factor of nine to ten. This imbalance raises concerns about the potential for model overfitting, where the model might tend to predict all instances as 0 to achieve a high accuracy. Therefore, even though the accuracy appears high, the model's performance is not good.

To address this imbalance, we should consider data resampling techniques or utilize stratified k-fold cross-validation to rebalance the dataset. Furthermore, it's essential to note that accuracy alone is an insufficient metric for evaluating model performance. Alternative evaluation metrics, such as precision, recall, the confusion matrix, ROC curve, AUC (Area Under the Curve), and others, should be taken into account to get a more comprehensive understanding of the model's performance.

For logistic regression models as baseline models, we tried just the basic logistic regression classifier to fit the data. Also we applied the basic data preprocessing like removing duplicates and nan values from the training, validation sets. Here we used evaluation metrics like training set accuracy, validation set accuracy, f1 score, Receiver Operating Characteristic (ROC) curves and the AUC metric of the ROC Curve, the goal rate as a function of the shot probability, the cumulative proportion of goals, and model percentile, and the calibration curve to compare with other models’ performances.

<!-- {% include image_full.html imageurl="/images/milestone2/.png" caption="Receiver Operating Characteristic (ROC) curves and the AUC metric of the ROC Curve" %} -->

<!-- {% include image_full.html imageurl="/images/milestone2/.png" caption="The goal rate as a function of the shot probability" %} -->

<!-- {% include image_full.html imageurl="/images/milestone2/.png" caption="the cumulative proportion of goals, and model percentile" %} -->

<!-- {% include image_full.html imageurl="/images/milestone2/.png" caption="the calibration curve" %} -->

[Comet link](https://www.comet.com/nhl-analytics-milestone-2#projects)

## 4. Feature Engineering II

For this part, we have the following features (some extracted, some created):

   - ShotType: categorical feature of the different shot types (wrist shot, slap shot, tip-in, etc)
   - Period: indicates which period of the game it is, numerical value from 1 to 7 ( 4,5,6,7 are OTs)
   - GameTime: total number of seconds elapsed in the game
   - XCoord: coordinate of the event on the x axis
   - YCoord: coordinate of the event on the y axis
   - isEmptyNet: 0 if the net is not empty, 1 if it is empty
   - isGoal: 0 if the event is not a goal, 1 if it is
   - DistanceToGoal: distance (in feet) between the shot and the net, Nan if it isn’t a shot
   - ShootingAngle: angle to the net of the shot, Nan if it isn’t a shot
   - LastEvent: categorical feature of the different types of event preceding the current one (shot, hit, faceoff, etc)
   - LastEvent_XCoord: coordinate of the last event on the x axis
   - LastEvent_YCoord: coordinate of the last event on the y axis
   - TimeLastEvent: seconds passed since the last event
   - DistanceLastEvent: distance between the current event and the last event
   - Rebound: True if the last event was also a shot, otherwise False
   - AngleChange: angle to the net of the shot if the shot is a rebound, otherwise 0
   - Speed: defined as the distance from the previous event, divided by the time since the previous event


Here's an image to illustrate better the "Speed" and "AngleChange" features:

{% include image_full.html imageurl="/images/milestone2/explain_angle_speed.png" caption="" %}


[Link](https://www.comet.com/nhl-analytics-milestone-2/feature-engineering-data/519fe224df9c448d9a35a4586141fd96?experiment-tab=assetStorage) to the experiment which stores the filtered DataFrame artifact

## 5. Advanced Models

### 5.1 Question 1

The training data has been split into training and validation data in a 1 to 4 ratio. A seed number for the split has been set at the beginning to keep consistency across the XGBoost model. We are not adding any extra hyperparameters at this time (no ratio adjustment for imbalance data etc.). 

The "basic" XGBoost model is doing just a bit better than logistic regression (71% accuracy compare to about 68%). (We did not overlayed the logistic regression curve since it's in a different file.)

{% include image_full.html imageurl="/images/milestone2/xgboost1_1.png" caption="1st xgboost model's Receiver Operating Characteristic (ROC) curves and the AUC metric of the ROC Curve" %}

{% include image_full.html imageurl="/images/milestone2/xgboost1_2.png" caption="1st xgboost model's Goal rate vs probability percentile Curve" %}

{% include image_full.html imageurl="/images/milestone2/xgboost1_3.png" caption="1st xgboost model's Cumulative proportion of goals vs probability percentile" %}

{% include image_full.html imageurl="/images/milestone2/xgboost1_4.png" caption="1st xgboost model's Reliability Curve" %}

Comet link [here](https://www.comet.com/nhl-analytics-milestone-2/model-registry/xgboost_2/1.0.0?tab=assets)

### 5.2 Question 2

First, Defining a dictionary for Hyperparameter tuning and using grid search to do cross-validations. From Grid search CV, we could select the 'best' Hyperparameters and used that for later models
For Hyperparameter choices, there are many choices of hyperparameter for tuning, but to save computing time and due to the limit for computing power, we selected 3: "gamma", "max_depth", and "n_estimators".
1. We selected "max_depth" because it's a tree specific hyperparameter. We would like to have deeper tree to capture more complex patter, but also we want to avoid overfitting, so we think this is an important hyperparameter.
2. "gamma" is to control complexity and verify regularization of our model.
3. "n_estimator" controls the number of trees in the mode, we chose this Hyperparameter to limit overfitting
We have also added theweight variable ("scale_pos_weight") to adjust for imbalanced dataset
After tuning, our model accuracy improved dramatically. (71% to 99%)

{% include image_full.html imageurl="/images/milestone2/xgboost2_1.png" caption="2nd xgboost model's Receiver Operating Characteristic (ROC) curves and the AUC metric of the ROC Curve" %}

{% include image_full.html imageurl="/images/milestone2/xgboost2_2.png" caption="2nd xgboost model's Goal rate vs probability percentile Curve" %}

{% include image_full.html imageurl="/images/milestone2/xgboost2_3.png" caption="2nd xgboost model's Cumulative proportion of goals vs probability percentile" %}

{% include image_full.html imageurl="/images/milestone2/xgboost2_4.png" caption="2nd xgboost model's Reliability Curve" %}

Comet link [here](https://www.comet.com/nhl-analytics-milestone-2/model-registry/xgboost_2/1.1.0?tab=assets)

### 5.3 Question 3

Since our model has a very high accuracy already, we decide to limit the number of features we used to predict. We carried out this process by using the feature importance function from XGBoost, then validate it using SHAP.
We have picked out the top 6 features and still achieving quite high accuracy but substantially lowering the computing time (gridsearchCV computing time from 10min to less than 5 mins).

{% include image_full.html imageurl="/images/milestone2/xgboost3_1.png" caption="3rd xgboost model's Receiver Operating Characteristic (ROC) curves and the AUC metric of the ROC Curve" %}

{% include image_full.html imageurl="/images/milestone2/xgboost3_2.png" caption="3rd xgboost model's Goal rate vs probability percentile Curve" %}

{% include image_full.html imageurl="/images/milestone2/xgboost3_3.png" caption="3rd xgboost model's Cumulative proportion of goals vs probability percentile" %}

{% include image_full.html imageurl="/images/milestone2/xgboost3_4.png" caption="3rd xgboost model's Reliability Curve" %}

{% include image_full.html imageurl="/images/milestone2/SHAP.png" caption="SHAP for feature selection" %}

Comet link [here](https://www.comet.com/nhl-analytics-milestone-2/model-registry/xgboost_2/1.2.0?tab=assets)


## 6. Give it your best shot!

### Feature selection

For feature selection, we have carried out randome forest classifier and correlation matrix:
1. Random Forests are commonly used for feature selection due to their ability to provide feature importances. Here are a few reasons why Random Forests are suitable for this purpose:
   - Randome forest classifier provides feature importance score for easy computable and interpretable references for feature selection.
   - Also since the accuracy of our model is very high, we are afraid of overfitting, so we choose this one
   - RFC could also captures feature interaction and correlations between features

2. For correlation matrix, we did this one mostly to have a general and global view of interactions between features, which substantiate our choices from RFC

{% include image_full.html imageurl="/images/milestone2/correlation_matrix.png" caption="Part of correlation matrix" %}

### Decision trees


### Neural Networks

We wanted to try a different type of model so we implemented a neural network. This was done using the keras library.

We shuffled the data in order to not have any order dependency issue. We also split the dataset into a training set, validation set and test set to follow better the performance of the model. Since we had an unbalanced dataset, we used the SMOTE method to create synthetic instances to have the same number of instances classified as goal and not goal.

In terms of preprocessing, we encoded the categorical variables, dropped, the rows with Nans, removed the GameID column, and scaled the columns that made sense to scale (so not the ‘Rebound’ column for instance because it has only 2 values possible, 0 and 1).

#### I. First attempt

At first, we created the neural network with 1 hidden layer(relu activation function) containing 32 neurons and trained it on all the features we had from feature engineering 2. It was trained on 100 epochs with the “adam” optimizer and with early stopping. We obtained an accuracy of around 98.5% on the validation set and the test set. The area under the ROC curve was 0.978. However, we can see that on the validation set, the accuracy is all over the place. We have very bad predictions and very good ones too. We figured we would need to optimize the model more. Also, we were worried about overfitting the model. Here are some graphs to illustrate the first attempt:

{% include image_full.html imageurl="/images/milestone2/model_accuracy_100it.png" caption="" %}

{% include image_full.html imageurl="/images/milestone2/model_loss_100it.png" caption="" %}

{% include image_full.html imageurl="/images/milestone2/ROC_curve_100it.png" caption="" %}

{% include image_full.html imageurl="/images/milestone2/reliability_curve_100it.png" caption="" %}

{% include image_full.html imageurl="/images/milestone2/more_metrics_100it.png" caption="More Metrics to Visualize" %}



#### II. Hyperparameter optimization

After reading some articles, we have come to the conclusion that a SGD optimizer would be more suited for our task. We now had to determine which value to give to the learning rate. We also wanted to test out different numbers of neurons for our layer (we started with only one hidden layer so that the model can be less complex, if it works it’s better) and we also tested with 2 hidden layers.

   1. Learning rate

      We trained and tested the model with these values for the learning rate: 0.00001; 0.0001; 0.001; 0.01; 0.1; 0.2; 0.3. 

      We got satisfying results with these values: 0.00001; 0.0001; 0.001; 0.01. However, the other ones gave horrible results, the accuracy was around 50% on the validation and test set. The best result was achieved with a learning rate of 0.0001 (accuracy of around 98%).

   2. Number of Neurons for 1 hidden layer

      We trained and tested the model with these values for the number of neurons: 4; 8; 16; 32. They all had similar results however, having 16 neurons in the hidden layer was slightly better than the others.

   3. 2nd hidden layer

      We tried the same thing for the second layer and there weren’t any significant differences. We decided to keep only one hidden layer so that our model is less complex and has less chance of overfitting.



#### III. Feature Selection

We have done several feature selection methods and trained our model on the selected ones. 

1. Random Forest Classifier

   With this method, we kept the following features (the weights were shown in the previously when we discussed about the feature selection methods we did):
      - TimeLastEvent
      - Speed
      - DistanceLastEvent
      - LastEvent_XCoord
      - LastEvent_YCoord
      - LastEvent
      - DistanceToGoal
      - XCoord
      - GameTime
      - Period


   Here were our results: accuracy of 97.78% on the test set (obtained from the training set as mentioned above).

   {% include image_full.html imageurl="/images/milestone2/model_accuracy_rfc_ft.png" caption="" %}
   {% include image_full.html imageurl="/images/milestone2/model_loss_rfc_ft.png" caption="" %}
   {% include image_full.html imageurl="/images/milestone2/ROC_curve_rfc_ft.png" caption="" %}
   {% include image_full.html imageurl="/images/milestone2/goal_rate_percentile_nn_rfc.png" caption="" %}
   {% include image_full.html imageurl="/images/milestone2/cumulative_proportions_nn_rfc.png" caption="" %}
   {% include image_full.html imageurl="/images/milestone2/reliability_curve_nn_rfc.png" caption="" %}

2. Correlation Matrix

   With this method, we included only the following features (the matrix was shown in the previously when we discussed about the feature selection methods we did):
      - Speed
      - DistanceLastEvent
      - LastEvent
      - DistanceToGoal
      - isEmptyNet
      - ShotType
      - Rebound
      - AngleChange

   Here were the results: accuracy of 65.03% on the test set (obtained from the training set as mentioned above).

   {% include image_full.html imageurl="/images/milestone2/model_accuracy_corr_ft.png" caption="" %}
   {% include image_full.html imageurl="/images/milestone2/model_loss_corr_ft.png" caption="" %}
   {% include image_full.html imageurl="/images/milestone2/ROC_curve_corr_ft.png" caption="" %}
   {% include image_full.html imageurl="/images/milestone2/reliability_curve_corr_ft.png" caption="" %}



#### IV. Best Neural Network model

We have come to the conclusion that the neural network trained with the optimized hyperparameters and the features selected by the random forest classifier had the best results overall. Indeed, it area under the ROC curve is nearly 1, the accuracy is high on new data and it had the best reliability curve of all.





### CONCLUSION:

For this part, we have attempted many different things to get the best model possible. We tried different types of models, different feature selection methods, shuffled the data and tuned the hyperparameters of our models. We have come to the conclusion that the best model was the best neural network model. When comparing all the different metrics from model to model, we can see that the best neural network had the best reliability curve even though the accuracy and ROC curve was slightly less good than the decision trees. Furthermore, it is the model we spent the most time tuning the hyperparamters and optimizing it as much as possible by trying different structures, optimizers, etc, therefore we felt the most confident with this model.


## 7. Evaluate on test set

### Question 1: 


### Question 2: 


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
