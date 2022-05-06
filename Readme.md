![GitHub](https://img.shields.io/github/license/Thomas-George-T/Prediciting-Hits-on-Spotify?style=flat)
![GitHub top language](https://img.shields.io/github/languages/top/Thomas-George-T/Prediciting-Hits-on-Spotify?style=flat)
![GitHub last commit](https://img.shields.io/github/last-commit/Thomas-George-T/Prediciting-Hits-on-Spotify?style=flat)
![ViewCount](https://views.whatilearened.today/views/github/Thomas-George-T/Prediciting-Hits-on-Spotify.svg?cache=remove)

# Predicting Hits on Spotify

<p align="center">  
    <br>
	<a href="#">
        <img height=100 src="https://cdn.svgporn.com/logos/spotify-icon.svg?response-content-disposition=attachment%3Bfilename%3Dspotify-icon.svg" hspace=80> 
  </a>	
	<a href="#">
		<img src="https://raw.githubusercontent.com/Thomas-George-T/Thomas-George-T/master/assets/python.svg" alt="Python" title="Python" width ="120" />
	</a>
    <br>
</p>

## Table of Contents

- Problem Setting
- Problem Definition..........................................................................................................................
- Data Source
- Data Description
- Data Collection
- Data Processing
- Data Exploration
   - Data Statistics..............................................................................................................................
   - Data Distributions
- Model Exploration and Model Selection
- Implementation of Selected Models
   - 5 - Fold Cross-Validation..............................................................................................................
   - Random Forest Model...............................................................................................................
   - Feature Importance
   - Principal Component Analysis..................................................................................................
   - Hyperparameter Tuning
   - Random Search
   - Grid Search
- Performance Evaluation and Interpretation
   - Precision
   - Recall (Sensitivity)....................................................................................................................
   - ROC Curve................................................................................................................................
- Project Results...............................................................................................................................
- Impact of the Project Outcomes


## Problem Setting

The music industry is a multibillion-dollar industry consisting of artists of diverse backgrounds producing songs in various genres that influence people worldwide. Yet, there are only a handful of songs and artists that make it to the top of the billboards. By taking historical data for every decade from 1960 to 2019 on Spotify, we try to predict if the track will be a hit or not.

## Problem Definition

In this dataset created by fetching features from Spotify’s Web API, we try to classify a track  as a ‘Hit’ or not for every decade from 1960 to 2019. We try to use multiple classification  models and assess their performance to pick the optimal one.

## Data Source

The dataset for this project is acquired from Kaggle:

https://www.kaggle.com/theoverman/the-spotify-hit-predictor-dataset

## Data Description

This dataset consists of 41,106 songs with 19 attributes from 6 different decades from 1960  to 2019. The target variable is a label, indicating that a song is a hit (1) or flop (0). The track is considered a flop following these conditions:

1. The track must not appear in the 'hit' list of that decade.
2. The track's artist must not appear in the 'hit' list of that decade.
3. The track must belong to a genre that could be considered non-mainstream and/or Avant-
    grade.
4. The track’s genre must not have a song on the ‘hit’ list.
5. The track must have ‘US’ as one of its markets.

Other attributes of the dataset are tracks, artist, Uri, danceability, energy, key, loudness, mode, speechiness, acousticness, instrumentalness, liveness, valence, tempo, duration, etc.


```
Figure 1 : Data Sample with Descriptions
```
## Data Collection

We downloaded and extracted the 6 CSV files corresponding to the data of each of the decades and then merged them into a single dataframe.

## Data Processing

We determined the shape of the data to see how many records and columns are present. We  saw that all the **41,106** records and **19** columns have been successfully imported.  In the next step, we looked at the data types and checked if we needed to convert any of them.  We observed that we didn’t need to make any explicit data type conversion.  Next, we checked if there were any missing values or nulls present in our data. We could see  that there are no null values or any missing values in the dataset.  Lastly, we looked at the statistics of the data set. We gathered insights about the mean,  median, and range of the data of all our variables.  Further, we observed that amongst the predictor variables, two of the columns can be  considered categorical. These would be the ‘Mode’ and ‘Time Signature’ columns. These  wouldn’t necessarily need to be converted since they are already numerical.  Our Response variable is the target column which holds only binary values of ‘0’ and ‘1’  which indicates if a song is a Hit or not.


## Data Exploration

### Data Statistics

```
In this part, we looked at the statistics of the variables.
```

```
From the statistics, we can see that almost all the values for the predictor time_signature are
```
4. Hence, we decide to remove this predictor in this step.

### Data Distributions

```
In this part first, we create a boxplot to look for outliers and see the distribution of all the predictors.
```

```
Figure 2 : Distribution of Scaled Predictors with outliers
```
There are many outliers in the data in predictors such as instrumentalness and Spechiness. They have 8920, and 5088 outliers respectively, therefore we remove these two predictors completely and after that remove all the observations which have an outlier.

After removing the outliers, we have 33401 songs and 12 predictors for building the model.

```
Figure 3 : Distribution of the Scaled Predictors after removing the outliers
```
This boxplot illustrates the distribution of the predictors, however, as our goal is to build a classification model, exploring the distribution of the predictors by the target variable will be much more efficient.


_Figure 4 : Side by side boxplots by target variable_


From figure 4, the predictor’s danceability, energy, loudness, acousticness, duration, liveness, sections, and valence have a different distribution amongst hit songs and other songs. Hence, they can be considered good predictors.

The total number of Songs after removing the outliers is 33401. We count the number of hit songs and other songs to find out if we need oversampling.

```
Figure 5 : Count of hit songs and other songs
```
From figure 5 we can see that almost half of the songs are hit, and half of the songs are not considered hit, and we do not need oversampling.

## Model Exploration and Model Selection

The purpose of this project is to predict whether a song is classified as a hit song or is classified as a non-hit song. Hence, the goal of the project is classification, we use different classification models to get the best results. At first, 8 models are selected for this classification task. These models are:

1. Logistic Regression
2. K-Nearest Neighbors
3. Decision Tree
4. Support Vector Machine (Linear Kernel)
5. Support Vector Machine (RBF Kernel)
6. Neural Network
7. Random Forest
8. Gradient Boosting

Next, we split the dataset into 8 0% training and 2 0% testing datasets. We use 3-Fold cross validation to choose the best models. The Scores of the models are shown in table 1.

```
Model Score
```
```
Logistic Regression
68. 1 %
```
```
K-Nearest Neighbors
```
68. 4 %

```
Decision Tree
64.8%
```
```
Support Vector Machine
(Linear Kernel)
```
```
68. 2 %
```
```
Support Vector Machine
(RBF Kernel)
```
73. 4 %

```
Neural Network
73 .1%
```
```
Random Forest
73. 6 %
```
```
Gradient Boosting
73. 0 %
```
```
Table 1 : Model Scores
```
From table 1, the models are divided into two groups. Models that have a score lower than 69%, and Models with a score higher than 72%. Therefore, we choose 4 models that have a score of 72% and above for the next step. These models are Support Vector Machine (RBF Kernel), Neural Network, Random Forest, and Gradient Boosting.

## Implementation of Selected Models

### 5 - Fold Cross-Validation..............................................................................................................

In this step, we use cross-validation to observe the performance of the selected models. We use 5 - fold cross-validation, where the data is divided into 5 folds and 4 are used for training and the other one is used as the validation set. Each time 1 of the folds are used as the validation and therefore the model is fitted 5 times.

```
Model Score
Support Vector Machine
(RBF Kernel)
```
```
71.88%
```
```
Neural Network
71.73%
Random Forest
74.00%
Gradient Boosting
71.299%
```
```
Table 2 : 5 - Folds Cross-Validation Scores
```

From the results of the cross-validation, we choose the **Random Forest Classifier** as our final model.

### Random Forest Model...............................................................................................................

We first train the model on 80% training data (as we have 34000 rows, 80% training is enough).

The model will give an accuracy of 73.74%. In the next steps, we use different methods to enhance the performance of the model and finally evaluate the model.

### Feature Importance

The random forest model splits the nodes based on more important features and implicitly performs feature selection. In this step, we will try to remove some features with less importance and see if the model accuracy and running time are enhanced or not. Figure 6 shows the importance of each feature.

```
Figure 6 Feature Importance
```

It is observed that sections, mode, and key don’t have a lot of importance in the random forest model.

After removing these three features, the performance of the model is decreased by 0. 6 %, and the running time is only decreased by 1 second. Therefore, we only use this section to observe the importance of each feature and we will keep all the features.

### Principal Component  Analysis
In this step, we will use dimension reduction methods and try to improve the performance of the model. First, we build all the principal components to be able to find a subset that will capture more than 90% of the total variance in the data.

```
Figure 7 : PCA
```
From figure 7 , we can conclude that by using 7 principal components, 90% of the total variance is captured. Building the model with the 7 principal components, we get a Cross validation score of 6 8. 5 % for the model.

This is 6% less than the accuracy that we had without dimension reduction, and as a result, we will not use dimension reduction before building our random forest model.

### Hyperparameter Tuning

A random forest model has various parameters, and in this part, we try to optimize the model performance by choosing the best parameters for our random forest model. Random Forest parameters are:

_Number of Estimators:_ The number of estimators is equal to the number of trees in the model.
_Maximum Depth:_ Maximum depth is the maximum number of splits in the tree. (max level)
_Minimum Sample Split:_ Minimum sample split is the minimum number of samples required
to be able to split a node.
_Minimum Sample Leaf:_ Minimum sample leaf is the minimum number of observations that
each leaf should have after all the splits.
_Bootstrap:_ To use bootstrap or not.

As there are a few parameters and each random forest model takes 9 seconds to run, we first try to find a set of parameters and perform a random grid search on them, and after that perform a grid search on the best parameters.

### Random Search

The parameters for the random search are shown in the table:

```
Parameter values
N_estimators [100,110,120,...,990,1000]
Max_depth [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
Min_sample_split [2,4,6,8,10,...,46,48,50]
Min_sample_leaf [2,4,6,8,10,...,46,48,50]
Bootstrap True, False
Table 3 : Random Search Parameters
```
After running the model with the random parameters and with 3-fold cross-validation for each of them, we obtain the following result:


```
Figure 8 : Random Search Results
```
For each of the parameters, we choose 2 or 3 of the highest-scoring ones based on figure 8 and perform a grid search on them.

### Grid Search

In this part, we built models with 3-fold cross-validation for all the combinations of the following parameters:

```
Parameter values
N_estimators [400,900]
Max_depth [12,13]
Min_sample_split [2,12,28]
Min_sample_leaf [28,34]
Bootstrap False
Max_features Auto, sqrt
Table 4 : Grid Search Parameters
```

A model is built for all the combinations and with 3-fold cross-validation, which leads to 148 total models being fitted.

Out of all the models, a model with the parameters in table 6 has the best performance, and therefore we choose the following parameters for our random forest model:

```
Parameter value
N_estimators 400
Max_depth 13
Min_sample_split 12
Min_sample_leaf 28
Bootstrap False
Max_features sqrt
Table 5 : Final Parameters of the Random Forest model
```
After all the steps we can see that the final accuracy of the model is 73.0% which is 0.82% lower than the accuracy of the baseline model.

_This shows that performing PCA and Hyperparameter tuning does not necessarily improve the model's performance and here we will proceed with our baseline model._

## Performance Evaluation and Interpretation

For measuring the performance of the random forest classifier, we use accuracy score, confusion matrix, Precision, Recall, and f1_score.

The accuracy of the predictions by the random forest classifier is 73.82% on the test data. This means that the classifier predicted correctly on the test data by 73.82%. This in turn implies that the error rate is 2 6.18%.

The confusion matrix of the random forest classifier is as follows:

```
Predicted Hit Predicted Non-Hit
```
```
Actual Hit 2940 666
```
```
Actual Non-Hit 1083 1992
```
```
Table 6 : Confusion Matrix
```
The classification report for the test data and the predictions are:

```
Precision Recall f1-score Support
```

```
Non-Hit 0.75 0.64 0.69 3075
Hit 0.73 0.81 0.77 3606
accuracy 0.74 6681
macro avg 0.74 0.73 0.73 6681
weighted avg 0.74 0.74 0.73 6681
Table 7 : Classification Report
```
### Precision

It is the number of correctly identified members of a class divided by all the times the model predicted that class. In the case of “hits”, the precision score would be the number of correctly identified “hits” divided by the total number of times the classifier predicted “hits,” rightly or wrongly. This value is around 73%, meaning that out of all the songs which are predicted as a hit song by the model, 73% of them are hit songs.

### Recall (Sensitivity)....................................................................................................................

It is the number of members of a class that the classifier identified correctly divided by the total number of members in that class. For “hits”, this would be the number of actual “hits” that the classifier correctly identified as such. This value is 81%, meaning that out of all the hit songs, 81% are correctly classified as hit songs.

### ROC Curve................................................................................................................................

```
Figure 9 : ROC Curve
```

The ROC Curve plots the sensitivity based on 1-specificity for 100 different cutoffs. Here the area under the curve is 0. 81 which shows that the model is performing well. The random model has an area of 0. 5 and the best classifier has an area of 1.

## Project Results...............................................................................................................................

In the original dataset consisting of 40k records, there were multiple outliers in the initial 19 features. The features with the most outliers are removed. The remaining dataset is then split into a train test split of 80 – 20. The classification models such as Logistic Regression, K-Nearest Neighbors, Decision Tree, Support Vector Machine (Linear Kernel), Support Vector Machine (RBF Kernel), Neural Network, Random Forest, and Gradient Boosting are considered. The results show that the Random Forest model is the best classification model for our dataset. It was able to classify hit songs and non-hit songs with an accuracy of **73.8 9 %.**

After selecting Random Forest, Dimension reduction techniques such as principal component analysis and determining feature importance are performed in the hopes of improving model accuracy. Hyper-parameter tuning techniques including Random, and Grid Search are performed. It is observed that the use of these techniques resulted only in a slight improvement in the performance of the model with a large increase in the computation time. It is determined that these techniques can be ignored since there was no significant impact on the model’s performance.

The precision, recall, and the area under the ROC curve showed that the model is performing well. For hit songs, the Recall was 81% meaning that out of all the hit songs, the model predicted it correctly 81% of the time.

## Impact of the Project Outcomes

As the global music industry is moving more and more towards online audio streaming services such as Spotify, this project provides insights into what makes a song a hit on the billboard. Through this project, the importance of a variety of features like ‘loudness’ and ‘energy’ is observed in making a song hit or not. This will in turn help artists, producers, and even online streaming platforms make better business decisions, promotions, and what would be in the making of a popular song globally. The goal of the project was to accurately label a hit song as ‘1’ or ‘0’ otherwise. In this regard, the Random Forest classifier outperformed all other classifiers with its high recall, high precision, and high f1 scores for hit songs. From the analysis in this project, The Random Forest classification model could be used to predict hit songs. It accurately determines hit songs and works best for predictors that are highly diverse and varied, which would have been difficult to do so in the past.
