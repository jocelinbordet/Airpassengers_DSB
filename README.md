# Description of the Airplanes prediction challenge

This project was carried out during the "Python for Data Science" course of the X-HEC Data Science for Business Master of Science.
I worked with [Jocelin Bordet](https://github.com/jocelinbordet) on the project.

The problem consisted in predicting the number of passengers between two American cities
(i.e. on domestic flights) at certain dates between 2011 and 2013. Hence this problem was a
regression problem. For privacy matters, this number of passengers was actually replaced
by another target (presumably obtained with a simple strictly increasing transformation of the
original data).

The objective was to build two files : estimator.py containing a scikit-learn pipeline which final step was a 
predictive model, and external_data.csv, which contained additional data that we had to find ourselves through 
various sources. They were then sent on the [RAMP Platform](https://ramp.studio/) which we used for the project, which is maintained by the INRIA team.

Considering the fact that the data initially provided was very limited, and after a few attempts
to produce predictions with the initial data as only input, we understood how crucial would be
the acquisition of external data to increase our RMSE and the overall performance of our
models. Hence data acquisition was the first stage of the problem. Since the flights
presented in the problems were between important American cities, and given that the dates
were only half a dozen years ago, data acquisition was likely to be easy enough, for an
almost-certain high increase in performance.

The second stage of the problem was to find an appropriate Machine Learning model to
solve this problem, and even more importantly, to tune the hyperparameters in order to
increase its performance. Since the interpretability of the model was actually not a problem
in our case, and since the volume of data was not important enough to raise significant
issues of computation time, we could use almost any algorithm/machine learning library we
saw fit.

Our best model (ICM_IAB_XIV) reached a RMSE of 0.284 on Ramp.

## Data acquisition

The data acquired basically lied among non-temporal and temporal data. Along the way, we
built several functions to increase our efficiency to preprocess data and automate the
integration of data from new sources.

### Acquisition of non-temporal data

When we brainstormed to find all the factors correlated to the use of flights that came to our
minds, we quickly thought of :

**(1) Distance between the cities** 
 We acquired this data by downloading large and
non-specific data on biggest American cities, that included their latitudes and
longitudes, and we implemented a function that computes the distance (in km)
between the two cities, using the Haversine law.
This data clearly had a positive impact on the performance of our model, because it
is highly discriminating (if the distance is too close, people could be tempted to use
less expansive means of transportation, while if it is too far, people could be tempted
not to pay an expansive plane ticket).

**(2) Population and density of the cities** 
Generic data that was apparently useful and
not much correlated to other data.
This gives a basis for the potential volume of passengers transiting in each airport,
even though this data is approximative in the sense that some of the airports are
known to be stop-over airports.

**(3) “Baseline” passengers per line**
This is the average number of passengers on the
whole dataset per line (e.g. Atlanta to Orlando).
It gives a baseline frequentation for the airlines. This feature has proven to be a
useful one according to our feature importance analysis (see below).


### Acquisition of temporal data

The other relevant element that came to our minds was that the predictions we had to make
were actually effective and expired, in the sense that they were not forecasts but predictions
of events that had actually happened in the last years. Therefore, we could use actual
events of the years 2010-2013 as predictors, and other data related (more or less) to the
number of passengers boarding a plane in the US in the 2010s.

**(1) Volume of Google search related to flight between two US cities**
For this, we used the Python module Selenium in order to automate the search (and download) onto
the Google trends website of the weekly volume of internet search for each of the
couples “flight [Departure city] to [Arrival city]”.
This was even more useful considering that in the initial data we were provided with
the average number of weeks that separated the plane tickets sales and the
departure, hence we could “lag” the data by a translation factor corresponding to the
average of the “WeeksToDeparture” feature. Plus, this data offered a very useful
level of granularity (data significantly variates from one week to another).

**(2) Calendar data**
 We integrated data on the American federal holidays, as well as the
weekday of each flight.
We thought it was useful (and it has proven to be) since whether this is a holiday, a
working day or a week-end day not only affects the overall volume of passengers, but
also the repartition between the flight lines (basically between “business cities” on
weekdays 1 to 5, and between “tourism/travelling cities” on weekdays 6 and 7).

**(3) Stock data**
 We used Booking, Delta and AAirlines. We did not choose Southwest
Airlines, even though it proposes domestic flights because their stock increased a lot
between 2011 and 2013 for almost strictly financial reasons non directly related to
day-to-day volume of flights.
The idea was to find data that was comprehensive and included all sorts of events
that positively or negatively impacted the use of planes in the US (plane crashes,
storms and weather events, oil price). In our sense, this data was even more relevant
that it evolved on a daily basis. Yet, we identified at least three flaws: this data is not
actualized on the week-end, is also affected by other events that are not related to
the volume of people, and is not specific to certain flights.

**(4) Monthly volume of people having transited through each airport**
Data obtained on the website of the American Bureau of Transportation.
This data would have been every more useful if it was more “granular” (e.g. weekly or
daily data), yet it was already beneficial since it made a distinction between
“domestic” and “international” passengers.

**Unexplored fields**
Data related to major sports or cultural events; explicit data (day-to-day) of storms, plane
crashes, and other events negatively impacting the use of planes; volume of internet search
for rental cars in arrival cities; specific weather/temperature/rain predictions; oil price.

## Model

### Setting a performance criterion

We had to define a criterion. The most common and interpretable one is actually the RMSE.
Plus, it was simplified as the ramp-test command on the terminal provided a simple way to
produce a quick performance analysis of our model.
Nonetheless, we used several other methods to increase the performance of our model,
especially to tune the hyperparameters of our models (see below).

### Choosing a model: CatBoost

We have tried several machine learning models and compared them using the ramp-test
command of the terminal (which is basically a cross-validation).
- **Decision trees** : These algorithms did not perform well (compared to the following)
on the train dataset, and terribly on the test dataset (problems of overfitting, structural
in Decision Trees models).
- **LASSO** : More powerful than a linear regression since we add a L1 penalty supposed
to trade a portion of bias to lower to variance. This did not perform poorly but was not
sufficient to compare with the following. This model was nonetheless very economic
in terms of computations.
- **Random Forests** : More powerful than the previous models, and performs better on
the test dataset (lower the risk of overfitting compared to Decision Trees).
- **Boosted Random Forests (xgboost library)** : This was one of the best models we
found in terms of RMSE, even with a poor initial choice of hyperparameters, and
performed particularly well on the test dataset compared to the other methods. Yet, in
order to make it usable in our mixed-data case (both categorical and numerical), we
had to encode the categorical features. Plus, this model was very sensitive to the
hyperparameters set, and the computations to find optimal ones were costly.
- **CatBoost** : This model is also a boosted random forest, however it handles the categorical features correctly
. Our RMSE increased significantly when we switched to this
model, and we managed to optimize it correctly.

### Choosing the features

We mostly used the feature importance attribute returned by the CatBoost model in order to
choose the right features : the most significant data was composed of the baseline frequentation for each flight, of the
date of departure and of the 3 stock index variables. We tried to remove population
departure and holiday, however as we lost in RMSE we figured out that those two variables
were still significant.

Here is an output showing the most significant features : 

![Features Importance](https://github.com/louistransfer/air-passengers-dsb/blob/master/docs/images/features_importance.png?raw=true)

### Tuning the hyperparameters

In order to test the hyperparameters, we decided to start with random parameters and to use
a GridSearch to tune them. We evaluated the following parameters for the xgboost model.
However, we ran out of time to gridsearch the CatBoost model. Here are the effects of those
parameters :
- **Learning rate and max depth** : they control the propensity of the model to overfit. The
learning rate is a penalisation used to control the train/test error ratio, while the max
depth can be used to stop a tree from making too many decisions on weak
predictors;
- **N estimators and colsample by tree** : they control the speed of convergence of the
algorithm.

### Overall optimization

Due to the heavy computation volume of the GridSearch, we decided to try to leverage a
gaming laptop’s GPU in order to accelerate the computations. A Lenovo Y520 equipped with
a GTX 1060 was used : we installed Nvidia’s CUDA drivers, and we set the parameter
tree_method = “gpu_hist” in the XGBoost Regressor object. We also set the n_jobs=2
parameter within the GridSearch in order to parallelize the computations as much as
possible, on a machine which had quite a weak CPU compared to the GPU (a i5-7300U Intel
Core).
The results were very satisfying with relatively quick results (900 tasks completed in 2
hours), which enabled us to brute force a lot of parameters which we had identified as key.

# RAMP detailled informations

More informations about RAMP can be obtained on the page of the Airplanes project template [here](https://github.com/ramp-kits/air_passengers).

