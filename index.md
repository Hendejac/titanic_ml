# A Machine Learning Project Predicting the Survival of Passengers on the Titanic
 
<p align="center">
 <img width="600" src="titanic_header.png">
</p>

[Image Credit](https://www.skyatnightmagazine.com/space-science/titanic-sinking-aurora-history/): Jon Powell

# Introduction 

Using the [Titanic - Machine Learning from Disaster](https://www.kaggle.com/c/titanic/data) dataset from [Kaggle](https://www.kaggle.com/), I have built scripts that use a variety of machine learning algorithms from [sklearn](https://scikit-learn.org/stable/) in Python to predict the survival of passengers from the Titanic tragedy.
Here, I will go through the process I used to clean the data. 
Then demonstrate how I used the Gradient Boosting Classifier (GBC) to obtain **a prediction success rate of 87%**.
While several machine learning (ML) classifiers were deployed (Random Forest, Multi-layer Perceptron, and Support Vector Machines) in this small project the Gradient Boosting Classifier appeared to provide the best results.
These results may be revisited in the future to improve the prediction ability.

# Included Jupyter Notebooks 

**titanic_ml.ipynb**:
*The initial workflow and testing notebook for the different classifiers*

**titanic_ml_age_filling_regressor_and_data_remaker.ipynb**:
*This is the notebook that was used to clean and make the final dataset used.*

**titanic_ml_clf_final**:
*This is the notebook where the final data was used and the Gradient Boosting Classifier was optimized.*

# The Titanic Passenger Dataset 

Below are the features that were provided from the Kaggle dataset. 
In total the training set had 891 passengers to train and test the different ML models.


| Variable | Definition   | Key |
| --------:|-------------:|----:|
| survival |  Survival    | 0=Died, Lived=1 |
|  pclass  | Ticket Class |    |
|   name   | Passenger Name and Title | |
|   sex    |     Sex      |    |
|  sibsp   | # of siblings / Spouses |  |
|  parch   | # of parents / Children |  |
|  ticket  | Ticket Number  |  |
|   fare   | Passenger Fare |  |
|  cabin   | Cabin Number   |  |
| embarked | Port of Embarkation | C = Cherbourg, Q = Queenstown, S = Southampton |

However, some passengers have missing information. 
For instance, 177 passengers are missing age values, roughly 80% of the passengers are missing cabin information, and 2 passengers are missing what port they embarked from.
Additionally, some of this information is difficult to convert into meaningful numeric values such as passenger names and ticket numbers.
So even though a rather clean dataset was provided, work still needs to be done to further clean and extract additional infromation from the dataset at hand. 

|    | Column      |  Non-Null Count | Dtype  | 
|---:| -----------:|----------------:|-------:|
| 0  | PassengerId | 891 non-null    | int64  |
| 1  | Survived    | 891 non-null    | int64  |
| 2  | Pclass      | 891 non-null    | int64  |
| 3  | Name        | 891 non-null    | object |
| 4  | Sex         | 891 non-null    | object |
| 5  | Age         | 714 non-null    | float64|
| 6  | SibSp       | 891 non-null    | int64  |
| 7  | Parch       | 891 non-null    | int64  |
| 8  | Ticket      | 891 non-null    | object |
| 9  | Fare        | 891 non-null    | float64|
| 10 | Cabin       | 204 non-null    | object |
| 11 | Embarked    | 889 non-null    | object |

# Cleaning the Data and Filling in Missing Values

### Missing Cabin Data

This data is a little tricky to work with and might need to be revisted, **since only ~%20 of the passengers have this data the column was simply dropped**. 
Some passengers have multiple cabins and this data could possibly by extended to other passengers, such as the passengers siblings/parents, but proper assignment of the rooms appears challenging and there would still be missing information. 
Additionally, this information doesn't appear to be connected to the Ticket Number, so information from the Ticket Number doesn't appear to help fill in missing cabin numbers.

### Missing Port of Embarkation Data

Only 2 passengers are missing this information and so those rows (passengers) were dropped. 

### Missing Age Data

In total 177 of the passengers were missing their ages values. 
Doing a quick glance at the correlation between age and survival shows essentially no correlation, but I remembered in the Titanic movie there was an attempt to load the women and children onto the lifeboats first. 
So this left me with a few options, and the opportunity to play with the data. 

#### Options

1.) Simply remove all passengers with missing ages. 

2.) Fill in all missing ages with the mean age of the passengers. 

3.) Use a ML regression to predict and fill in the missing ages. 

##### Option 1

This is not ideal as removing the 177 passengers with missing ages removes a lot of data.
This resulted in all classifiers performing poorly compared to the other two options.

##### Option 2

The mean age of the passengers was ~30 +/- 15 years. 
This is not a great estimate of the age, but atleast it doesn't scew the age one way or another, and is straight forward to implement with the pandas dataframe fillna function. 
This resulted in pretty good **prediction success rate of 85%**.

##### Option 3

Using the Gradient Boosting Regressor (GBR) I tried to predict the missing ages. 
The GBR was able to predict the age with a mean average error (MAE) of 8.2 years while simply using the mean age provides a MAE of 12.6 years.
This only marginally improved the results giving a **prediction success rate of 85-86%**.
So for simplicity option two is likely the best.

### Adding Title Data from the "Name" Column 

I was unsure at first how to use the name column, but realized that the title (Mr., Mrs. Miss., Master. Rev., ect...) of each person says a little something about them.
In total there were really about 15 titles, some unique titles like "Mlle./Mme. (meaning Mademoiselle)" were converted to Miss., because they were rare and meant the same thing.
This could be done for some of the other rare titles, and might be a place of improvement for further iterations of the process.
Each of these titles were converted into One-hot Encoding (OHE), and used in the prediction. 
Surprisingly, this information actually hindered the success rate and was not used directly used in the final GBC model.
This information, however, was helpful was in the processes of predicting and filling in the missing ages. 
Using the OHE of the titles for the GBR prediction of the ages improved the prediction by reducing the MAE of 9.9 years to 8.2 years.
The ages predicted by the GBR using the OHE title data was used in the final GBC model.

# The Final Data 

The final data set can be seen below. 
Here, all the categorical data have been converted to numerical values and all missing data have been added or those rows/columns have been removed.

|   |PassengerId |	Survived |	Pclass |	Sex |	Age	      | SibSp |	Parch |	Fare    |	Embarked |
|--:|-----------:|---------:|-------:|----:|----------:|------:|------:|--------:|---------:|
|0	 | 1	         | 0	       |  3	    | 1.0 |	22.000000	|  1	   | 0     |	7.2500  |	  2.0    |
|1	 | 2	         | 1	       |  1	    | 0.0 |	38.000000	|  1	   | 0	    | 71.2833 |	  0.0    |
|2	 | 3	         | 1	       |  3	    | 0.0	| 26.000000	|  0	   | 0	    | 7.9250  |   2.0	   |
|...|   ...      |  ...     | ...    | ... | ...       | ...   |  ...  | ...     |  ...     |
|887|	890	       | 1	       | 1	     | 1.0	| 26.000000	|  0	   | 0	    | 30.0000	|   0.0    |
|888|	891	       | 0	       | 3	     | 1.0	| 32.000000	|  0	   | 0	    | 7.7500	 |   1.0    |


# Testing Different Classifiers and Hyperparameter Optimization

In total I tested four different classifiers: the Random Forest classifier, a Neural Network Classifier (Multi-layer Perceptron classifier), a Support Vector Machine classifier, and the Gradient Boosting Classifier. 
I really only manually played around with these classifiers and found that the GBC provided the best results (80-83% sucess rate, hyperparameters not optimized) compared to the others (79-81% success rate, hyperparameters not optimized). 
Overall, the different classifiers still performed pretty well with very little tweaking of the hyperparameters, but the GBC appeared to work just a bit better. 
Surely, more time spent with the other classifier would have likely boosted their performance.

Having chosen the GBC I wrote a short function to test a variety of different hyperparameters to determine the best inputs to use. 
Once these hyperparameters were elucidated and with the data cleaned up, the resulting **Titantic passenger survival prediction success rate was 87%**. 





