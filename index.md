# A Machine Learning Project Predicting the Survival of Passengers on the Titanic
 
<p align="center">
 <img width="600" src="titanic_header.png">
</p>

[Image Credit](https://www.skyatnightmagazine.com/space-science/titanic-sinking-aurora-history/): Jon Powell

# Introduction 

Using the [Titanic - Machine Learning from Disaster](https://www.kaggle.com/c/titanic/data) dataset from [Kaggle](https://www.kaggle.com/), I have built scripts that uses a variety of machine learning algorithms to predict the survival of passengers from the Titanic tragedy.
Here, I will go through the process I used to clean the data and use the Gradient Boosting Classifier to obtain a prediction survival rate of 85%.
While several machine learning (ML) classifiers were deployed (Random Forest, Multi-layer Perceptron, and Support Vector Machine) in this small project the Gradient Boosting Classifier appeared to provide the best results.
These result may be revised in the future, but the point of here is to show a non-chemistry example of machine learning work on my resume.

# The Titanic Passenger Dataset 

Below are the features that were provide from the Kaggle dataset. 
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

However, not all passengers have all of the information. 
For instance, 177 passangers are missing age values, roughly 80% of the passengers are missing there cabin information, and 2 passengers are missing what port they embarked from.
Additionally some of this information is difficult to convert into meaningful numeric values such as passenger names and ticket numbers.
So even though a rather clean dataset was provide work still need to be done to furth clean and extract additional infromation from the dataset at hand. 

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
