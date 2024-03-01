## Recommendation System

This is a recommendation system that predicts the amount of money needed to cover the insurance cost. It is aimed at the American citizens.
The original data was taken from [here](https://www.kaggle.com/datasets/mirichoi0218/insurance/data).


### How to run a project
Start ***main.py*** in order to run a project.
You don't need to train any models before running the program, because they have already be trained and could be found inside *saved_model_dnn/* and *saved_model_forest/*. If you do want to train the models by yourself or change some hyperparameters, then feel free to run *model_train_dnn.py* or *model_train_forest.py*.


### Data description
The recommendations system takes into account:
- Age
- Sex
- BMI (Body Mass Index)
- Number of children
- Whether the person smokes
- Region pf the person


### Project structure
The project main directory contains the following folders:
- ***data*** with a *medical_cost.csv* file, which contains the raw data, and *preliminary_work.ipynb*, which illustrates the comparison of different ML models and a DNN model.
- ***icon*** contains various icons used in the app.
- ***images*** with a few statistical plots generated from the data.
- ***saved_model_dnn*** with a *model_train_dnn.py* file, which loads and preprocesses data, builds and trains a DNN, and saves it to *dnn_model.tf*. *dnn_evaluation.txt* contains the DNN model's scores.
- ***saved_model_forest*** with a *model_train_forest.py* file, which loads and preprocesses data, builds and trains a Random Forest Regressor, and saves it to *forest_model.joblib*. *forest_evaluation.txt* contains the RFR model's scores.
- ***screens*** contains the screenshots of the running app
- ***main.py*** that you can run to start the project.
- ***requirements.txt*** contains external python modules used in the project and their versions.


### Application preview
#### Successful prediction
![Screenshot of a successful prediction](/screens/successful_prediction.png)
#### Prediction with a warning
![Prediction with a warning](/screens/prediction_with_warning.png)
#### Running app
![Running app](/screens/running_app.gif)
