## Recommendation System

This is a recommendation system that predicts the amount of money needed to cover the insurance cost. It is aimed at the American citizens.
The original data was taken from [here](https://www.kaggle.com/datasets/mirichoi0218/insurance/data).


### How to run a project
1. Navigate to the project folder.
2. Set up a virtual environment. See `requirements.txt` for the versions.
3. Activate a virtual environment.
4. Run `python main.py` command.

**N.B.** You don't need to train any models before running the program, because they have already been trained and could be found inside `saved_model_dnn/`, `saved_model_wdnn/`, and `saved_model_forest/`. If you do want to train the models by yourself or change some hyperparameters, then feel free to modify and run `model_train_dnn.py`, `model_train_wdnn.py` or `model_train_forest.py`, respectively.


### Models available
This recommendation system can predict the costs with 3 models:
- Deep Neural Network (you can find its architecture in `saved_models/saved_model_dnn/model_train_dnn.py`)
- Wide&Deep Neural Network (you can find its architecture in `saved_models/saved_model_wdnn/model_train_wdnn.py`)
- Random Forest Regression (you can find its setup in `saved_models/saved_model_forest/model_train_forest.py`)
Deep NN and Wide&Deep NN have approximately the same scores. And they are slightly better than the Random Forest Regression model.


### Data description
The recommendations system takes into account:
- Age
- Sex
- BMI (Body Mass Index)
- Number of children
- Whether the person smokes
- Region of the person


### Project structure
The project main directory contains the following folders:
- `data` with a *medical_cost.csv* file, which contains the raw data, and *preliminary_work.ipynb*, which illustrates the comparison of different ML models and a DNN model.
- `saved_models` contains `saved_model_forest`, `saved_model_dnn`, `saved_model_wdnn`. Each of these 3 folders contains the saved model (Random Forest Regressor or Deep Neural Network or Wide&Deep Neural Network), a Python file, which you can run to train the model again (though it has already been trained), and a `.txt` file with model's scores (mean squared error, mean absolute error, and the coefficient of determination $R^2$)
- `supplementary` contains folders with various images used in the app and in this README file.
- `main.py` that you can run to start the project.
- `requirements.txt` contains external python modules used in the project and their versions.


### Application preview
#### Successful prediction
![Screenshot of a successful prediction](/supplementary/screens/successful_prediction.png)
#### Prediction with a warning
![Prediction with a warning](/supplementary/screens/prediction_with_warning.png)