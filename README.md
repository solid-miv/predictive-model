## Recommendation System

This is a recommendation system that predicts the amount of money needed to cover the insurance cost. It is aimed at the American citizens.


### How to run a project
Start ***main.py*** in order to run a project.


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
- ***images*** with a few statistical plots generated from the data.
- ***saved_model*** with a *model_train.py* file, which loads and preprocesses data, builds and trains a DNN, and saves it to *dnn_model.tf*.
- ***main.py*** that you can run to start the project.