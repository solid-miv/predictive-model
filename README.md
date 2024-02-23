## Recommendation System

This is a recommendation system that predicts the amount of money needed to cover the insurance cost. It is aimed at the American citizens.


### Data description
The recommendations system takes into account:
- Age
- Sex
- BMI (Body Mass Index)
- Number of children
- Whether the person smokes
- Region pf the person


### Project structure
You can run the project by starting **main.py** file in the project main directory. The project also contains folders:
- **data** with a __medical_cost.csv__ file containing the data itself and __preliminary_work.ipynb__ that shows testing various ML algorithms and building a DNN for regression task.
- **images** with a few statistical plots.
- **saved_model** with a __model_train.py__ that loads and preprocesses data, builds and trains DNN, saves the trained DNN model into __dnn_model.tf__ file
- **main.py** which you can run to start the project