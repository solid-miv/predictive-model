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
You can run the project by starting __main.py__ file in the project main directory. The project also contains folders:
- __data__ with a _medical_cost.csv_ file containing the data itself and _preliminary_work.ipynb_ that shows testing various ML algorithms and building a DNN for regression task.
- __images__ with a few statistical plots.
- __saved_model__ with a _model_train.py_ that loads and preprocesses data, builds and trains DNN, saves the trained DNN model into _dnn_model.tf_ file
- __main.py__ which you can run to start the project