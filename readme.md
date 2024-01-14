# Predicting the patient's risk of stroke

This repository contains the exploratory data analysis notebook, the model's development notebook, custom functions file, and the folder for all the files that are needed for local fastapi deployment.

## Summary

In this project I am using three different boosting models to predict the probability of the patient having a stroke. The models used are: XGBoost, LightGBM and CatBoost. The dataset used for the classification is the [Stroke Prediction Dataset](https://www.kaggle.com/fedesoriano/stroke-prediction-dataset) from Kaggle. This dataset contains 11 features that can be easily obtained from a patient with no need for complicated medical tests to be performed. The goal of this project is to try to build a model that can predict the probability of a patient having a stroke with the highest accuracy possible. This could be a useful tool for determining if the patient has a high risk of having a stroke and needs further medical testing to prevent that from happening.

## Deploy Locally  with FastAPI

To deploy the model locally you need to have [Docker](https://www.docker.com/) and [FastAPI](https://fastapi.tiangolo.com) installed on your machine. Then you need to clone this repository and run the following commands in the terminal:

```bash
cd stroke_prediction/fastapi
docker build -t stroke_prediction-app .
docker run -p 80:80 stroke_prediction-app
```