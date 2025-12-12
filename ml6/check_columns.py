
import pandas as pd
import statsmodels.api as sm
import numpy as np

# Load data to simulate the environment
data = pd.read_csv("ml6/car-mpg.csv", header=None)
columns = [
    "cylinders",
    "displacement",
    "horsepower",
    "weight",
    "acceleration",
    "model_year",
    "origin",
    "mpg",
]
data.columns = columns
# Normalize check (the notebook says it is normalized, but let's check head)
# Actually the notebook does read_csv and then head.

model = sm.OLS(data.iloc[:, -1], data.iloc[:, :-1])
results = model.fit()
influence = results.get_influence()
sm_fr = influence.summary_frame()
print(sm_fr.columns)
