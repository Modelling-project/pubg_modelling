import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor
from lightgbm import LGBMRegressor
from lightgbm import plot_importance
from category_encoders import BinaryEncoder


def Modeling_LGBM(df) :
    features = df.drop(["Id","groupId", "matchType","matchId", "numGroups","damageDealt","winPlacePerc"], axis=1) #all
    target = df["winPlacePerc"]
    
    train_X, test_X, train_y, test_y = train_test_split(features, target, test_size=0.2, random_state=589)
    
    print("Fitting...")
    model = LGBMRegressor()
    model.fit(train_X, train_y)
    pred_y = model.predict(test_X)
    print("test MAE : ",np.round(mean_absolute_error(pred_y, test_y),6))
    pred_y = model.predict(train_X)
    print("train MAE : ",np.round(mean_absolute_error(pred_y, train_y),6))