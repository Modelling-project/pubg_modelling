import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings 
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression as linear
from sklearn.linear_model import Ridge as ridge
from sklearn.linear_model import Lasso as lasso
from lightgbm import LGBMRegressor as lgbm

#import pandas as pd
#from sklearn.tree import DecisionTreeRegressor
#from lightgbm import plot_importance
#from category_encoders import BinaryEncoder

def fit_model(df) :
    def constfig(pred, test, name):
        d = test.reset_index()
        d["pred"] = pred
        d = d.sort_values("winPlacePerc").reset_index(drop=True)
        
        fig = plt.figure(figsize = (10,5))
        ax1 = fig.add_subplot(111)
        
        sns.lineplot(x=d.index, y = d.winPlacePerc, data =d, label = "Original", ax=ax1, color='orange')
        sns.scatterplot(x=d.index, y=d.pred, label = name, ax=ax1)
        ax1.set_ylim([-2,2])
        plt.title(f"{name} Prediction", fontsize=12)
        plt.savefig(f"{name}_test_pred.png")
    
    features = df.drop(["Id","groupId", "matchType","matchId", "numGroups","damageDealt","winPlacePerc"], axis=1) 
    target = df["winPlacePerc"]
    
    train_X, test_X, train_y, test_y = train_test_split(features, target, test_size=0.2, random_state=589)
    
    for model_func in [linear, ridge, lasso, lgbm]:
      model_list = {linear:"LinearRegression", ridge:"RidgeRegression", lasso:"LassoRegression", lgbm:"LGBMRegression"}
      print(f"\n{model_list[model_func]} Fitting...")
      model = model_func().fit(train_X, train_y)
      pred_val_y = model.predict(test_X)
      pred_tr_y = model.predict(train_X)
      print("train MAE : ",np.round(mean_absolute_error(pred_tr_y, train_y),6))
      print("test MAE : ",np.round(mean_absolute_error(pred_val_y, test_y),6))
      constfig(pred_val_y, test_y, model_list[model_func])