import EDA
import dPP
import Modeling
import pandas as pd

def main() :
    print("Data loading...")
    train = pd.read_csv("./data/train_V2.csv")
    test = pd.read_csv("./data/test_V2.csv")
    print("Data loaded!")
    
    train = dPP.dropOutlier(train)
    train = dPP.encodeMatch(train)
    train = dPP.makeCols(train)
    
    train = Modeling.Modeling_LGBM(train)

if __name__=="__main__" :
    main()
