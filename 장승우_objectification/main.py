import Preprocessing
import Engineering
import Modeling
import pandas as pd

def main() :
    print("Data loading...")
    train = pd.read_csv("./data/train_V2.csv")
    test = pd.read_csv("./data/test_V2.csv")
    print("Data loaded!\n")
    
    Preprocessing.checkNaN(train)
    train = Preprocessing.dropNaN(train)
    
    train = Engineering.dropOutlier(train)
    train = Engineering.encodeMatch(train)
    train = Engineering.makeCols(train)
    
    train = Modeling.fit_model(train)

if __name__=="__main__" :
    main()

