import numpy as np
import pandas as pd
from localfunc import reduce_ram_usage
import gc



def dropIdx(df, idx) :
    df.drop(index=idx, inplace=True)
    dropIdx.dpIdx_sum +=len(idx)
    return df
dropIdx.dpIdx_sum = 0

def dropOutlier (df):
    def dropIdx(df, idx) :
        df.drop(index=idx, inplace=True)
        dropIdx.dpIdx_sum +=len(idx)
        return df
    dropIdx.dpIdx_sum = 0

    print("Pre-Processing...")
    for i in df.columns.to_list() :
        df.drop(index=df[df[i].isnull()==True].index, inplace=True)
        
    print("Droping Outliers...")

    vip_features = ["assists","boosts","DBNOs","heals","kills","killStreaks","walkDistance", "revives", "roadKills", "vehicleDestroys"]

    group = df.groupby('groupId').count()
    df = dropIdx(df, df[df.groupId.isin(group[group["Id"]>group["Id"].quantile(0.9999)].index)==True].index) 

    for col in (vip_features + ["damageDealt","longestKill", "rideDistance", "swimDistance","weaponsAcquired", "matchDuration"]):
        df = dropIdx(df, df[df[col]>df[col].quantile(0.999)].index)
    
    for col in vip_features:
        df = dropIdx(df, df[df["walkDistance"]<df[col]].index)

    df = dropIdx(df, df[df.groupby('matchId')['kills'].transform('max')  > df.groupby('matchId')['Id'].transform('count')  ].index)
    df = dropIdx(df, df[(df['rideDistance']==0) & (df['roadKills']>0)  ].index)

    #edge case
    df.loc[(df.maxPlace>1)&(df.numGroups==1), "maxPlace"] = 1

    print(f"{dropIdx.dpIdx_sum} Columns has deleted!") 

    del vip_features, group      
    gc.collect()
    
    return df

def encodeMatch (df):
    print("Encoding matchType...")

    mapper = lambda x: 'normal' if ('normal' in x) or ('crash' in x)or ('flare' in x)else x 
    df["matchType"]=df["matchType"].apply(mapper)

    mapper = lambda x: 'solo' if ('solo' in x) else 'duo' if ('duo' in x) else 'normal' if ('normal' in x) else 'squad' 
    df["matchType"]=df["matchType"].apply(mapper)

    df = pd.concat([df,pd.get_dummies(df["matchType"])], axis=1)

    del mapper
    return df

def makeCols (df) :
    print("Making columns...")
    df["killPlace"] = df.groupby("matchId")["kills"].transform('rank', ascending=False)
    #data leakage 없는 killPlace data

    stat_feature = ["assists",
                    "boosts",
                    "DBNOs",
                    "heals",
                    "kills",
                    "killStreaks",
                    "walkDistance", 
                    "revives", 
                    "roadKills", 
                    "vehicleDestroys",
                    "damageDealt",
                    "longestKill", 
                    "rideDistance", 
                    "swimDistance",
                    "weaponsAcquired"]
    stat_list = ["max","mean","median","min"]
    for col in stat_feature :
        for stat in stat_list:
            df[f"{col}_{stat}"] = df.groupby("groupId")[col].transform(stat)
            df[f"{col}_{stat}Place"] = df.groupby("matchId")[f"{col}_{stat}"].transform('rank', ascending=False)
    #group별 column stats, match별 group stats 순위

    print(len(stat_feature)*len(stat_list)+1, f"columns Made! Now {len(df.columns)} column in DF.")
    df = reduce_ram_usage(df)
    return df
