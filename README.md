# pubg_modelling
## Feature Engineering
- Id, groupId, matchId 컬럼들은 modelling을 하는데 방해가 되는 feature들 이므로 feature enginnering 과정에서 사용하고 drop한다
- matchType은 ordinal encoding을 통해 수치화를 한다
- killPlace는 data leakage 문제로 drop을 하거나 변경을 한다
- kills와 damageDealt의 상관계수를 보면 0.89로 너무 높은 상관관계를 보여주기 때문에 damageDealt를 drop한다
- killPoints, rankPoints, winPoints 컬럼들은 winPlacePerc과의 관계에서 낮은 상관관계를 보여줌으로써 drop한다

## Model
- XGBoost와 lightGBM은 거의 흡사한 결과를 보여주지만 lightGBM의 속도가 더욱 빠르므로 LightGBM을 선택
