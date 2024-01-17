import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.ensemble import GradientBoostingRegressor,VotingRegressor
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, make_scorer,r2_score, mean_absolute_error
from sklearn.model_selection import cross_val_score, train_test_split, KFold
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from scipy.stats import norm, skew


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

df_train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
df_test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')

df_train.head()

df_train.describe()

(mu, sigma) = norm.fit(df_train['SalePrice'])
sns.displot(df_train['SalePrice'], kde = True, stat="density", height=6, aspect=2)
plt.xlabel("House's sale Price in $", fontsize = 10)
plt.ylabel("Density", fontsize = 10)
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)])
plt.show()

# Logarithm of SalePrice
df_train.loc[:, 'SalePrice'] = np.log1p(df_train.SalePrice)

(mu, sigma) = norm.fit(df_train['SalePrice'])
sns.displot(df_train['SalePrice'], kde = True, stat="density", height=6, aspect=2)
plt.xlabel("House's sale Price in $", fontsize = 10)
plt.ylabel("Density", fontsize = 10)
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)])
plt.show()

# Correlation Heatmap
corr = df_train.drop('Id', axis=1).select_dtypes('number').corr()
plt.figure(figsize=(15, 15))
sns.heatmap(corr , fmt = '0.1f', cmap = 'YlGnBu', annot=True, cbar=False)
plt.tight_layout()
plt.show()

df_train.info()

df_alldata = pd.concat([df_train, df_test], axis=0, ignore_index=True, sort=False)
df_alldata.index += 1

test_id = df_test.Id
df_alldata = df_alldata.drop(['Utilities', 'Id'], axis=1)

df_alldata.PoolQC = df_alldata.PoolQC.fillna("None")
df_alldata.MiscFeature = df_alldata.MiscFeature.fillna("None")
df_alldata.Alley = df_alldata.Alley.fillna("None")
df_alldata.Fence = df_alldata.Fence.fillna("None")
df_alldata.MSSubClass = df_alldata.MSSubClass.fillna("None")
df_alldata.Functional = df_alldata.Functional.fillna("Typ")
df_alldata.MasVnrType = df_alldata.MasVnrType.fillna("None")
df_alldata.FireplaceQu = df_alldata.FireplaceQu.fillna('None')

df_alldata.GarageYrBlt = df_alldata.GarageYrBlt.fillna(df_alldata.YearBuilt)

for feature in df_alldata[df_alldata.isna()].columns[:-1]:
    if df_alldata[feature].dtype=='object':
        df_alldata[feature] = df_alldata[feature].fillna(df_alldata[feature].mode()[0])
    else:
        df_alldata[feature] = df_alldata[feature].fillna(0)

df_alldata.info()

df_alldata['HasWoodDeck'] = (df_alldata['WoodDeckSF'] == 0) * 1
df_alldata['HasOpenPorch'] = (df_alldata['OpenPorchSF'] == 0) * 1
df_alldata['HasEnclosedPorch'] = (df_alldata['EnclosedPorch'] == 0) * 1
df_alldata['Has3SsnPorch'] = (df_alldata['3SsnPorch'] == 0) * 1
df_alldata['HasScreenPorch'] = (df_alldata['ScreenPorch'] == 0) * 1
df_alldata['YearsSinceRemodel'] = df_alldata['YrSold'].astype(int) - df_alldata['YearRemodAdd'].astype(int)
df_alldata['Total_Home_Quality'] = df_alldata['OverallQual'] + df_alldata['OverallCond']
df_alldata['TotalSF'] = df_alldata['TotalBsmtSF'] + df_alldata['1stFlrSF'] + df_alldata['2ndFlrSF']
df_alldata['YrBltAndRemod'] = df_alldata['YearBuilt'] + df_alldata['YearRemodAdd']
df_alldata['Total_sqr_footage'] = (df_alldata['BsmtFinSF1'] + df_alldata['BsmtFinSF2'] +
                                   df_alldata['1stFlrSF'] + df_alldata['2ndFlrSF'])
df_alldata['Total_Bathrooms'] = (df_alldata['FullBath'] + (0.5 * df_alldata['HalfBath']) +
                                 df_alldata['BsmtFullBath'] + (0.5 * df_alldata['BsmtHalfBath']))
df_alldata['Total_porch_sf'] = (df_alldata['OpenPorchSF'] + df_alldata['3SsnPorch'] +
                                df_alldata['EnclosedPorch'] + df_alldata['ScreenPorch'] +
                                df_alldata['WoodDeckSF'])
df_alldata['TotalBsmtSF'] = df_alldata['TotalBsmtSF'].apply(lambda x: np.exp(6) if x <= 0.0 else x)
df_alldata['2ndFlrSF'] = df_alldata['2ndFlrSF'].apply(lambda x: np.exp(6.5) if x <= 0.0 else x)
df_alldata['GarageArea'] = df_alldata['GarageArea'].apply(lambda x: np.exp(6) if x <= 0.0 else x)
df_alldata['GarageCars'] = df_alldata['GarageCars'].apply(lambda x: 0 if x <= 0.0 else x)
df_alldata['LotFrontage'] = df_alldata['LotFrontage'].apply(lambda x: np.exp(4.2) if x <= 0.0 else x)
df_alldata['MasVnrArea'] = df_alldata['MasVnrArea'].apply(lambda x: np.exp(4) if x <= 0.0 else x)
df_alldata['BsmtFinSF1'] = df_alldata['BsmtFinSF1'].apply(lambda x: np.exp(6.5) if x <= 0.0 else x)
df_alldata['haspool'] = df_alldata['PoolArea'].apply(lambda x: 1 if x > 0 else 0)
df_alldata['has2ndfloor'] = df_alldata['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)
df_alldata['hasgarage'] = df_alldata['GarageArea'].apply(lambda x: 1 if x > 0 else 0)
df_alldata['hasbsmt'] = df_alldata['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)
df_alldata['hasfireplace'] = df_alldata['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)

df_alldata = pd.get_dummies(df_alldata, dtype=int).reset_index(drop=True)

train = df_alldata[~df_alldata["SalePrice"].isnull()]
test = df_alldata[df_alldata["SalePrice"].isnull()]

# Split data
X = train.drop(['SalePrice'],axis=1)
y = train.SalePrice
X_test = test.drop(['SalePrice'], axis=1)

kf = KFold(n_splits=5, shuffle=True, random_state=42)

def rmse(model, X=X):
    rmse = np.sqrt(-cross_val_score(model, X, y, scoring="neg_mean_squared_error", cv=kf))
    return (rmse)

cat_params = {'objective': 'RMSE',
              'logging_level': 'Silent',
              'random_seed': 42,
              'iterations': 1418,
              'learning_rate': 0.02652868035829293,
              'depth': 5,
              'subsample': 0.8461673472269735,
              'colsample_bylevel': 0.2090812096076299,
              'min_data_in_leaf': 18,
              'bagging_temperature': 0.38443667709265117,
              'leaf_estimation_iterations': 26,
              'reg_lambda': 88.65102640088449}
cat = CatBoostRegressor(**cat_params)

lgb_params = {'objective': 'root_mean_squared_error',
              'metric': 'rmse', 'max_depth': 4,
              'num_leaves': 573,
              'min_child_samples': 16,
              'learning_rate': 0.020226081533263125,
              'n_estimators': 3575,
              'min_child_weight': 12,
              'subsample': 0.5000525014552387,
              'colsample_bytree': 0.3117329991936672,
              'reg_alpha': 0.33069864870290777,
              'reg_lambda': 0.6276331560546634,
              'random_state': 42,
              'extra_trees': True}
lgb = LGBMRegressor(**lgb_params)

xgb_params = {'booster': 'gbtree',
              'max_depth': 4,
              'max_leaves': 448,
              'learning_rate': 0.019589865645907364,
              'n_estimators': 1148,
              'min_child_weight': 16,
              'subsample': 0.6045002721574211,
              'reg_alpha': 0.18506040958773265,
              'reg_lambda': 0.820379255120091,
              'colsample_bylevel': 0.3949602640893716,
              'colsample_bytree': 0.5949407650079529,
              'colsample_bynode': 0.44329210952485487,
              'random_state': 42,
              'objective': 'reg:squarederror',
              'n_jobs': -1}
xgb = XGBRegressor(**xgb_params)

gbr_params = {'max_depth': 5,
              'learning_rate': 0.012785410864644397,
              'n_estimators': 3649,
              'subsample': 0.4561581884406722,
              'min_samples_leaf': 14,
              'min_samples_split': 10,
              'random_state': 42,
              'loss': 'huber',
              'max_features': 'sqrt'}
gbr = GradientBoostingRegressor(**gbr_params)

base_models = [
    ('LGBM', lgb),
    ('GradientBoosting', gbr),
    ('XGBoost' , xgb),
    ('CatBoost', cat),
]

voting_model = VotingRegressor(estimators=base_models)
voting_model.fit(X, y)

# test_id.loc[:, 'SalePrice'] = np.expm1(voting_model.predict(X_test))
# submission = df_test[['Id', 'SalePrice']]
submission = pd.DataFrame({
    'Id': test_id,
    'SalePrice': np.expm1(voting_model.predict(X_test))
})
submission.to_csv("submission.csv", index=False)