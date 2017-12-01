#load data
#clean data - handle missing data, outliers and categorical variables
#test assumptions

#libraries
#pandas - data structures and data analysis tools
import pandas as pd
#matplotlib - 2d plotting, publication quality
import matplotlib.pyplot as plt
#seaborn - based on matplotlib, draw attractive statistical graphs
import seaborn as sb
#numpy - fundamental package for scientific computing
import numpy as np
#scipy.stats - statistical functions
from scipy.stats import skew
from scipy import stats
#sklearn.preprocessing - standardize/normalize data
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, LassoCV, LassoLarsCV
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_regression
from sklearn.feature_selection import RFE
from sklearn import svm

def rmse_cv(model,x,y):
    rmse = np.sqrt(-cross_val_score(model,x,y,scoring="neg_mean_squared_error",cv = 5))
    return(rmse)

#import dataset into dataframe using pandas
df = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')
all_data = pd.concat((df.loc[:,'MSSubClass':'SaleCondition'],
                      df_test.loc[:,'MSSubClass':'SaleCondition']))
#EDA - Exploratory data analysis
#first we will use a scatterplot matrix to see the pair-wise correlations
#between the features
sb.set(style='whitegrid',context='notebook')
k = 10
cols = df.corr().nlargest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(df[cols].values.T)
sb.set(font_scale=1.25)
hm = sb.heatmap(cm, cbar=True, annot=True, square=True,
        fmt='.2f',annot_kws={'size':10}, yticklabels=cols.values,xticklabels=cols.values)
#plt.show()
plt.clf()
#from the plot we can see that OverallQual, GrLivArea, GarageCars, GarageArea
#TotalBsmtSF and 1stFlrSF all have above a 0.6 correlation with SalePrice
#However, GarageCars and GarageArea are highly correlated with each other
#(0.88) so it would be good to drop GarageArea.
#It would also appear that TotalBsmtSF and 1stFlrSF are strongly correlated with
#eachother (0.82) so we shall drop 1stFlrSF

#removing columns with null data
#df = df.dropna(axis=1,how='any')
#print(df.isnull().sum().max())

##saleprice_scaled = StandardScaler().fit_transform(df['SalePrice'][:,np.newaxis])
##low_range = saleprice_scaled[saleprice_scaled[:,0].argsort()][:10]
##high_range = saleprice_scaled[saleprice_scaled[:,0].argsort()][-10:]
##print('outer range(low) of distribution:')
##print(low_range)
##print('outer range(high) of distribution:')
##print(high_range)

#log sale price
df['SalePrice'] = np.log1p(df['SalePrice'])

numeric_features = all_data.dtypes[all_data.dtypes != "object"].index

skewed_features = all_data[numeric_features].apply(lambda x: skew(x.dropna()))
skewed_features = skewed_features[skewed_features > 0.75]
skewed_features = skewed_features.index
all_data[skewed_features] = np.log1p(all_data[skewed_features])

all_data = pd.get_dummies(all_data)
all_data = all_data.fillna(all_data.mean())

##df.drop(['SalePrice'], axis = 1,inplace=True)
##df.drop(['Id'], axis = 1, inplace=True)

x_train = all_data[:df.shape[0]]
x_test = all_data[df.shape[0]:]
y = df.SalePrice


#SVR with select percentile
x_select_percentile = SelectPercentile(f_regression,percentile=10).fit_transform(x_train,y)
model_svm = svm.SVR(C=1, epsilon=0.2, kernel='linear')
svm_rmse_cv = rmse_cv(model_svm,x_select_percentile,y).mean()
print("Select percentile Mean RMSE with 5-fold cv: "+str(svm_rmse_cv))

#SVR with RFE
model_svm = svm.SVR(C=1, epsilon=0.2,kernel='linear')
rfe = RFE(model_svm,step=1,n_features_to_select = 10)
rfe_rmse_cv = rmse_cv(rfe,x_train,y).mean()
print("RFE Mean RMSE with 5-fold cv: "+str(rfe_rmse_cv))

#MODEL FITTING
#ridge
##model_ridge = Ridge()
##alphas = [0.05, 0.1, 0.3, 1, 3, 5, 10, 15, 30, 50, 75]
##cv_ridge = [rmse_cv(Ridge(alpha=alpha)).mean() for alpha in alphas]
##cv_ridge = pd.Series(cv_ridge, index = alphas)
##cv_ridge.plot(title = "Validation")
##plt.xlabel("alpha")
##plt.ylabel("rmse")
##plt.show()
###the plot shows that an alpha of 10 will give us the lowest rmse
##print(cv_ridge.min())
###we get an rmsle of 0.127
model_ridge = Ridge(alpha=10)
model_ridge.fit(x_train,y)
rmse_ridge =  rmse_cv(model_ridge,x_train,y).mean()
print("Ridge Mean RMSE with 5-fold cv: "+str(rmse_ridge))
###lasso
model_lasso = LassoCV(alphas = [1,0.1,0.001,0.0005]).fit(x_train,y)
lasso_rmse_cv = rmse_cv(model_lasso,x_train,y).mean()
print("Lasso Mean RMSE with 5-fold cv: "+str(lasso_rmse_cv))
#lasso gives us an rmsle of 0.123

coef = pd.Series(model_lasso.coef_, index = x_train.columns)
##print("Lasso picked " + str(sum(coef != 0)) + " variables and left out "
##      + str(sum(coef == 0)) + "variables")

###predicting test data
##ridge_p = np.expm1(model_ridge.predict(x_test))
##lasso_p = np.expm1(model_lasso.predict(x_test))
##
##
##ridge_solution = pd.DataFrame({"id":df_test.Id,"SalePrice":ridge_p})
##lasso_solution = pd.DataFrame({"id":df_test.Id,"SalePrice":lasso_p})
##ridge_solution.to_csv("ridge_solution.csv", index=False)
##lasso_solution.to_csv("lasso_solution.csv", index=False)
