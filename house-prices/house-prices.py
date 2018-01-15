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
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV, ElasticNetCV, ElasticNet, Lasso, LassoCV, LassoLarsCV
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_regression
from sklearn.feature_selection import RFE
from sklearn import svm
from boruta import BorutaPy
import warnings
warnings.filterwarnings("ignore")

def rmse_cv(model,x,y):
    rmse = np.sqrt(-cross_val_score(model,x,y,scoring="neg_mean_squared_error",cv = 5))
    return(rmse)

#import dataset into dataframe using pandas
df = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')
all_data = pd.concat((df.loc[:,'MSSubClass':'SaleCondition'],
                      df_test.loc[:,'MSSubClass':'SaleCondition']))

df.drop('Id',axis=1,inplace=True)
#EDA - Exploratory data analysis
#first we will use a scatterplot matrix to see the pair-wise correlations
#between the features
corrmat = df.corr()
graph = plt.axes()

sb.heatmap(corrmat, vmax=0.8, square=True, ax=graph)
graph.set_title("Correlation Heatmap")
#plt.show()
plt.clf()
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

continous_feats = df.select_dtypes(include=[np.number])
print(continous_feats.columns)


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

#logging saleprice and logging skewed continous features
df['SalePrice'] = np.log1p(df['SalePrice'])
numeric_features = all_data.dtypes[all_data.dtypes != "object"].index
skewed_features = all_data[numeric_features].apply(lambda x: skew(x.dropna()))
skewed_features = skewed_features[skewed_features > 0.75]
skewed_features = skewed_features.index
all_data[skewed_features] = np.log1p(all_data[skewed_features])
#get dummie features from categorical features
all_data = pd.get_dummies(all_data)
#fill all missing data with mean for that column
all_data = all_data.fillna(all_data.mean())

#split data into training, testing and y
x_train = all_data[:df.shape[0]]
x_test = all_data[df.shape[0]:]
y = df.SalePrice

###SVR with select percentile
x_select_percentile = SelectPercentile(f_regression,percentile=50).fit_transform(x_train,y)
model_ridge = Ridge(alpha=10)
model_ridge.fit(x_select_percentile,y)
uvf_rmse = rmse_cv(model_ridge, x_select_percentile,y)

#RFE
model_ridge = Ridge(alpha=10)
rfe = RFE(model_ridge, step=1)
rfe.fit(x_train,y)
rfe_rmse = rmse_cv(rfe,x_train,y)
print("RFE Mean RMSE with 5-fold cv: "+str(rfe_rmse))

#RIDGE
model_ridge = Ridge()
alphas = [0.05, 0.1, 0.3, 1, 3, 5, 10, 15, 30, 50, 75]
cv_ridge = [rmse_cv(Ridge(alpha=alpha),x_train,y).mean() for alpha in alphas]
cv_ridge = pd.Series(cv_ridge, index = alphas)
##cv_ridge.plot(title = "Validation")
##plt.xlabel("alpha")
##plt.ylabel("rmse")
##plt.show()
#the plot shows that an alpha of 10 will give us the lowest rmse
#we get an rmsle of 0.127
model_ridge = Ridge(alpha=10)
model_ridge.fit(x_train,y)
ridge_rmse =  rmse_cv(model_ridge,x_train,y)
print("Ridge Mean RMSE with 5-fold cv: "+str(ridge_rmse))

#LASSO
lasso_alphas = [1,0.1,0.01,0.001]
cv_lasso = [rmse_cv(Lasso(alpha=alpha),x_train,y).mean() for alpha in lasso_alphas]
cv_lasso = pd.Series(cv_lasso, index = lasso_alphas)
#cv_lasso.plot(title="Validation of lasso")
#plt.xlabel("alpha")
#plt.ylabel("rmse")
#plt.show()
model_lasso = Lasso(alpha = 0.001).fit(x_train,y)
lasso_rmse = rmse_cv(model_lasso,x_train,y)
print("Lasso Mean RMSE with 5-fold cv: "+str(lasso_rmse))

#ELASTIC NET
en_alphas = [0.01,0.001,0.0001]
cv_en = [rmse_cv(ElasticNet(alpha=alpha), x_train, y).mean() for alpha in en_alphas]
cv_en = pd.Series(cv_en, index=en_alphas)
#cv_en.plot(title="Validation of ElasticNet")
##plt.xlabel("alpha")
##plt.ylabel("rmse")
##plt.show()
#the plot shows that an alpha of 0.001 will give the lowest rmse for elastic net

model_elasticnet = ElasticNet(alpha = 0.001).fit(x_train,y)
en_rmse = rmse_cv(model_elasticnet, x_train, y)

#BORUTA
#boruta needs numpy arrays
boruta_x = x_train.values
boruta_y = y.values
boruta_y = y.ravel()

rfr = RandomForestRegressor()
boruta_model = BorutaPy(rfr, n_estimators='auto', random_state=1)
boruta_model.fit(boruta_x,boruta_y)

filtered_x = boruta_model.transform(boruta_x)
##boruta_cv = [rmse_cv(Ridge(alpha=alpha),filtered_x,boruta_y).mean() for alpha in alphas]
##boruta_cv = pd.Series(boruta_cv, index=alphas)
##boruta_cv.plot(title="Validation of Boruta")
##plt.xlabel("alpha")
##plt.ylabel("rmse")
##plt.show()
boruta_model = Ridge(alpha=1)
boruta_rmse = rmse_cv(boruta_model, filtered_x,boruta_y)
print("Boruta RMSE values: " + str(boruta_rmse))

plt.plot(en_rmse)
plt.plot(lasso_rmse)
plt.plot(ridge_rmse)
plt.plot(rfe_rmse)
plt.plot(uvf_rmse)
plt.plot(boruta_rmse)
plt.xlabel("Cross validation iteration")
plt.ylabel("rmse")
plt.legend(['Elastic Net','Lasso','Ridge','RFE','UVF','boruta'],loc='upper left')
plt.show()
print("ElasticNet Mean RMSE with 5-fold cv: "+str(en_rmse))
##print("Lasso Mean RMSE with 5-fold cv: "+str(lasso_rmse_cv))
###lasso gives us an rmsle of 0.123
##
##coef = pd.Series(model_lasso.coef_, index = x_train.columns)
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
