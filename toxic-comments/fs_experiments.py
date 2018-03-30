from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import RFE
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectFromModel
from toxic_comments import *

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

##import warnings
##warnings.filterwarnings("ignore")
y = train.iloc[:,2:]
LABELS = ['toxic','severe_toxic','obscene','threat','insult','identity_hate']
x = train.drop(labels = ['toxic','severe_toxic','obscene','threat','insult','identity_hate'],axis=1)
x['comment_text'].fillna("unknown", inplace=True)
x = x.drop(labels = ['id'],axis = 1)
test_ids = test.drop(labels = ['comment_text'], axis = 1)
test = test.drop(labels = ['id'],axis= 1)
test['comment_text'].fillna("unknown", inplace=True)
del train

df = pd.concat([x['comment_text'], test['comment_text']], axis=0)
#GET TFIDF FEATURE
tfidf_vec = TfidfVectorizer()
tfidf_vec.fit(df)
train_tfidf = tfidf_vec.transform(x['comment_text'])
test_tfidf = tfidf_vec.transform(test['comment_text'])

model = LogisticRegression()
#RFE MODELS
rfe_estimator = LogisticRegression()
rfe_benchmarks = benchmark('rfe-nfeats-300000',model,train_tfidf, y,fs=RFE(rfe_estimator, step=0.05, n_features_to_select=300000))
print(rfe_benchmarks)
#write_dict_to_csv(rfe_benchmarks,'benchmarks.csv')
rfe_benchmarks = benchmark('rfe-nfeats-250000',model,train_tfidf, y,fs=RFE(rfe_estimator, step=0.05, n_features_to_select=250000))
#write_dict_to_csv(rfe_benchmarks,'benchmarks.csv')
rfe_benchmarks = benchmark('rfe-nfeats-200000',model,train_tfidf, y,fs=RFE(rfe_estimator, step=0.05, n_features_to_select=200000))
#write_dict_to_csv(rfe_benchmarks,'benchmarks.csv')
rfe_benchmarks = benchmark('rfe-nfeats-150000',model,train_tfidf, y,fs=RFE(rfe_estimator, step=0.05, n_features_to_select=150000))
#write_dict_to_csv(rfe_benchmarks,'benchmarks.csv')
rfe_benchmarks = benchmark('rfe-nfeats-100000',model,train_tfidf, y,fs=RFE(rfe_estimator, step=0.05, n_features_to_select=100000))
#write_dict_to_csv(rfe_benchmarks,'benchmarks.csv')
rfe_benchmarks = benchmark('rfe-nfeats-50000',model,train_tfidf, y,fs=RFE(rfe_estimator, step=0.05, n_features_to_select=50000))
#write_dict_to_csv(rfe_benchmarks,'benchmarks.csv')
rfe_benchmarks = benchmark('rfe-nfeats-25000',model,train_tfidf, y,fs=RFE(rfe_estimator, step=0.05, n_features_to_select=25000))
#write_dict_to_csv(rfe_benchmarks,'benchmarks.csv')
rfe_benchmarks = benchmark('rfe-nfeats-10000',model,train_tfidf, y,fs=RFE(rfe_estimator, step=0.05, n_features_to_select=10000))
#write_dict_to_csv(rfe_benchmarks,'benchmarks.csv')
rfe_benchmarks = benchmark('rfe-nfeats-5000',model,train_tfidf, y,fs=RFE(rfe_estimator, step=0.05, n_features_to_select=5000))
#write_dict_to_csv(rfe_benchmarks,'benchmarks.csv')



#FEATURE SELECTION EXPERIMENTS
##    print("SelectKBest")
num_features = len(tfidf_vec.get_feature_names())
k1 = int(num_features*0.1)
k2 = int(num_features*0.25)
k3 = int(num_features*0.5)
k4 = int(num_features*0.75)
k_list = [k1, k2, k3, k4]
kbest_find_k(k_list, tfidf, y)
#k2((num_features)*0.25) found to be the best  for kbest

kbest_benchmarks = benchmark(SelectKBest(chi2, k=k2), model, tfidf, y)

#SELECT FROM MODEL EXPERIMENTS
sfm = SelectFromModel(LogisticRegression(), threshold="0.1*mean")
sfm_benchmarks = benchmark("sfm-th-0.1*mean",model, train_tfidf, y, fs=sfm)
#write_dict_to_csv(sfm_benchmarks,'benchmarks.csv')

sfm = SelectFromModel(LogisticRegression(), threshold="0.2*mean")
sfm_benchmarks = benchmark("sfm-th-0.2*mean",model, train_tfidf, y, fs=sfm)  
#write_dict_to_csv(sfm_benchmarks,'benchmarks.csv')

sfm = SelectFromModel(LogisticRegression(), threshold="0.3*mean")
sfm_benchmarks = benchmark("sfm-th-0.3*mean",model, train_tfidf, y, fs=sfm)
#write_dict_to_csv(sfm_benchmarks,'benchmarks.csv')

sfm = SelectFromModel(LogisticRegression(), threshold="0.4*mean")
sfm_benchmarks = benchmark("sfm-th-0.4*mean",model, train_tfidf, y, fs=sfm)
#write_dict_to_csv(sfm_benchmarks,'benchmarks.csv')

sfm = SelectFromModel(LogisticRegression(), threshold="0.5*mean")
sfm_benchmarks = benchmark("sfm-th-0.5*mean",model, train_tfidf, y, fs=sfm)
#write_dict_to_csv(sfm_benchmarks,'benchmarks.csv')


#PREDICTIONS FOR OPTIMAL FS MODELS
sfm_preds = make_prediction(train_tfidf, y, test_tfidf, test_ids, fs=SelectFromModel(LogisticRegression(), threshold="0.1*mean"), model=model)
rfe_preds = make_prediction(train_tfidf, y, test_tfidf, test_ids, fs=RFE(LogisticRegression(),step=1, n_features_to_select=None),model=model)
kbest_preds = make_prediction(train_tfidf, y, test_tfidf, test_ids,fs = SelectKBest(chi2, k=k2), model=model)
#to csv
rfe_preds.to_csv('submission.csv',index=False)
sfm_preds.to_csv('submission.csv',index=False)
kbest_preds.to_csv('submission.csv',index = False)
