from toxic_comments import *

train = pd.read_csv("train.csv")
#test = pd.read_csv("test.csv")

##import warnings
##warnings.filterwarnings("ignore")
y = train.iloc[:,2:]
LABELS = ['toxic','severe_toxic','obscene','threat','insult','identity_hate']
x = train.drop(labels = ['toxic','severe_toxic','obscene','threat','insult','identity_hate'],axis=1)
x['comment_text'].fillna("unknown", inplace=True)
x = x.drop(labels = ['id'],axis = 1)
#test_ids = test.drop(labels = ['comment_text'], axis = 1)
#test = test.drop(labels = ['id'],axis= 1)
#test['comment_text'].fillna("unknown", inplace=True)
del train
df = x['comment_text']
#df = pd.concat([x['comment_text'], test['comment_text']], axis=0)
#GET BINARY FEATURE
binary_vec = TfidfVectorizer(use_idf = False,norm = None, binary = True)
binary_vec.fit(df)
bin_train = binary_vec.transform(x['comment_text'])
#bin_test = binary_vec.transform(test['comment_text'])
#GET TFIDF FEATURE

tfidf_vec = TfidfVectorizer()
tfidf_vec.fit(df)
train_tfidf = tfidf_vec.transform(x['comment_text'])
#test_tfidf = tfidf_vec.transform(test['comment_text'])

# GET TERM FREQUENCY FEATURE
countVectorizer = CountVectorizer(analyzer='word',ngram_range=(1,2),min_df=1e-05)
countVectorizer.fit(df)
count_train = countVectorizer.transform(x['comment_text'])
#count_test = countVectorizer.transform(test['comment_text'])

model = LogisticRegression()

d = collections.OrderedDict()
d['TF-IDF-Base'] = train_tfidf
d['Count-Best'] = count_train
d['Binary'] = bin_train
get_balanced_accuracy_fe(d,y)
#get_auroc_fe(d,y)
#plot_cm_fe(d,y)
##bin_benchmarks = benchmark("binary",model, bin_train, y)
##write_dict_to_csv(bin_benchmarks, 'benchmarks.csv')
##
##tfidf_benchmarks = benchmark("char-ngram-1-2-mindf-0.001-max-features-1000",model, train_tfidf, y)
##write_dict_to_csv(tfidf_benchmarks, 'benchmarks.csv')
##
##count_benchmarks = benchmark("count-char-ngram-1-2-mindf-0.000001-maxfeatures-10000",model, count_train, y)
##write_dict_to_csv(count_benchmarks, 'benchmarks.csv')
##
##
##tfidf_preds = make_prediction(train_tfidf, y, test_tfidf, test_ids)
##count_preds = make_prediction(count_train, y, count_test, test_ids)
##bin_preds = make_prediction(bin_train, y, bin_test, test_ids)
##bin_preds.to_csv('submission.csv', index=False)
##tfidf_preds.to_csv('submission.csv',index=False)
##count_preds.to_csv('submission.csv',index=False)
