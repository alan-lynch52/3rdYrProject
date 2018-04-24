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
#test_ids = test.drop(labels = ['comment_text'], axis = 1)
#test = test.drop(labels = ['id'],axis= 1)
#test['comment_text'].fillna("unknown", inplace=True)
del train
df = x['comment_text']
#df = pd.concat([x['comment_text'], test['comment_text']], axis=0)
#FEATURE EXTRACTION EXPERIMENTS
#GET BINARY FEATURE
binary_vec = TfidfVectorizer(use_idf = False,norm = None, binary = True)
binary_vec.fit(df)
bin_train = binary_vec.transform(x['comment_text'])
#bin_test = binary_vec.transform(test['comment_text'])

#GET TFIDF WORD FEATURE
tfidf_vec = TfidfVectorizer()
tfidf_vec.fit(df)
train_tfidf = tfidf_vec.transform(x['comment_text'])
#test_tfidf = tfidf_vec.transform(test['comment_text'])



#stack word and char vectors
#train_word_char = hstack([train_tfidf, train_tfidf_char])
#test_word_char = hstack([test_tfidf, test_tfidf_char])

#GET TERM FREQUENCY FEATURE
countVectorizer = CountVectorizer()
countVectorizer.fit(df)
count_train = countVectorizer.transform(x['comment_text'])
#count_test = countVectorizer.transform(test['comment_text'])

#Get benchmarks
model = LogisticRegression()

#train_x = hstack([count_train, bin_train])
#train_x = hstack([train_x,bin_train])
#test_x = hstack([count_test, bin_test])
#test_x = hstack([test_x,bin_test])

##stack_benchmarks = benchmark("stack-count-binary",model,train_x, y)
##write_dict_to_csv(stack_benchmarks,'benchmarks.csv')
##stack_tfidf_benchmarks = benchmark('stack-tfidf-word-char-ngram-1-6',model,train_word_char,y)
##write_dict_to_csv(stack_tfidf_benchmarks, 'benchmarks.csv')
##
##stack_preds = make_prediction(train_x, y, test_x, test_ids, fs=None, model=model)
##tfidf_stack_preds = make_prediction(train_word_char, y, test_word_char, test_ids, fs=None, model=model)
##stack_preds.to_csv('submission.csv',index=False)
##tfidf_stack_preds.to_csv('tfidf-stack-submission.csv',index=False)
d = collections.OrderedDict()
d['stack1'] = hstack([train_tfidf, count_train, bin_train])
d['stack2'] = hstack([train_tfidf, count_train])
d['stack3'] = hstack([train_tfidf, bin_train])
d['stack4'] = hstack([count_train, bin_train])
del bin_train
del count_train
#GET TFIDF CHAR FEATURE
char_vec = TfidfVectorizer(analyzer='char',ngram_range=(4,6),max_features=100000)
char_vec.fit(df)
train_tfidf_char_ngram = char_vec.transform(x['comment_text'])

char_vec = TfidfVectorizer(analyzer='char',max_features=100000)
char_vec.fit(df)
train_tfidf_char = char_vec.transform(x['comment_text'])

#test_tfidf_char = char_vec.transform(test['comment_text'])
d['stack5'] = hstack([train_tfidf, train_tfidf_char])
d['stack6'] = hstack([train_tfidf, train_tfidf_char_ngram])
#get_auroc_fe(d,y)
get_balanced_accuracy_fe(d,y)
#plot_cm_fe(d,y)
