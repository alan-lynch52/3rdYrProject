#https://www.kaggle.com/c/quora-question-pairs
import sys
sys.path.insert(0,'..')
from toxic_comments import *
def main():
    train = pd.read_csv('q_pairs_train.csv')
    print(list(train))
    print(train.isnull.any.values())
    q1 = train['question1']
    q2 = train['question2']
    y = train['is_duplicate']
    tfidf_vec = TfidfVectorizer()
    tfidf_q1 = tfidf_vec.fit_transform(q1)
    tfidf_q2 = tfidf_vec.fit_transform(q2)
    
if __name__ == '__main__':
    main()
