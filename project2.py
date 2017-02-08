import warnings
import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

warnings.filterwarnings("ignore")

### PRINT ALL 20 NEWSGROUPS ###
newsgroups_train = fetch_20newsgroups(subset='train')
print("Names of All Newsgroups: " + str(list(newsgroups_train.target_names)))
print('\n')

### PART A ###
comp_categories = ['comp.graphics','comp.os.ms-windows.misc','comp.sys.ibm.pc.hardware','comp.sys.mac.hardware']
rec_categories = ['rec.autos','rec.motorcycles','rec.sport.baseball','rec.sport.hockey']

comp_train = fetch_20newsgroups(subset='train', categories=comp_categories, shuffle=True, random_state=42)
comp_test = fetch_20newsgroups(subset='test', categories=comp_categories, shuffle=True, random_state=42)
rec_train = fetch_20newsgroups(subset='train', categories=rec_categories, shuffle=True, random_state=42)
rec_test = fetch_20newsgroups(subset='test', categories=rec_categories, shuffle=True, random_state=42)

print("Training Set - Computer Technology: %s Recreation: %s" %(comp_train.filenames.shape[0],rec_train.filenames.shape[0]))
print("Test Set - Computer Technology: %s Recreation: %s" %(comp_test.filenames.shape[0],rec_test.filenames.shape[0]))
print('\n')

comp_train_list=comp_train.target.tolist()
rec_train_list=rec_train.target.tolist()
comp_test_list=comp_test.target.tolist()
rec_test_list=rec_test.target.tolist()

train_counts = [[comp_train_list.count(x)] for x in set(comp_train_list)] + [[rec_train_list.count(x)] for x in set(rec_train_list)]
test_counts = [[comp_test_list.count(x)] for x in set(comp_test_list)] + [[rec_test_list.count(x)] for x in set(rec_test_list)]

objects = ('graphics', 'windows', 'ibm', 'mac', 'autos', 'cycles','baseball','hockey')
y_pos = np.arange(len(objects))

plt.bar(y_pos, np.array(train_counts), align='center', alpha=0.5)
plt.xticks(y_pos, objects)
plt.ylabel('Counts')
plt.title('Counts for Training Set')
plt.tight_layout()
plt.show()

plt.bar(y_pos, np.array(test_counts), align='center', alpha=0.5)
plt.xticks(y_pos, objects)
plt.ylabel('Counts')
plt.title('Counts for Test Set')
plt.tight_layout()  
plt.show()

### PART B & PART C ###
categories = list(newsgroups_train.target_names)
elements = ['comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'misc.forsale', 'soc.religion.christian']

vectorizer = CountVectorizer(analyzer = 'word',stop_words='english', token_pattern=u'(?u)\\b\\w\\w+\\b')
tfidf_transformer = TfidfTransformer()
for category in categories:

    twenty_train = fetch_20newsgroups(subset='train', categories=[category], shuffle=True, random_state=42)
    X_train_counts = vectorizer.fit_transform(twenty_train.data)
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
    print "Number of terms extracted for %s: " %(category) + str(X_train_tfidf.shape[1])

    if category in elements:

        X_train_tfidf = np.sum(X_train_tfidf.toarray(), axis=0)
        most_important_word_indices = np.argsort(X_train_tfidf)[::-1][:10]
        most_important_words = [vectorizer.get_feature_names()[i] for i in most_important_word_indices]
        print "Most Important Ten Words for %s: " %(category) + str(most_important_words)
