import re
import math
import string
import operator
import warnings
import numpy as np
import matplotlib.pyplot as plt; plt.rcdefaults()
import sklearn.linear_model as sk
from sklearn.multiclass import OneVsOneClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction import text
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report, confusion_matrix, roc_curve
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC, SVC
from statsmodels.discrete.discrete_model import Logit
from nltk import SnowballStemmer
from collections import Counter
from collections import defaultdict

### PRINT ALL 20 NEWSGROUPS ###
# newsgroups_train = fetch_20newsgroups(subset='train')
# print("Names of All Newsgroups: " + str(list(newsgroups_train.target_names)))
# print('\n')

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
plt.savefig('training_set_histogram.png')
#plt.show()

plt.bar(y_pos, np.array(test_counts), align='center', alpha=0.5)
plt.xticks(y_pos, objects)
plt.ylabel('Counts')
plt.title('Counts for Test Set')
plt.tight_layout()
plt.savefig('test_set_histogram.png')
#plt.show()

### PART B ###
def tokenize(data):

    stemmer = SnowballStemmer("english")
    stop_words = text.ENGLISH_STOP_WORDS
    temp = data
    temp = re.sub('[,.-:/()?{}*$#&><@...-]', ' ', temp)
    temp = "".join([a for a in temp if a not in string.punctuation])
    temp = "".join(b for b in temp if ord(b) < 128)
    temp = temp.lower()
    words = temp.split()
    no_stop_words = [w for w in words if not w in stop_words]
    stemmed = [stemmer.stem(item) for item in no_stop_words]

    return stemmed

categories = comp_categories + rec_categories
vectorizer = CountVectorizer(analyzer='word', stop_words='english', tokenizer=tokenize)
tfidf_transformer = TfidfTransformer()
twenty_train = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=42)
X_train_counts = vectorizer.fit_transform(twenty_train.data)
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
print "Number of terms extracted: " + str(X_train_tfidf.shape)

### PART C ###
newsgroups_train = fetch_20newsgroups(subset='train')
all_newsgroups = newsgroups_train.target_names

index_ibm_pc = all_newsgroups.index("comp.sys.ibm.pc.hardware")
index_mac = all_newsgroups.index("comp.sys.mac.hardware")
index_forsale = all_newsgroups.index("misc.forsale")
index_religion = all_newsgroups.index("soc.religion.christian")

class_indices = [index_ibm_pc, index_mac, index_forsale, index_religion]

all_data = []
all_words = []
all_word_freqs = []
word_class_dict = defaultdict(list)

for category in all_newsgroups:
    newsgroup_category = fetch_20newsgroups(subset='train', categories=[category], shuffle=True, random_state=42)
    newsgroup_category_data = newsgroup_category.data
    temp = ''
    for file in newsgroup_category_data:
        temp += ' ' + file
    all_data.append(temp)

for class_data,index in zip(all_data, range(len(all_data))):
    tokenize_data = tokenize(class_data)
    unique_words = set(tokenize_data)
    all_words.append(list(unique_words))
    word_count = Counter(tokenize_data)
    all_word_freqs.append(word_count)
    for word in unique_words:
        word_class_dict[word].append(all_newsgroups[index])

for class_index in class_indices:
    terms_extracted_in_class = all_words[class_index]
    freq_of_terms_in_class = all_word_freqs[class_index]
    number_of_terms_extracted = len(terms_extracted_in_class)
    maxFreq = max(freq_of_terms_in_class.values())
    tficf = dict()
    
    for each_term in range(number_of_terms_extracted):
        term = terms_extracted_in_class[each_term]
        frequency_term = freq_of_terms_in_class.get(term)
        number_of_classes_with_term = len(word_class_dict[term])
        tficf[term] = (0.5 + (0.5 * frequency_term/maxFreq)) * math.log(len(all_newsgroups)/number_of_classes_with_term)

    print "Most significant 10 terms for class: " + str(all_newsgroups[class_index])
    most_significant_terms = dict(sorted(tficf.iteritems(), key=operator.itemgetter(1), reverse=True)[:10])
    print (most_significant_terms.keys())
