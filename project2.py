import warnings
import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt
import re
from sklearn.datasets import fetch_20newsgroups
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report, confusion_matrix, roc_curve
from sklearn.svm import LinearSVC
from nltk import SnowballStemmer
import string

warnings.filterwarnings("ignore")

### PRINT ALL 20 NEWSGROUPS ###
# newsgroups_train = fetch_20newsgroups(subset='train')
# print("Names of All Newsgroups: " + str(list(newsgroups_train.target_names)))
# print('\n')

### PART A ###
comp_categories = ['comp.graphics','comp.os.ms-windows.misc','comp.sys.ibm.pc.hardware','comp.sys.mac.hardware']
rec_categories = ['rec.autos','rec.motorcycles','rec.sport.baseball','rec.sport.hockey']

'''
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

###TODO: Uncomment Before Submission ###
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
'''

### PART B ###
def tokenize(data):

    stemmer = SnowballStemmer("english")
    temp = data
    temp = "".join([a for a in temp if a not in set(string.punctuation)])
    temp = re.sub('[,.-:/()?{}*$#&]', ' ', temp)
    temp = "".join(b for b in temp if ord(b) < 128)
    words = temp.split()
    stemmed = [stemmer.stem(item) for item in words]

    return stemmed

'''
###TODO: Remove
### This part is for testing ###
corpus = [
    'This is the first document.',
    'This is the second22 second document.',
    'And the third one.',
    'Is this the first document?',
    'He will play playing plays...',
    'Walking I playing 5898 swimming',
    'Thanks walks to school swim go play bee'
]

vectorizer = CountVectorizer(analyzer='word', stop_words='english', tokenizer=tokenize)
tfidf_transformer = TfidfTransformer()
X_train_counts = vectorizer.fit_transform(corpus)
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

print X_train_counts.shape
print vectorizer.vocabulary_.keys()
print X_train_tfidf.toarray()

stop_words = text.ENGLISH_STOP_WORDS
print list(stop_words).index("about")

'''

categories = comp_categories + rec_categories
vectorizer = CountVectorizer(analyzer='word', stop_words='english', tokenizer=tokenize)
tfidf_transformer = TfidfTransformer()
twenty_train = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=42)
X_train_counts = vectorizer.fit_transform(twenty_train.data)
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
print "Number of terms extracted: " + str(X_train_tfidf.shape)

'''
### PART C ###
elements = ['comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'misc.forsale', 'soc.religion.christian']

vectorizer = CountVectorizer(analyzer = 'word',stop_words='english', token_pattern=u'(?u)\\b\\w\\w+\\b')
tfidf_transformer = TfidfTransformer()
for category in elements:

    twenty_train = fetch_20newsgroups(subset='train', categories=[category], shuffle=True, random_state=42)
    X_train_counts = vectorizer.fit_transform(twenty_train.data)
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
    print "Number of terms extracted for %s: " %(category) + str(X_train_tfidf.shape[1])

    X_train_tfidf = np.sum(X_train_tfidf.toarray(), axis=0)
    most_important_word_indices = np.argsort(X_train_tfidf)[::-1][:10]
    most_important_words = [vectorizer.get_feature_names()[i] for i in most_important_word_indices]
    print "Most Important Ten Words for %s: " %(category) + str(most_important_words)
'''

### Part D: Dimension Reduction ###
svd = TruncatedSVD(n_components=50, random_state=42)
X_train_reduced = svd.fit_transform(X_train_tfidf)
print "Size of TF-IDF matrix after dimension reduction: " + str(X_train_reduced.shape)

### Part E: Linear Suppor Vector Machines ###
# First obtain test data and perform above transformations
twenty_test = fetch_20newsgroups(subset='test', categories=categories, shuffle=True, random_state=42)
Y_test_counts = vectorizer.transform(twenty_test.data)
Y_test_tfidf = tfidf_transformer.transform(Y_test_counts)
Y_test_reduced = svd.transform(Y_test_tfidf)

# Second reduce multiclass problem to binary classfication problem
# by reducing all subclasses of comp and subclasses of rec
# for train and test data => comp = 0, rec = 1

train_targets = map(lambda x: int(x>=4), twenty_train.target.tolist())
test_targets = map(lambda x: int(x>=4), twenty_test.target.tolist())

# Train model and predict labels for test set
linear_SVM = LinearSVC(dual=False, random_state=42).fit(X_train_reduced, train_targets)
predicted_svm = linear_SVM.predict(Y_test_reduced)
accuracy_svm = np.mean(predicted_svm == test_targets)

# Report results
print "Accuracy of Linear SVM: " + str(accuracy_svm)
print(classification_report(test_targets, predicted_svm))
print "Confusion Matrix:"
print(confusion_matrix(test_targets, predicted_svm))

fpr, tpr, thresholds = roc_curve(test_targets, predicted_svm)
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Linear SVM: Receiver Operating Characteristic Curve')
plt.legend(loc="lower right")
plt.savefig('roc_curve.png')
plt.show()
