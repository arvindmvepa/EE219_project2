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
from sklearn import preprocessing, cross_validation
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
        temp = temp + file
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

    tficf = dict()
    for each_term in range(number_of_terms_extracted):
        term = terms_extracted_in_class[each_term]
        frequency_term = freq_of_terms_in_class.get(term)
        number_of_classes_with_term = len(word_class_dict[term])
        tficf[term] = 0.5 + ((0.5 * frequency_term/number_of_terms_extracted) * math.log(len(all_newsgroups)/number_of_classes_with_term))

    print "Most significant 10 terms for class: " + str(all_newsgroups[class_index])
    most_significant_terms = dict(sorted(tficf.items(), key=operator.itemgetter(1), reverse=True)[:10])
    print (most_significant_terms.keys())

### Part D: Dimension Reduction ###
svd = TruncatedSVD(n_components=50, random_state=42)
X_train_reduced = svd.fit_transform(X_train_tfidf)
print "Size of TF-IDF matrix after dimension reduction: " + str(X_train_reduced.shape)

# Feature Scaling For Certain Algorithms Require Nonnegative Values
min_max_scaler = preprocessing.MinMaxScaler()
X_train = min_max_scaler.fit_transform(X_train_reduced)

### Part E: Linear Suppor Vector Machines ###
# First obtain test data and perform above transformations
twenty_test = fetch_20newsgroups(subset='test', categories=categories, shuffle=True, random_state=42)
X_test_counts = vectorizer.transform(twenty_test.data)
X_test_tfidf = tfidf_transformer.transform(X_test_counts)
X_test_reduced = svd.transform(X_test_tfidf)
X_test = min_max_scaler.transform(X_test_reduced)

# Second reduce multiclass problem to binary classfication problem
# by reducing all subclasses of comp and subclasses of rec
# for train and test data => comp = 0, rec = 1

train_targets = map(lambda x: int(x>=4), twenty_train.target.tolist())
test_targets = map(lambda x: int(x>=4), twenty_test.target.tolist())

# Train model and predict labels for test set
linear_SVM = LinearSVC(dual=False, random_state=42).fit(X_train, train_targets)
predicted_svm = linear_SVM.predict(X_test)
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
plt.savefig('svm_roc_curve.png')
plt.show()

### Part F: Soft Margin SVM ###
accuracies = []
predictions = dict()
gamma_values = [10 ** i for i in range(-3,4)]

for value in gamma_values:
    soft_margin_SVM = SVC(C=value, kernel='linear').fit(X_train, train_targets)
    scores = cross_validation.cross_val_score(soft_margin_SVM, np.concatenate((X_train, X_test), axis=0),
                                np.append(train_targets, test_targets), cv=5, scoring='accuracy')
    accuracies.append(np.average(scores))

best_gamma_value = gamma_values[accuracies.index(max(accuracies))]

soft_margin_SVM = SVC(C=best_gamma_value, kernel='linear').fit(X_train, train_targets)
predicted_soft_svm = soft_margin_SVM.predict(X_test)
accuracy_soft_svm = np.mean(predicted_soft_svm == test_targets)

print "Best Gamma Value: " + str(gamma_values[accuracies.index(max(accuracies))])
print "Accuracy of Soft Margin SVM: " + str(accuracy_soft_svm)
print(classification_report(test_targets, predicted_soft_svm))
print "Confusion Matrix:"
print(confusion_matrix(test_targets, predicted_soft_svm))

fpr, tpr, thresholds = roc_curve(test_targets, predicted_soft_svm)
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rte')
plt.title('Soft Margin SVM: Receiver Operating Characteristic Curve')
plt.legend(loc="lower right")
plt.savefig('soft_margin_svm_roc_curve.png')
plt.show()

### Part G: Naive Bayes ###
clf = MultinomialNB().fit(X_train, train_targets)
predicted_bayes = clf.predict(X_test)
accuracy_bayes = np.mean(predicted_bayes == test_targets)

print "Accuracy of Multinomial Naive Bayes: " + str(accuracy_bayes)
print(classification_report(test_targets, predicted_bayes))
print "Confusion Matrix:"
print(confusion_matrix(test_targets, predicted_bayes))

fpr, tpr, thresholds = roc_curve(test_targets, predicted_bayes)
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Multinomial Naive Bayes: Receiver Operating Characteristic Curve')
plt.legend(loc="lower right")
plt.savefig('mnnb_roc_curve.png')
plt.show()

### Part H: Logistic Regression (Unregularized) ###
logit = sk.LogisticRegression().fit(X_train, train_targets)
probabilities = logit.predict(X_test)
predicted_lr = (probabilities > 0.5).astype(int)
accuracy_lr = np.mean(predicted_lr == test_targets)

print "Accuracy of Logistic Regression (Unregularized): " + str(accuracy_lr)
print(classification_report(test_targets, predicted_lr))
print "Confusion Matrix:"
print(confusion_matrix(test_targets, predicted_lr))

fpr, tpr, thresholds = roc_curve(test_targets, predicted_lr)
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Logistic Regression (Unregularized): Receiver Operating Characteristic Curve')
plt.legend(loc="lower right")
plt.savefig('lru_roc_curve.png')
plt.show()

### Part I: Logistic Regression (Regularized) ###
for lam in [.0001,.001,.01,.5,1,10,100]:
    logit_r = sk.LogisticRegression(penalty='l1', C = float(1)/lam).fit(X_train, train_targets)
    probabilities_r = logit_r.predict(X_test)
    predicted_lr_r = (probabilities_r > 0.5).astype(int)
    accuracy_lr_r = np.mean(predicted_lr_r == test_targets)

    print "Accuracy of Logistic Regression L1 Regularization for lambda = "+str(lam)+ ": " + str(accuracy_lr_r)
    print(classification_report(test_targets, predicted_lr_r))
    print "Confusion Matrix:"
    print(confusion_matrix(test_targets, predicted_lr_r))

for lam in [.0001,.001,.01,.5,1,10,100]:
    logit_r = sk.LogisticRegression(penalty='l2', C = float(1)/lam).fit(X_train, train_targets)
    probabilities_r = logit_r.predict(X_test)
    predicted_lr_r = (probabilities_r > 0.5).astype(int)
    accuracy_lr_r = np.mean(predicted_lr_r == test_targets)

    print "Accuracy of Logistic Regression for L2 Regularization for lambda = "+str(lam)+ ": " + str(accuracy_lr_r)
    print(classification_report(test_targets, predicted_lr_r))
    print "Confusion Matrix:"
    print(confusion_matrix(test_targets, predicted_lr_r))

### Part J: Multinomial Naive Bayes and Linear Suppor Vector Machines with One vs. One and One vs. The Rest Classifiers ###
categories = ['comp.sys.ibm.pc.hardware','comp.sys.mac.hardware', 'misc.forsale', 'soc.religion.christian']

twenty_train = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=42)
X_train_counts = vectorizer.fit_transform(twenty_train.data)
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
X_train_reduced = svd.fit_transform(X_train_tfidf)
X_train = min_max_scaler.fit_transform(X_train_reduced)

twenty_test = fetch_20newsgroups(subset='test', categories=categories, shuffle=True, random_state=42)
X_test_counts = vectorizer.transform(twenty_test.data)
X_test_tfidf = tfidf_transformer.transform(X_test_counts)
X_test_reduced = svd.transform(X_test_tfidf)
X_test = min_max_scaler.transform(X_test_reduced)    
    
#Multinomial Naive Bayes
clf = MultinomialNB().fit(X_train,  twenty_train.target.tolist())
predicted_bayes = clf.predict(X_test)
accuracy_bayes = np.mean(predicted_bayes ==  twenty_test.target.tolist())

print "Accuracy of Multinomial Naive Bayes: " + str(accuracy_bayes)
print(classification_report( twenty_test.target.tolist(), predicted_bayes))
print "Confusion Matrix:"
print(confusion_matrix( twenty_test.target.tolist(), predicted_bayes))

#One vs. One Linear SVM
linear_SVM = OneVsOneClassifier(LinearSVC(dual=False, random_state=42)).fit(X_train, twenty_train.target.tolist())
predicted_svm = linear_SVM.predict(X_test)
accuracy_svm = np.mean(predicted_svm == twenty_test.target.tolist())


print "Accuracy of Linear SVM - One vs. One: " + str(accuracy_svm)
print(classification_report(twenty_test.target.tolist(), predicted_svm))
print "Confusion Matrix:"
print(confusion_matrix(twenty_test.target.tolist(), predicted_svm))

#One vs. The Rest Linear SVM
linear_SVM = OneVsRestClassifier(LinearSVC(dual=False, random_state=42)).fit(X_train, twenty_train.target.tolist())
predicted_svm = linear_SVM.predict(X_test)
accuracy_svm = np.mean(predicted_svm == twenty_test.target.tolist())


print "Accuracy of Linear SVM - One vs. The Rest: " + str(accuracy_svm)
print(classification_report(twenty_test.target.tolist(), predicted_svm))
print "Confusion Matrix:"
print(confusion_matrix(twenty_test.target.tolist(), predicted_svm))
