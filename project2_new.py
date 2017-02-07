from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction import text
import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt

#Part a
comp_categories = [['comp.graphics'],['comp.os.ms-windows.misc'],['comp.sys.ibm.pc.hardware'],['comp.sys.mac.hardware']]
rec_categories = [['rec.autos'],['rec.motorcycles'],['rec.sport.baseball'],['rec.sport.hockey']]

comp_train = {}
comp_test = {}
for category in comp_categories:

    comp_train["{0}".format(category)] = fetch_20newsgroups(subset='train', categories=category, shuffle=True, random_state=42)
    comp_test["{0}".format(category)] = fetch_20newsgroups(subset='test', categories=category, shuffle=True, random_state=42)

rec_train = {}
rec_test = {}
for category in rec_categories:

    rec_train["{0}".format(category)] = fetch_20newsgroups(subset='train', categories=category, shuffle=True, random_state=42)
    rec_test["{0}".format(category)] = fetch_20newsgroups(subset='test', categories=category, shuffle=True, random_state=42)

comp_train_total = comp_test_total = 0
comp_train_counts = []
comp_test_counts = []
for category in comp_categories:
    comp_train_total += comp_train["{0}".format(category)].filenames.shape[0]
    comp_train_counts.append(comp_train["{0}".format(category)].filenames.shape[0])
    comp_test_total += comp_test["{0}".format(category)].filenames.shape[0]
    comp_test_counts.append(comp_test["{0}".format(category)].filenames.shape[0])

rec_train_total = rec_test_total = 0
rec_train_counts = []
rec_test_counts = []
for category in rec_categories:
    rec_train_total += rec_train["{0}".format(category)].filenames.shape[0]
    rec_train_counts.append(rec_train["{0}".format(category)].filenames.shape[0])
    rec_test_total += rec_test["{0}".format(category)].filenames.shape[0]
    rec_test_counts.append(rec_test["{0}".format(category)].filenames.shape[0])

print("Training Set - Computer Technology: %s Recreation: %s" %(comp_train_total,rec_train_total))
print("Test Set - Computer Technology: %s Recreation: %s" %(comp_test_total,rec_test_total))

objects = ('graphics', 'windows', 'ibm', 'mac', 'autos', 'cycles','baseball','hockey')
y_pos = np.arange(len(objects))

train_counts = comp_train_counts + rec_train_counts
test_counts = comp_test_counts + rec_test_counts

plt.bar(y_pos, train_counts, align='center', alpha=0.5)
plt.xticks(y_pos, objects)
plt.ylabel('Counts')
plt.title('Counts for Training Set')
plt.tight_layout() 
plt.show()

plt.bar(y_pos, test_counts, align='center', alpha=0.5)
plt.xticks(y_pos, objects)
plt.ylabel('Counts')
plt.title('Counts for Test Set')
plt.tight_layout()  
plt.show()

#Part b
categories = [ 'comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey']

twenty_train = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=42)
twenty_test = fetch_20newsgroups(subset='test', categories=categories, shuffle=True, random_state=42)
stop_words = text.ENGLISH_STOP_WORDS

#vectorizer_train = TfidfVectorizer()
#vectors_train = vectorizer_train.fit_transform(twenty_train.data)

#vectorizer_test = TfidfVectorizer()
#vectors_test = vectorizer_test.fit_transform(twenty_test.data)'''