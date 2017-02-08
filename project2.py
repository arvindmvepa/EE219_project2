import warnings
import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction import text

warnings.filterwarnings("ignore")

#Part a
comp_categories = ['comp.graphics','comp.os.ms-windows.misc','comp.sys.ibm.pc.hardware','comp.sys.mac.hardware']
rec_categories = ['rec.autos','rec.motorcycles','rec.sport.baseball','rec.sport.hockey']

comp_train = fetch_20newsgroups(subset='train', categories=comp_categories, shuffle=True, random_state=42)
comp_test = fetch_20newsgroups(subset='test', categories=comp_categories, shuffle=True, random_state=42)
rec_train = fetch_20newsgroups(subset='train', categories=rec_categories, shuffle=True, random_state=42)
rec_test = fetch_20newsgroups(subset='test', categories=rec_categories, shuffle=True, random_state=42)

print("Training Set - Computer Technology: %s Recreation: %s" %(comp_train.filenames.shape[0],rec_train.filenames.shape[0]))
print("Test Set - Computer Technology: %s Recreation: %s" %(comp_test.filenames.shape[0],rec_test.filenames.shape[0]))

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
plt.show()

plt.bar(y_pos, np.array(test_counts), align='center', alpha=0.5)
plt.xticks(y_pos, objects)
plt.ylabel('Counts')
plt.title('Counts for Test Set')
plt.tight_layout()
plt.savefig('test_set_histogram.png')
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