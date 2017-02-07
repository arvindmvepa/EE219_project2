
# coding: utf-8

# In[33]:

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction import text
import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt

#Part a
comp_graphics_train = fetch_20newsgroups(subset='train', categories=['comp.graphics'], shuffle=True, random_state=42)
comp_graphics_test = fetch_20newsgroups(subset='test', categories=['comp.graphics'], shuffle=True, random_state=42)
comp_os_ms_windows_misc_train = fetch_20newsgroups(subset='train', categories=['comp.os.ms-windows.misc'], shuffle=True, random_state=42)
comp_os_ms_windows_misc_test = fetch_20newsgroups(subset='test', categories=['comp.os.ms-windows.misc'], shuffle=True, random_state=42)
comp_sys_ibm_pc_hardware_train = fetch_20newsgroups(subset='train', categories=['comp.sys.ibm.pc.hardware'], shuffle=True, random_state=42)
comp_sys_ibm_pc_hardware_test = fetch_20newsgroups(subset='test', categories=['comp.sys.ibm.pc.hardware'], shuffle=True, random_state=42)
comp_sys_mac_hardware_train = fetch_20newsgroups(subset='train', categories=['comp.sys.mac.hardware'], shuffle=True, random_state=42)
comp_sys_mac_hardware_test = fetch_20newsgroups(subset='test', categories=['comp.sys.mac.hardware'], shuffle=True, random_state=42)


rec_autos_train = fetch_20newsgroups(subset='train', categories=['rec.autos'], shuffle=True, random_state=42)
rec_autos_test = fetch_20newsgroups(subset='test', categories=['rec.autos'], shuffle=True, random_state=42)
rec_motorcycles_train = fetch_20newsgroups(subset='train', categories=['rec.motorcycles'], shuffle=True, random_state=42)
rec_motorcycles_test = fetch_20newsgroups(subset='test', categories=['rec.motorcycles'], shuffle=True, random_state=42)
rec_sport_baseball_train = fetch_20newsgroups(subset='train', categories=['rec.sport.baseball'], shuffle=True, random_state=42)
rec_sport_baseball_test = fetch_20newsgroups(subset='test', categories=['rec.sport.baseball'], shuffle=True, random_state=42)
rec_sport_hockey_train = fetch_20newsgroups(subset='train', categories=['rec.sport.hockey'], shuffle=True, random_state=42)
rec_sport_hockey_test = fetch_20newsgroups(subset='test', categories=['rec.sport.hockey'], shuffle=True, random_state=42)

comp_train_total = comp_graphics_train.filenames.shape[0]+comp_os_ms_windows_misc_train.filenames.shape[0]+comp_sys_ibm_pc_hardware_train.filenames.shape[0]+comp_sys_mac_hardware_train.filenames.shape[0]
rec_train_total = rec_autos_train.filenames.shape[0]+rec_motorcycles_train.filenames.shape[0]+rec_sport_baseball_train.filenames.shape[0]+rec_sport_hockey_train.filenames.shape[0]

comp_test_total = comp_graphics_test.filenames.shape[0]+comp_os_ms_windows_misc_test.filenames.shape[0]+comp_sys_ibm_pc_hardware_test.filenames.shape[0]+comp_sys_mac_hardware_test.filenames.shape[0]
rec_test_total = rec_autos_test.filenames.shape[0]+rec_motorcycles_test.filenames.shape[0]+rec_sport_baseball_test.filenames.shape[0]+rec_sport_hockey_test.filenames.shape[0]

print("Training Set - Computer Technology: %s Recreation: %s" %(comp_train_total,rec_train_total))
print("Test Set - Computer Technology: %s Recreation: %s" %(comp_test_total,rec_test_total))

objects = ('graphics', 'windows', 'ibm', 'mac', 'autos', 'cycles','baseball','hockey')
y_pos = np.arange(len(objects))
train_counts = [comp_graphics_train.filenames.shape[0],comp_os_ms_windows_misc_train.filenames.shape[0],comp_sys_ibm_pc_hardware_train.filenames.shape[0],comp_sys_mac_hardware_train.filenames.shape[0],rec_autos_train.filenames.shape[0],rec_motorcycles_train.filenames.shape[0],rec_sport_baseball_train.filenames.shape[0],rec_sport_hockey_train.filenames.shape[0]]
 
plt.bar(y_pos, train_counts, align='center', alpha=0.5)
plt.xticks(y_pos, objects)
plt.ylabel('Counts')
plt.title('Counts for Training Set')
plt.tight_layout() 
plt.show()

test_counts = [comp_graphics_test.filenames.shape[0],comp_os_ms_windows_misc_test.filenames.shape[0],comp_sys_ibm_pc_hardware_test.filenames.shape[0],comp_sys_mac_hardware_test.filenames.shape[0],rec_autos_test.filenames.shape[0],rec_motorcycles_test.filenames.shape[0],rec_sport_baseball_test.filenames.shape[0],rec_sport_hockey_test.filenames.shape[0]]
 
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
#vectors_test = vectorizer_test.fit_transform(twenty_test.data)

