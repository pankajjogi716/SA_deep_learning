#!/usr/bin/env python
# coding: utf-8

# In[382]:


# from stackabuse.com text classification

import pandas as pd
import tensorflow as tf
import numpy as np
import re 
import nltk
from nltk.corpus import stopwords
import itertools
from appostophes import appos 
from autocorrect import Speller


from numpy import array
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers.core import Activation, Dropout, Dense
from keras.layers import Flatten, Conv1D, LSTM, Bidirectional
from keras.layers import GlobalMaxPooling1D, MaxPooling1D
from keras.layers.embeddings import Embedding
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam

from sklearn.preprocessing import LabelEncoder


# In[351]:




import pandas as pd
# Read TRAINING data
train = pd.read_csv('/home/pankaj/PROJECT/Datasets/pytree_sst/train2.txt', sep='\t', header=None, names=['truth', 'text'])
#train['truth'] = train['truth'].str.replace('__label__', '')
train['truth'] = train['truth'].astype(int).astype('category')
train.head()


# In[353]:


import pandas as pd
# Read TRAINING data
test = pd.read_csv('/home/pankaj/PROJECT/Datasets/pytree_sst/test2.txt', sep='\t', header=None, names=['truth', 'text'])
#test['truth'] = test['truth'].str.replace('__label__', '')
test['truth'] = test['truth'].astype(int).astype('category')
test.head()


# In[354]:


import pandas as pd
# Read TRAINING data
dev = pd.read_csv('/home/pankaj/PROJECT/Datasets/pytree_sst/dev2.txt', sep='\t', header=None, names=['truth', 'text'])
#dev['truth'] = dev['truth'].str.replace('__label__', '')
dev['truth'] = dev['truth'].astype(int).astype('category')
dev.head()


# In[355]:


# DATA PRE-PROCESSING
def preprocess_text(sen):
    # 1. Removing HTML tags
    sentence = remove_tags(sen)
   # print(sentence)
    # 2. Removing multiple spaces
    sentence = re.sub(r'\s+', ' ', sentence)
    #print(sentence) 
    # 3. Convert to lower case
    sentence= sentence.lower()
   # print(sentence)
    # 4. Apostrophes 
    sentence= remove_apostrophes(sentence)
    #print(sentence)
    # 5. Remove special characters
    sentence = remove_special_characters(sentence)
   # print(sentence)
    # 6. Single character removal
    sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', sentence)
    #print(sentence)
    # 7. Standardise words
    #sentence= standardise1(sentence)
    #print(sentence)
    #8. spell check
   # sentence= spell_check(sentence)
    #print(sentence)

    return sentence


# In[356]:


# 1. remove HTML tags 
TAG_RE = re.compile(r'<[^>]+>')
def remove_tags(text):
    return TAG_RE.sub('', text)


# In[357]:


# 4. Appostophes
def remove_apostrophes(text):
    #print(appos)
    words=text.split()  
    #print(words)
    reformed = [appos[word] if word in appos else word for word in words]
    reformed = " ".join(reformed)
    return reformed


# In[358]:


# 5. Remove special characters and digits
def remove_special_characters(text, remove_digits=True):
   pattern=r'[^a-zA-z\s]'
   text=re.sub(pattern,' ',text)
   return text


# In[359]:


# 7.1 Standardize words function-1
def standardise1(text):
    sent = ''.join(''.join(s)[:2] for _, s in itertools.groupby(text))
    return sent


# In[360]:


# 7.2 Standardize words function-2
def standardise2(text): 
    words=text.split()
    output_str= []
    for w in words:
        sc_removed = re.sub("[^a-zA-Z]", '', str(w))
        if len(sc_removed) > 1:
                output_str.append(sc_removed)
        joined = ' '.join(output_str)
    spell_corrected = re.sub(r'(.)\1+', r'\1\1', joined)
    return spell_corrected


# In[361]:


# spell check
spell = Speller(lang='en')
def spell_check(text):
    words=text.split()
    corrected=[]
    for w in words:
        corrected.append(spell(w))
    corrected=" ".join(str(x) for x in corrected)
    return corrected    


# In[362]:


#spell = Speller(lang='en')
#spell("SOOO")
#preprocess_text("Helloooooo moble   007 can't  sooo  A #hapPy!!!")
preprocess_text("At first it was very odd and pretty funny but as the movie progressed I didn't find the jokes or oddness funny anymore.Its a low-budget film (thats never ")
#remove_special_characters("Hello 007 I'm happy!!!")


# In[363]:


# pre-process every statement and store it 
print("pre-processing starts")
train_reviews = []
p=0
sentences = list(train['text'])
for sen in sentences:
    train_reviews.append(preprocess_text(sen))
    p+=1

print("pre-processing over")    
print(p)


# In[364]:


# pre-process every statement and store it 
print("pre-processing starts")
test_reviews = []
sentences = list(test['text'])
for sen in sentences:
    test_reviews.append(preprocess_text(sen))

print("pre-processing over")    


# In[365]:


# pre-process every statement and store it 
print("pre-processing starts")
dev_reviews = []
sentences = list(dev['text'])
for sen in sentences:
    dev_reviews.append(preprocess_text(sen))

print("pre-processing over")    


# In[366]:


train_label= train['truth']
test_label=test['truth']
dev_label=dev['truth']


# In[367]:


len(train_reviews)


# In[368]:


tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(train_reviews)

train_reviews = tokenizer.texts_to_sequences(train_reviews)
test_reviews = tokenizer.texts_to_sequences(test_reviews)
dev_reviews =tokenizer.texts_to_sequences(dev_reviews)


# In[369]:


# Adding 1 because of reserved 0 index
vocab_size = len(tokenizer.word_index) + 1
#print(type(X_train))
print("Number of training sample :",len(train_reviews))
print("Number of unique words :",vocab_size)
#print(X_train[0])
print((train_reviews[6]))


# In[370]:


maxlen = 200
print
train_reviews = pad_sequences(train_reviews, padding='post', maxlen=maxlen)
test_reviews  = pad_sequences(test_reviews, padding='post', maxlen=maxlen)
dev_reviews =pad_sequences(dev_reviews, padding='post', maxlen=maxlen)
print(type(train_reviews))
print(train_reviews.shape)


# In[371]:


print(type(train_reviews))
print(len(train_reviews))
print(train_reviews[6])
print(len(train_reviews[6]))


# In[372]:


from numpy import array
from numpy import asarray
from numpy import zeros
from keras.preprocessing.text import Tokenizer




w2v_embeddings_dictionary = dict()
glove_file = open('/home/pankaj/PROJECT/Datasets/glove.6B/glove.6B.300d.txt', encoding="utf8")
w2v_vocab_size=0

for line in glove_file:
    w2v_vocab_size+=1
    records = line.split()   # sandberger 1 2 3 ... 100
    word = records[0]        # word = sandberger
    vector_dimensions = asarray(records[1:], dtype='float32')      # vector_dimensions = 1 2 ..100
    w2v_embeddings_dictionary [word] = vector_dimensions               # add to dict word--> Embedding
glove_file.close()

print("NUmber of words :", w2v_vocab_size)

embedding_matrix = zeros((vocab_size, 300))
w2v_absent=0
for word, index in tokenizer.word_index.items():
   # print(word)
    w2v_embedding_vector = w2v_embeddings_dictionary.get(word)
    #if embedding_vector is not None:
     #   embedding_matrix[index] = embedding_vector
    
    if w2v_embedding_vector is  None:
        w2v_absent+=1
        #print(word)
        w2v_embedding_vector= np.random.uniform(0,1,300)
   
        embedding_matrix[index]= w2v_embedding_vector

print("missing words in word2vec: ",w2v_absent)

# In[269]:




#============== LSTM =================================
model = Sequential()
embedding_layer = Embedding(vocab_size, 300, weights=[embedding_matrix], input_length=maxlen , trainable=False)
model.add(embedding_layer)

model.add(LSTM(300))
model.add(Dropout(0.7))
model.add(Dense(2, activation='softmax'))



# In[375]:


#from keras.utils import to_categorical
#train_label = to_categorical(train_label)
#train_label.shape
...
# encode class values as integers
from keras.utils import np_utils
encoder = LabelEncoder()
encoder.fit(train_label)
en_train_label = encoder.transform(train_label)
# convert integers to dummy variables (i.e. one hot encoded)
dummy_train_label = np_utils.to_categorical(en_train_label)

encoder.fit(test_label)
en_test_label=encoder.transform(test_label)
dummy_test_label=np_utils.to_categorical(en_test_label)

encoder.fit(dev_label)
en_dev_label=encoder.transform(dev_label)
dummy_dev_label=np_utils.to_categorical(en_dev_label)


# In[376]:
learning_rate=0.05
adam=Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, amsgrad=False)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
print(model.summary())


# patient early stopping
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=200)

# In[377]:


history = model.fit(train_reviews, dummy_train_label, batch_size=100, validation_data=(dev_reviews,dummy_dev_label), epochs=5000, verbose=1,callbacks=[es])


# In[378]:


score = model.evaluate(test_reviews, dummy_test_label, verbose=1)


# In[379]:


print("Test Score:", score[0])
print("Test Accuracy:", score[1])
print(history.history["loss"])

#print(history.history["val_acc"])


# In[389]:


from matplotlib import pyplot as plt
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train','test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])

plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train','test'], loc='upper left')
plt.show()
plt.savefig('books_read.png')


# In[393]:


# ==== PREDICT POLARITY OF SINGLE REVIEW ====
#instance= "The movie was awesome"
instance=train['text'][57]
print(instance)
instance = tokenizer.texts_to_sequences(instance)

flat_list = []
for sublist in instance:
    for item in sublist:
        flat_list.append(item)

flat_list = [flat_list]

instance = pad_sequences(flat_list, padding='post', maxlen=maxlen)



# In[395]:


model.predict(instance)

