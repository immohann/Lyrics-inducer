

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
import tensorflow.keras.utils as ku 
import numpy as np 
import pickle
from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
import numpy
import os
import tensorflow as tf
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression




lyric=open('dataset.txt').read()

#OUTPUT EXAMPLE
# I\'ll give you all my love\nIf you treat me right, baby, I\'ll give you everything\nTalk to me,
# I need to hear you need me like I need you\nFall for me, I wanna know you feel how I feel for you,


# ### 3.2 Preprocessing 
# 

# #### Converting to lowercase 
# Since the words carries capital letters as well, changing it to lowercase.
# Also keeoing dataset as per line would be more apt as it'll learn from the sentences formed to yield better performance

# In[19]:


#lowercase and split the datset
corpus=lyric.lower().split('\n')
for i in range(40,60):
    print(corpus[i])


# #### Tokenizing 
# Tokenizer creates the token for each line present in the corpus and measuring the number of the tokens created

# In[20]:


tokenizer = Tokenizer()
tokenizer.fit_on_texts(corpus)
total_words = len(tokenizer.word_index) + 1
print('\n\n WORDs: ', total_words)


# #### Creating Sequences
# create input sequences using list of tokens. Here we are generating tokens for each word and its preceeding word for each line. i.e: 
# 
# _come closer, i'll give you_    
# 
# will become: 
# 
# [[come],[come,closer],[come,closer,i'll],[come,closer,i'll,give]] 

# In[ ]:


input_sequences = []
for line in corpus:
	token_list = tokenizer.texts_to_sequences([line])[0]
	for i in range(1, len(token_list)):
		n_gram_sequence = token_list[:i+1]
		input_sequences.append(n_gram_sequence)


# In[22]:

for i in range(20):
    print(input_sequences[i])


# #### Padding
# Since the length of the arrays formed is different hence padding of length of the longest array is required in order to make the array length uniform.
# 
# We can either do pre-padding or post-padding

# In[23]:


max_sequence_len = max([len(x) for x in input_sequences])
input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))
print('max seq: ',max_sequence_len)


# ### 4. Building Model
# As we've formed our data into array form, now we can build a model to process the same.
# 

# In[24]:
print('\n\n Training model\n\n')

model = Sequential()
model.add(Embedding(total_words, 160, input_length=max_sequence_len-1))
model.add(Bidirectional(LSTM(200, return_sequences = True)))
model.add(Dropout(0.2))
model.add(LSTM(100))
model.add(Dense(total_words/2, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
model.add(Dense(total_words, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())


# In[ ]:


# create predictors and label
predictors, label = input_sequences[:,:-1],input_sequences[:,-1]

label = ku.to_categorical(label, num_classes=total_words)


# ### 5. Training the Model
# Since in this case we don't have any validation set, we don't have to worry about the overfitting of model.

# In[26]:


history = model.fit(predictors, label, epochs=100, verbose=1)



# ### 6. Analysing the results
# By plotting the Training accuraccy and Training loss using matplot-lib, we can infer the model performance

# In[27]:



acc = history.history['accuracy']
loss = history.history['loss']

epochs = range(len(acc))



model.save('my_model.h5')
print('Model saved!')
#model_json = model.to_json()
#with open("model.json", "w") as json_file:
 #   json_file.write(model_json)
# serialize weights to HDF5
#model.save_weights("model.h5")
#print("Saved model to disk")



# In[40]:


# ### 10. Conclusion 
# Hence, we can say that we sucessfully built the Lyrics-Inducer using NLP and LSTM, the model is more like predicting the NEXT WORD according to previous set of words, hence not much accurate for predicting longer sentences. 
# And there are plenty of applications possible using the LSTM and NLP, do try.

# In[ ]:



next_words = 100
seed_text = 'Lemme love you '
  


# In[29]:
for _ in range(next_words):
	token_list = tokenizer.texts_to_sequences([seed_text])[0]
	token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
	predicted = model.predict_classes(token_list, verbose=2)
	output_word = ""
	for word, index in tokenizer.word_index.items():
		if index == predicted:
			output_word = word
			break
	seed_text += " " + output_word
print(seed_text)





