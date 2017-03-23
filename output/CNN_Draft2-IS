

get_ipython().magic(u'matplotlib inline')
import utils; reload(utils)
from utils import *


# ## Introduction

# In[ ]:

get_ipython().magic(u'matplotlib inline')
from __future__ import division,print_function
import os, json
from glob import glob
import numpy as np
import scipy
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix
np.set_printoptions(precision=4, linewidth=100)
from matplotlib import pyplot as plt
import utils; reload(utils)
from utils import plots, get_batches, plot_confusion_matrix, get_data


# In[ ]:

from numpy.random import random, permutation
from scipy import misc, ndimage
from scipy.ndimage.interpolation import zoom

import keras
from keras import backend as K
from keras.utils.data_utils import get_file
from keras.models import Sequential
from keras.layers import Input
from keras.layers.core import Flatten, Dense, Dropout, Lambda
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD, RMSprop
from keras.preprocessing import image


# ## Linear models in keras

# In[ ]:

x = random((30,2))
y = np.dot(x, [2., 3.]) + 1.


# In[ ]:

x[:5]


# In[ ]:

y[:5]


# In[ ]:

lm = Sequential([ Dense(1, input_shape=(2,)) ])
lm.compile(optimizer=SGD(lr=0.1), loss='mse')


# In[ ]:

lm.evaluate(x, y, verbose=0)


# In[ ]:

lm.fit(x, y, nb_epoch=5, batch_size=1)


# In[ ]:

lm.evaluate(x, y, verbose=0)


# In[ ]:

lm.get_weights()


# ## Train linear model on predictions

# ### Training the model

# In[ ]:


path = '/Users/IS/Desktop/training_data 2/raw_images/'
model_path = path + 'models/'
if not os.path.exists(model_path): os.mkdir(model_path)


# We will process as many images at a time as our graphics card allows. This is a case of trial and error to find the max batch size - the largest size that doesn't give an out of memory error.

# In[ ]:

batch_size=100
#batch_size=4


# We need to start with our VGG 16 model, since we'll be using its predictions and features.

# In[ ]:

from vgg16 import Vgg16
vgg = Vgg16()
model = vgg.model


# In[ ]:

# Use batch size of 1 since we're just doing preprocessing on the CPU
val_batches = get_batches(path+'valid', shuffle=False, batch_size=1)
batches = get_batches(path+'train', shuffle=False, batch_size=1)


# In[ ]:

import bcolz
def save_array(fname, arr): c=bcolz.carray(arr, rootdir=fname, mode='w'); c.flush()
def load_array(fname): return bcolz.open(fname)[:]


# We have provided a simple function that joins the arrays from all the batches - let's use this to grab the training and validation data:

# In[ ]:

val_data = get_data(path+'valid')


# In[ ]:

trn_data = get_data(path+'train')


# In[ ]:

trn_data.shape


# In[ ]:

save_array(model_path+ 'train_data.bc', trn_data)
save_array(model_path + 'valid_data.bc', val_data)


# We can load our training and validation data later without recalculating them:

# In[ ]:

trn_data = load_array(model_path+'train_data.bc')
val_data = load_array(model_path+'valid_data.bc')


# In[ ]:

val_data.shape


# Keras returns *classes* as a single column, so we convert to one hot encoding

# In[ ]:

def onehot(x): return np.array(OneHotEncoder().fit_transform(x.reshape(-1,1)).todense())


# In[ ]:

val_classes = val_batches.classes
trn_classes = batches.classes
val_labels = onehot(val_classes)
trn_labels = onehot(trn_classes)


# In[ ]:

trn_labels.shape


# In[ ]:

trn_classes[:4]


# In[ ]:

trn_labels[:4]


# ...and their 1,000 imagenet probabilties from VGG16--these will be the *features* for our linear model:

# In[ ]:

trn_features = model.predict(trn_data, batch_size=batch_size)
val_features = model.predict(val_data, batch_size=batch_size)


# In[ ]:

trn_features.shape


# In[ ]:

save_array(model_path+ 'train_lastlayer_features.bc', trn_features)
save_array(model_path + 'valid_lastlayer_features.bc', val_features)


# We can load our training and validation features later without recalculating them:

# In[ ]:

trn_features = load_array(model_path+'train_lastlayer_features.bc')
val_features = load_array(model_path+'valid_lastlayer_features.bc')


# Now we can define our linear model, just like we did earlier:

# In[ ]:

# 1000 inputs, since that's the saved features, and 2 outputs, for dog and cat
lm = Sequential([ Dense(2, activation='softmax', input_shape=(1000,)) ])
lm.compile(optimizer=RMSprop(lr=0.1), loss='categorical_crossentropy', metrics=['accuracy'])


# We're ready to fit the model!

# In[ ]:

batch_size=64


# In[ ]:

batch_size=4


# In[ ]:

lm.fit(trn_features, trn_labels, nb_epoch=3, batch_size=batch_size, 
       validation_data=(val_features, val_labels))


# In[ ]:

lm.summary()


# ### Viewing model prediction examples

# Calculate predictions on validation set, so we can find correct and incorrect examples:

# In[ ]:

# We want both the classes...
preds = lm.predict_classes(val_features, batch_size=batch_size)
# ...and the probabilities of being a cat
probs = lm.predict_proba(val_features, batch_size=batch_size)[:,0]
probs[:8]


# In[ ]:

preds[:8]


# Get the filenames for the validation set, so we can view images:

# In[ ]:

filenames = val_batches.filenames


# In[ ]:

# Number of images to view for each visualization task
n_view = 4


# Helper function to plot images by index in the validation set:

# In[ ]:

def plots_idx(idx, titles=None):
    plots([image.load_img(path + 'valid/' + filenames[i]) for i in idx], titles=titles)


# In[ ]:

#1. A few correct labels at random
correct = np.where(preds==val_labels[:,1])[0]
idx = permutation(correct)[:n_view]
plots_idx(idx, probs[idx])


# In[ ]:

#2. A few incorrect labels at random
incorrect = np.where(preds!=val_labels[:,1])[0]
idx = permutation(incorrect)[:n_view]
plots_idx(idx, probs[idx])


# In[ ]:

#3. The images we most confident were cats, and are actually cats
correct_cats = np.where((preds==0) & (preds==val_labels[:,1]))[0]
most_correct_cats = np.argsort(probs[correct_cats])[::-1][:n_view]
plots_idx(correct_cats[most_correct_cats], probs[correct_cats][most_correct_cats])


# In[ ]:

# as above, but dogs
correct_dogs = np.where((preds==1) & (preds==val_labels[:,1]))[0]
most_correct_dogs = np.argsort(probs[correct_dogs])[:n_view]
plots_idx(correct_dogs[most_correct_dogs], 1-probs[correct_dogs][most_correct_dogs])


# In[ ]:

#3. The images we were most confident were cats, but are actually dogs
incorrect_cats = np.where((preds==0) & (preds!=val_labels[:,1]))[0]
most_incorrect_cats = np.argsort(probs[incorrect_cats])[::-1][:n_view]
plots_idx(incorrect_cats[most_incorrect_cats], probs[incorrect_cats][most_incorrect_cats])


# In[ ]:

#3. The images we were most confident were dogs, but are actually cats
incorrect_dogs = np.where((preds==1) & (preds!=val_labels[:,1]))[0]
most_incorrect_dogs = np.argsort(probs[incorrect_dogs])[:n_view]
plots_idx(incorrect_dogs[most_incorrect_dogs], 1-probs[incorrect_dogs][most_incorrect_dogs])


# In[ ]:

#5. The most uncertain labels (ie those with probability closest to 0.5).
most_uncertain = np.argsort(np.abs(probs-0.5))
plots_idx(most_uncertain[:n_view], probs[most_uncertain])


# Perhaps the most common way to analyze the result of a classification model is to use a [confusion matrix](http://www.dataschool.io/simple-guide-to-confusion-matrix-terminology/). Scikit-learn has a convenient function we can use for this purpose:

# In[ ]:

cm = confusion_matrix(val_classes, preds)


# We can just print out the confusion matrix, or we can show a graphical view (which is mainly useful for dependents with a larger number of categories).

# In[ ]:

plot_confusion_matrix(cm, val_batches.class_indices)


# ### About activation functions

# In[ ]:

lm = Sequential([ Dense(2, activation='softmax', input_shape=(1000,)) ])


# In[ ]:

model.add(Dense(4096, activation='relu'))


# # Modifying the model

# ## Retrain last layer's linear model

# In[ ]:

vgg.model.summary()


# In[ ]:

model.pop()
for layer in model.layers: layer.trainable=False


# In[ ]:

model.add(Dense(2, activation='softmax'))


# In[ ]:

get_ipython().magic(u'pinfo2 vgg.finetune')


# In[ ]:

gen=image.ImageDataGenerator()
batches = gen.flow(trn_data, trn_labels, batch_size=batch_size, shuffle=True)
val_batches = gen.flow(val_data, val_labels, batch_size=batch_size, shuffle=False)


# In[ ]:

def fit_model(model, batches, val_batches, nb_epoch=1):
    model.fit_generator(batches, samples_per_epoch=batches.N, nb_epoch=nb_epoch, 
                        validation_data=val_batches, nb_val_samples=val_batches.N)


# In[ ]:

opt = RMSprop(lr=0.1)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])


# In[ ]:

fit_model(model, batches, val_batches, nb_epoch=2)


# In[ ]:

model.save_weights(model_path+'finetune1.h5')


# In[ ]:

model.load_weights(model_path+'finetune1.h5')


# In[ ]:

model.evaluate(val_data, val_labels)


# In[ ]:

preds = model.predict_classes(val_data, batch_size=batch_size)
probs = model.predict_proba(val_data, batch_size=batch_size)[:,0]
probs[:8]


# In[ ]:

cm = confusion_matrix(val_classes, preds)


# In[ ]:

plot_confusion_matrix(cm, {'cat':0, 'dog':1})


# ## Retraining more layers

# In[ ]:

# sympy let's us do symbolic differentiation (and much more!) in python
import sympy as sp
# we have to define our variables
x = sp.var('x')
# then we can request the derivative or any expression of that variable
pow(2*x,2).diff()


# ### Training multiple layers in Keras

# The code below will work on any model that contains dense layers; it's not just for this VGG model.
# 

# In[ ]:

layers = model.layers
# Get the index of the first dense layer...
first_dense_idx = [index for index,layer in enumerate(layers) if type(layer) is Dense][0]
# ...and set this and all subsequent layers to trainable
for layer in layers[first_dense_idx:]: layer.trainable=True


# In[ ]:

K.set_value(opt.lr, 0.01)
fit_model(model, batches, val_batches, 3)


# In[ ]:

model.save_weights(model_path+'finetune2.h5')


# In[ ]:

for layer in layers[12:]: layer.trainable=True
K.set_value(opt.lr, 0.001)


# In[ ]:

fit_model(model, batches, val_batches, 4)


# In[ ]:

model.save_weights(model_path+'finetune3.h5')


# In[ ]:

model.load_weights(model_path+'finetune2.h5')
model.evaluate_generator(get_batches('valid', gen, False, batch_size*2), val_batches.N)


# In[ ]:

