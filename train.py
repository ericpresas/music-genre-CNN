from dataset import *
from music_tagger_cnn import *
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.engine import Model, Input
from keras.layers.normalization import BatchNormalization
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
from keras import backend as K
K.set_image_dim_ordering('th')
from matplotlib import pyplot
from scipy import misc
from keras.layers.convolutional import ZeroPadding2D
from skimage.transform import resize
from load_dataset import *
from music_tagger_cnn import *
from sklearn.preprocessing import LabelEncoder
from load_dataset import *
from dataset import *
from matplotlib import pyplot as plt
from keras.callbacks import Callback
from plot_confusion_matrix import *
from sklearn.metrics import confusion_matrix



def plot_metrics(history):

    print(history.history.keys())

    fig = plt.figure(1)

    # summarize history for accuracy

    plt.subplot(211)
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')

    # summarize history for loss

    plt.subplot(212)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')

    fig.savefig('metrics.png', dpi=fig.dpi)

# Load dataset

data = dataset('/imatge/epresas/music_genre/data/genres', 10, 100)
data.create()
print(data.X_train.shape)

# Build the VGG model
input_tensor = Input(shape=(1, 18, 119))
model =MusicTaggerCNN(input_tensor=input_tensor, include_top=False, weights=None)
last_layer = model.get_layer('pool3').output
out = Flatten()(last_layer)
out = Dense(128, activation='relu', name='fc2')(out)
out = Dropout(0.5)(out)
out = Dense(data.n_classes, activation='softmax', name='fc3')(out)
model = Model(input=model.input, output=out)
sgd = SGD(lr=0.01, momentum=0, decay=0.002, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
history = model.fit(data.X_train, data.labels_train,
                              validation_data=(data.X_val, data.labels_val), nb_epoch=100,
                              batch_size=5)
plot_metrics(history)
predict_labels = model.predict(data.X_val)
predictions = []
orig_predictions = []
for prediction, orig_prediction in zip(predict_labels, data.labels_val):
    ind1 = np.argmax(prediction)
    ind2 = np.argmax(orig_prediction)
    predictions.append(ind1)
    orig_predictions.append(ind2)
cnf_matrix = confusion_matrix(orig_predictions, predictions)
plot_confusion_matrix(cnf_matrix, data.names)