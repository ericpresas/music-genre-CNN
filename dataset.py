from load_dataset import *
from keras.utils import np_utils
from python_speech_features.python_speech_features import mfcc
from python_speech_features.python_speech_features import logfbank

class dataset:
    """
        Class Dataset:
            - Parameters:
                - X_train, X_val => train and val samples
                - labels_train, labels_val => train and val labels
                - n_classes => number of classes
                - n_representations => number of exemplar representations per class
                - total => total number of exemplars
                - out_dim => dimensions of data (Theano 'th' or Tensorflow 'tf')

    """

    def __init__(self, path, n_classes, n_representations):
        self.X_train = []
        self.labels_train = []
        self.X_val = []
        self.labels_val = []
        self.labels = []
        self.root_path = path  # Ha de ser desde la carpeta on hi han totes les classes
        self.n_classes = n_classes
        self.n_representations = n_representations  # number of representations for each class, the representations will be splited in 3 (train, val, test)
        self.total = n_representations * n_classes

    def create(self):
        names = load_names_from_folders(self.root_path)
        filelist = []  # list of lists
        labels = []  # list of labels related to data list
        # Creation list of files for each class stored in a list
        print('Loading images...')

        for i in range(self.n_classes):
            filelist.append(make_filelist(self.root_path + "/" + names[i], self.n_representations))
            labels.append(names[i])
        # Store images in a
        data = []
        i = 0
        for data_class in filelist:
            printProgressBar(i + 1, len(filelist), prefix='Progress:', suffix='Complete', length=50)
            i += 1
            class_data = []
            for file in data_class:
                class_data.append(load_audio(file))
            data.append(class_data)

        n_samples_class = self.n_representations
        n_samples_train = round(n_samples_class * 0.6)
        n_samples_test = n_samples_class - n_samples_train
        print('')
        print('Creating data vectors...')
        count_progress = 0
        X_train = []
        X_val = []
        for class_sample, label in zip(data, labels):
            count = 0
            printProgressBar(count_progress+1, len(labels), prefix='Progress:', suffix='Complete', length=50)
            count_progress += 1

            for sample in class_sample:
                x = []
                mfcc_feat = mfcc(sample, 44100)
                longi = len(mfcc_feat)
                punt = round(longi / 2)
                tmp_feat = mfcc_feat[punt - 2 * 44100:punt + 2 * 44100, :]
                if(mfcc_feat[punt - 2*44100:punt + 2*44100, :].shape[0] > 560):
                    x.append(tmp_feat[0:560, :])
                else:
                    x.append(mfcc_feat[punt - 2*44100:punt + 2*44100, :])
                x = np.array(x)
                x = x.transpose(1, 2, 0)
                if count < n_samples_train:
                    X_train.append(x)
                    self.labels_train.append(label)
                    count += 1
                else:
                    X_val.append(x)
                    self.labels_val.append(label)
        # Convert data values to float32
        self.X_train = np.array([image.astype('float32') for image in X_train])
        self.X_val = np.array([image.astype('float32') for image in X_val])
        # convert labels strings to integers
        train_sort = np.sort(np.unique(self.labels_train))
        val_sort = np.sort(np.unique(self.labels_val))
        lab_sort = np.sort(np.unique(labels))
        y_train = np.searchsorted(train_sort, self.labels_train)
        y_val = np.searchsorted(val_sort, self.labels_val)
        lab = np.searchsorted(lab_sort, labels)
        self.X_train = self.X_train.transpose(0, 3, 2, 1)
        self.X_val = self.X_val.transpose(0, 3, 2, 1)

        # convert integers to dummy variables (i.e. one hot encoded)
        self.labels_train = np_utils.to_categorical(y_train, self.n_classes)
        self.labels_val = np_utils.to_categorical(y_val, self.n_classes)
        self.labels = np_utils.to_categorical(lab, self.n_classes)

        print('')
        print("Number of classes: ", self.n_classes)
        print('n_samples train: ', len(self.X_train), n_samples_train)
        print('n_samples test: ', len(self.X_val), n_samples_test)



