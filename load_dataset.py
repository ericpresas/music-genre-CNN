import numpy as np
from skimage import io
from random import sample
from scipy import misc
import os, sys
from scipy.spatial import distance
import scipy.io.wavfile as wav
def printProgressBar(iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = '\r')
    # Print New Line on Complete
    if iteration == total:
        print()


def load_names_from_folders(path):
    names = os.listdir(path)
    return names

def make_filelist(path, n_representations):
    #make a list of files for one class with n_representations
    filelist = []
    files = os.listdir(path)
    count = 0
    for file in files:
        if file.endswith('.wav') and count < n_representations:
            filelist.append(path + "/" + file)
            count += 1
    return filelist

def load_labels(file_path):
    # Load txt file
    # fnam = 'people.txt'
    print('Loading Labels......')
    fnames = np.genfromtxt(file_path, dtype=None, delimiter='   ')

    # Labels and number of faces each label of entire set
    labels = []
    n_faces = []
    for names in fnames:
        temp_labels = names.split(b'\t')
        if (len(temp_labels) == 2):
            labels.append(temp_labels[0].decode())
            n_faces.append(temp_labels[1].decode())
    return labels, n_faces




def load_audio(path):
    # Open audiofile from path
    samplerate, samples = wav.read(path)
    return samples
