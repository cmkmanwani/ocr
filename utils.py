import numpy as np
import imageio as io
import imutils
import tensorflow as tf
from keras.backend import ctc_decode, get_value
from PIL import Image
from param import alphabet, rev_alphabet, max_string_len


def encode(label):
    len_label = len(label)
    ret = np.ones(max_string_len) * len(alphabet)
    for idx, char in zip(range(len_label), label):
        ret[idx] = alphabet[char]
    return ret.astype(int)


def decode(encoded_label):
    len_alphabet = len(rev_alphabet)
    ret = []
    for encoded_char in encoded_label:
        if encoded_char == len_alphabet:  # CTC Blank
            ret.append("")
        else:
            ret.append(rev_alphabet[encoded_char])
    return "".join(ret)


def debug_decode(encoded_label):
    len_alphabet = len(rev_alphabet)
    ret = []
    for encoded_char in encoded_label:
        if encoded_char == len_alphabet:  # CTC Blank
            ret.append("_")
        else:
            ret.append(rev_alphabet[encoded_char])
    return "".join(ret)


def rem_duplicates(encoded_label_dupl):
    len_label = len(encoded_label_dupl)
    len_alphabet = len(alphabet)
    encoded_label = [encoded_label_dupl[0]]
    for idx in range(1, len_label):
        if encoded_label_dupl[idx] == len_alphabet or encoded_label_dupl[idx] != encoded_label_dupl[idx-1]:
            encoded_label.append(encoded_label_dupl[idx])
    return encoded_label


def activation_to_label(y_pred):
    pred_labels = np.argmax(y_pred[0, :, :], axis=1).tolist()

    return decode(rem_duplicates(pred_labels)), debug_decode(pred_labels)


def get_image(img_filename):
    img = io.imread(img_filename)
    img = np.array(img)
    return np.expand_dims(img, axis=-1)


def rgb2gray(rgb):
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray


def process(img):
    gray_img = rgb2gray(img)
    rotated_img = imutils.rotate_bound(gray_img, 90)
    final_img = Image.fromarray(rotated_img).resize((32, 100))
    return np.array(final_img, dtype='uint8')
def word_error(filename):
    with open(filename , 'r') as fl:
        lines = fl.read().split('\n')
    correct = 0
    total = 0
    for line in lines:
        words = line.split(' ')
        if(words[0] == words[1]):
            correct+=1
        total+=1
    error = 1-(correct/total)
    return error

def character_error(filename):
    with open(filename , 'r') as fl:
        lines = fl.read().split('\n')
    incorrect = 0
    total = 0
    for line in lines:
        words = line.split(' ')
        m = len(words[0])
        n = len(words[1])
        i = 0
        j = 0
        while(i<m or j<n):
            if i>=m :
                incorrect+=n-j
                total+=n-j
                break
            elif j>=n:
                total+=m-i
                incorrect+=m-i
                break
            else:
                if(words[0][i] != words[1][j]):
                    incorrect+=1
                total+=1
            i+=1
            j+=1
    error = incorrect/total
    return error