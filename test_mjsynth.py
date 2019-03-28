import sys
import os
import numpy as np
from utils import get_image, activation_to_label
from crnn_model import get_model


def main(args):
    model = get_model(training=False)
    model.summary()

    try:
        model.load_weights(args[2])
        print("....Previous weight data...")
    except:
        print("Weights not found, Aborting")
        exit()

    data_dir = args[0]
    annotation_file = args[1]
    outfile = open('outfile', 'w')

    with open(os.path.join(data_dir, annotation_file), 'r') as f:
        filenames = [line.split(' ')[0][2:] for line in f]

    iters = 0
    max_iters = 100
    for img_name in filenames:
        if iters == max_iters:
            break
        iters += 1
        img = get_image(os.path.join(data_dir, img_name))
        img = np.expand_dims(img, axis=0)

        y_pred = model.predict(img)

        label_true = img_name.split('_')[1]
        label_pred, label_unprocessed = activation_to_label(y_pred)

        outfile.write(label_true + ' ' + label_pred + ' ' + label_unprocessed + '\n')

    outfile.close()


if __name__ == "__main__":
    if(len(sys.argv) != 4):
        print("Invalid command line inputs. Give Inputs - <name.py> <data_folder> <annotation_test> <weights>")
        exit()
    else:
        main(sys.argv[1:])
