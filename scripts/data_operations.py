import pandas as pd
import numpy as np
import cv2
import csv
import tensorflow as tf
from sklearn.model_selection import KFold
from os import path
from progress.bar import Bar


def load_dataframe(path_csv):
    """
    Load a single dataframe from an csv file
    """
    return pd.read_csv(path_csv)


def slice_img(dataframe):
    """
    Return the dataframe with all the images reduced to 28x28 to eliminate the background
    """
    bar = Bar('Slicing', max=len(dataframe.values))
    data = []
    for image in dataframe.values:
        re = np.reshape(image, (48, 48))
        sub_matrix = re[9:37, 9:37]
        data.append(sub_matrix.flatten())
        bar.next()
    reduced = pd.DataFrame(data, columns=range(0, 28**2))
    bar.finish()
    return reduced


def normalize(dataframe):
    """
    Create the normalized version of the train_smpl
    The image's pixels are converted in a range [0-255]
    """
    normalized = []
    bar = Bar('Normalizing\t', max=len(dataframe.values))
    for pixels in dataframe.values:
        (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(pixels)
        new_row = []
        for pixel in pixels:
            pixel = int((pixel - minVal) * (255 / (maxVal - minVal)))
            new_row.append(pixel)
        normalized.append(new_row)
        bar.next()
    bar.finish()
    return pd.DataFrame(normalized, columns=dataframe.columns)


def save_dataframe_csv(dataframe, file_path):
    """
    Helper method to save a dataframe in the correct format
    """
    if file_path != "":
        with open(file_path, 'w', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(dataframe.columns)
            bar = Bar('Saving csv\t', max=len(dataframe.values))
            for el in dataframe.values:
                csv_writer.writerow(el)
                bar.next()
            bar.finish()


def make_kfold_tf_dataset(sample, labels, k_splits):
    def generator():
        for train_index, test_index in KFold(k_splits).split(sample):
            sample_train, sample_test = sample[train_index], sample[test_index]
            label_train, label_test = labels[train_index], labels[test_index]
            yield sample_train, label_train, sample_test, label_test
    return tf.data.Dataset.from_generator(generator, (tf.float64, tf.float64, tf.float64, tf.float64))


if __name__ == '__main__':
    df_train = load_dataframe('../data/')
