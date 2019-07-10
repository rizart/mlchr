"""This module contains various utilities for the mlchr package."""
import os
import sys
import glob
import json
import numpy as np
from PIL import Image
from mlchr.utils import image

from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels


def get_confusion_matrix_dict(y_test, y_pred):
    """
    :param y_test: Test target values.
    :param y_pred: Predicted target values.
    :return: Confusion matrix dictionary.
    """

    # create confusion matrix dictionary
    conf_matrix_j = {}
    classes = unique_labels(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred, labels=classes)

    # normalize (e.g. get percentage of accuracy and not support)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    for i in range(len(classes)):
        conf_matrix_j[classes[i]] = {}
        conf_matrix_j[classes[i]]['count'] = int(sum(cm[i]))
        for j in range(len(classes)):
            if cm_normalized[i][j] != 0:
                conf_matrix_j[classes[i]][classes[j]] = [
                    int(cm[i][j]), cm_normalized[i][j]
                ]
    return conf_matrix_j


def extract_json_stats(y_test, y_pred, filename, metrics_map, folds=5):
    """
    :param y_test: Test target values.
    :param y_pred: Predicted target values.
    :param filename: Output json filename.
    :param metrics_map: Metrics map.
    :return: None
    """

    accuracy_score_j = metrics_map['accuracy_score_j']
    balanced_accuracy_score_j = metrics_map['balanced_accuracy_score_j']
    precision_score_j = metrics_map['precision_score_j']
    recall_score_j = metrics_map['recall_score_j']
    f_measure_j = metrics_map['f_measure_j']
    conf_matrix_j = get_confusion_matrix_dict(y_test, y_pred)

    classification_report_json = {
        'avg_accuracy_score': accuracy_score_j,
        'avg_balanced_accuracy_score': balanced_accuracy_score_j,
        'avg_precision_score (micro)': precision_score_j,
        'avg_recall_score (micro)': recall_score_j,
        'avg_f_measure_score (micro)': f_measure_j,
        'confusion_matrix(fold {0}/{0})'.format(folds): conf_matrix_j
    }

    with open(filename, 'w') as fp:
        json.dump(classification_report_json, fp, indent=4)


def read_from_folder(args, n_values=50):
    """Loads images from folder with subfolders as image classes.
    :param args: Argparse arguments map.
    :param n_values: Number of images to load per class.
    :return: A list of OCRImage class instances.
    """
    images = []
    img_id = 0
    basedir = str(args['input_train'])
    class_dirs = os.listdir(basedir)
    # load images from base directory
    for class_dir in class_dirs:
        image_files = glob.glob(os.path.join(basedir, class_dir, "*"))

        # test case
        if args['test']:
            image_files = image_files[0:n_values]

        for image_file in image_files:
            img = image.OCRImage(pil_image=Image.open(image_file),
                                 img_id=img_id,
                                 img_class=class_dir,
                                 img_hex=image_file[:-4][-4:])
            images.append(img)
            img_id += 1

    return images


def print_flush(msg):
    """Prints message to stdout instantly"""
    print(msg, end='')
    sys.stdout.flush()
