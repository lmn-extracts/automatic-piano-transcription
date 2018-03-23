import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

def print_score(true_pos, false_pos, false_neg):
    precision = true_pos * 1. / (true_pos + false_pos)
    recall = true_pos * 1. / (true_pos + false_neg)
    F1 = 2. * precision * recall / (precision + recall)
    print 'Precision: {}'.format(precision)
    print 'Recall: {}'.format(recall)
    print 'F1 Score: {}'.format(F1)
    return

def main():
    true_pos = 0
    true_neg = 0
    false_pos = 0
    false_neg = 0

    actuals = []
    # notes = []

    # with open('eval-972.csv','r') as f:
    #     for line in f:
    #         frame_no, actual, detected = line.strip().split(',')
    #         actuals.append(actual)
    #         notes.append(detected)
    #         if actual == detected:
    #             if actual == '0':
    #                 true_neg += 1
    #             else:
    #                 true_pos += 1
    #         elif actual == '0':
    #             false_pos += 1
    #         elif detected == '0':
    #             false_neg += 1
    #         else:
    #             false_neg += 1
    # print true_pos
    # print false_pos
    # print false_neg

    # classes = ['C5', 'C#5', 'D5', 'D#5', 'E5', 'F5', 'F#5', 'G5', 'G#5', 'A5', 'A#5', 'B5', 'C6']
    # cm = confusion_matrix(actuals, notes, labels = classes)
    # plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    # plt.colorbar()
    # tick_marks = np.arange(len(classes))
    # plt.xticks(tick_marks, classes, rotation=45)
    # plt.yticks(tick_marks, classes)
    # plt.show()



    # precision = true_pos * 1. / (true_pos + false_pos)
    # recall = true_pos * 1. / (true_pos + false_neg)
    # F1 = 2. * precision * recall / (precision + recall)
    # print 'Precision: {}'.format(precision)
    # print 'Recall: {}'.format(recall)
    # print 'F1 Score: {}'.format(F1)
    print_score(1562,91,84)
    return

if __name__ == '__main__':
    main()