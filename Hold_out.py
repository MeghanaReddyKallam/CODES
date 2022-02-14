import numpy as np
import argparse
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn.metrics import (precision_recall_curve, average_precision_score,
                             roc_curve, roc_auc_score)
                             
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
import sklearn
import matplotlib
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pandas
import math
import operator

import shap

def find_optimal_cutoff_closest_to_perfect(fpr, tpr, threshold):
  """ Tries to find the best threshold to select on ROC.

  """
  distance = float("inf")
  threshold_index = -1
  for i in range(len(fpr)):
    d1 = math.sqrt((fpr[i] - 0) * (fpr[i] - 0) + (tpr[i] - 1) * (tpr[i] - 1))
    if d1 < distance:
      distance = d1
      threshold_index = i

  return threshold_index

def analysis(figurenum, 
             train_y, 
             train_X, 
             test_y, 
             test_X, 
             plot, 
             problem_name):
  """
    Performs the analysis.
    
    Arguments:
    figurenum - Used to increment the figurenum between calls so we don't plot
                different data to the same figure.
    train_y - The training labels.
    train_X - The training data.
    test_y - The test labels.
    test_X - The test data.
    plot - If true, plots the ROC and Precision/Recall curves.
    problem_name - Used to label the plot. 
    test_src_ips - The source ips for the netflows of test set
    test_dest_ips - The dest ips for the netflows of the test set
  """

  clf = RandomForestClassifier()
  clf.fit(train_X, train_y)

  train_scores = clf.predict_proba(train_X)[:,1]
  test_scores = clf.predict_proba(test_X)[:,1]
  train_auc = roc_auc_score(train_y, train_scores)
  test_auc = roc_auc_score(test_y, test_scores)
  train_average_precision = average_precision_score(train_y, train_scores)
  test_average_precision = average_precision_score(test_y, test_scores)
  fpr, tpr, thresholds = roc_curve(test_y, test_scores)
  threshold_index = find_optimal_cutoff_closest_to_perfect(fpr, tpr, thresholds)

  print( "AUC of ROC on train", train_auc )
  print( "AUC of ROC on test", test_auc )
  print( "Average Precision on train", train_average_precision )
  print( "Average Precision on test", test_average_precision )
  print( "Optimal point", fpr[threshold_index], tpr[threshold_index])
  if plot:
    precision, recall, thresholds = precision_recall_curve(
                                      test_y, clf.predict_proba(test_X)[:,1])

    figurenum += 1
    f1 = plt.figure(figurenum)
    plt.step(recall, precision, color='b', alpha=0.2, where='post')
    plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    str_average_precision = "{0:.3f}".format(test_average_precision)
    plt.title('Precision-Recall curve of {}: AUC={}'.format(
      problem_name, str_average_precision))
    f1.show()

    figurenum += 1
    f2 = plt.figure(figurenum)
    plt.plot([0,1],[0,1], 'k--')
    plt.plot(fpr, tpr)
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    str_auc_roc = "{0:.3f}".format(test_auc)
    plt.title('ROC curve of {}: AUC={}'.format(problem_name, str_auc_roc))
    plt.plot([fpr[threshold_index]], [tpr[threshold_index]], marker='o', 
              markersize=10, color="red")
    f2.show()

    return figurenum


def main():

  # Process command line arguments
  parser = argparse.ArgumentParser()
  parser.add_argument('--inputfile1', type=str, required=True,
                      help="The file with the features and labels.")
  parser.add_argument('--inputfile2', type=str, required=True,
                      help="The file with the features and labels.")
  parser.add_argument('--problem_name', type=str,
                      default="SpecifyProblemName",
                      help="The problem name used in plots")
  parser.add_argument('--plot', action='store_true')
  parser.add_argument('--subset', type=str,
    help="Comma-separated list of features to include") 
                      
  FLAGS = parser.parse_args()

  # Open a file with the extracted features
  with open(FLAGS.inputfile1, "r") as infile1:
    data = np.loadtxt(infile, delimiter=",")
   with open(FLAGS.inputfile2, "r") as infile2:
    test = np.loadtxt(infile, delimiter=",")


    y1 = data[:, 0] #labels are in the first column
    x1 = data[:, 1:] #features are in the rest of the columns
    y2 = test[:, 0] #labels are in the first column
    x2 = test[:, 1:]

   
#   x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=0.2)
    from sklearn.linear_model import LinearRegression
    lreg=LinearRegression()
#   regression is applied and the library is used
    lreg.fit(x1,y1)
    lreg.predict(x2)
    lreg.socre(x2,y2)
    
    figurenum = 0
    figurenum = analysis(figurenum, y1,x1,y2,x2 FLAGS.plot, 
                          FLAGS.problem_name)#, srcIps[i:], destIps[i:])

    if FLAGS.plot:
      input()

main()
