#-------------------------------------------------------------------------
# AUTHOR: Zhenxiang Peng
# FILENAME: svm.py
# SPECIFICATION: Read the file optdigits.tra to build multiple SVM classifiers and print the highest accuracy
# FOR: CS 4210- Assignment #3
# TIME SPENT: 30 min
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard vectors and arrays

#importing some Python libraries
from sklearn import svm
import csv

dbTest = []
X_training = []
Y_training = []
c = [1, 5, 10, 100]
degree = [1, 2, 3]
kernel = ["linear", "poly", "rbf"]
decision_function_shape = ["ovo", "ovr"]
highestAccuracy = 0

#reading the data in a csv file
with open('optdigits.tra', 'r') as trainingFile:
  reader = csv.reader(trainingFile)
  for i, test_sample in enumerate(reader):
      X_training.append(test_sample[:-1])
      Y_training.append(test_sample[-1])

#reading the data in a csv file
with open('optdigits.tes', 'r') as testingFile:
  reader = csv.reader(testingFile)
  for i, test_sample in enumerate(reader):
      dbTest.append (test_sample)

#created 4 nested for loops that will iterate through the values of c, degree, kernel, and decision_function_shape
#--> add your Python code here
highest_accuracy = 0

for c_val in c: #iterates over c
    for d_val in degree: #iterates over degree
        for k_val in kernel: #iterates kernel
           for dfs_val in decision_function_shape: #iterates over decision_function_shape

                #Create an SVM classifier that will test all combinations of c, degree, kernel, and decision_function_shape as hyperparameters. For instance svm.SVC(c=1)
                clf = svm.SVC(C=c_val, degree=d_val, kernel=k_val, decision_function_shape=dfs_val)

                #Fit SVM to the training data
                clf.fit(X_training, Y_training)

                #make the classifier prediction for each test sample and start computing its accuracy
                #--> add your Python code here
                correct_count = 0

                for test_sample in dbTest:
                    class_predicted = clf.predict([test_sample[:-1]])
                    if class_predicted[0] == test_sample[-1]:
                        correct_count += 1

                accuracy = correct_count / len(dbTest)

                #check if the calculated accuracy is higher than the previously one calculated. If so, update update the highest accuracy and print it together with the SVM hyperparameters
                #Example: "Highest SVM accuracy so far: 0.92, Parameters: a=1, degree=2, kernel= poly, decision_function_shape = 'ovo'"
                #--> add your Python code here
                if accuracy > highest_accuracy:
                    highest_accuracy = accuracy
                    print(f"Highest SVM accuracy so far: {accuracy}, Parameters: c={c_val}, degree={d_val}, kernel={k_val}, decision_function_shape={dfs_val}")










