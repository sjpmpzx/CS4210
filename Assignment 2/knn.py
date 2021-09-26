#-------------------------------------------------------------------------
# AUTHOR: Zhenxiang Peng
# FILENAME: knn.py
# SPECIFICATION: Compute and output the LOO-CV error rate for 1NN
# FOR: CS 4210- Assignment #2
# TIME SPENT: 1 hr
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard vectors and arrays

#importing some Python libraries
from sklearn.neighbors import KNeighborsClassifier
import csv

db = []

#reading the data in a csv file
with open('binary_points.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    for i, row in enumerate(reader):
        if i > 0: #skipping the header
            db.append (row)


error_num = 0
#loop your data to allow each instance to be your test set
for i, instance in enumerate(db):

    #add the training features to the 2D array X removing the instance that will be used for testing in this iteration. For instance, X = [[1, 3], [2, 1,], ...]]. Convert each feature value to
    # float to avoid warning messages
    #--> add your Python code here
    # X =
    X = []
    for index, row in enumerate(db):
        if i != index:
            X.append([float(val) for val in row[:-1]])

    #transform the original training classes to numbers and add to the vector Y removing the instance that will be used for testing in this iteration. For instance, Y = [1, 2, ,...]. Convert each
    #  feature value to float to avoid warning messages
    #--> add your Python code here
    # Y =
    Y = []
    Y_dict = {
        "+" : 1,
        "-" : 2
    }
    for index, row in enumerate(db):
        if i != index:
            Y.append(float(Y_dict[row[-1]]))

    #store the test sample of this iteration in the vector testSample
    #--> add your Python code here
    #testSample =
    testSampe = list.copy(db[i])
    testSampe[:-1] = [float(val) for val in testSampe[:-1]]
    testSampe[-1] = float(Y_dict[instance[-1]])

    #fitting the knn to the data
    clf = KNeighborsClassifier(n_neighbors=1, p=2)
    clf = clf.fit(X, Y)

    #use your test sample in this iteration to make the class prediction. For instance:
    #class_predicted = clf.predict([[1, 2]])[0]
    #--> add your Python code here
    class_predicted = clf.predict([testSampe[:-1]])[0]

    #compare the prediction with the true label of the test instance to start calculating the error rate.
    #--> add your Python code here
    if class_predicted != testSampe[-1]:
        error_num = error_num + 1

#print the error rate
#--> add your Python code here
error_rate = error_num / len(db)
print("The error rate is: {}".format(error_rate))
