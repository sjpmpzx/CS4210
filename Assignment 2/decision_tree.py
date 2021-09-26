#-------------------------------------------------------------------------
# AUTHOR: Zhenxiang Peng
# FILENAME: decision_tree.py
# SPECIFICATION: Train, test and output the performance of the models created by using training sets on the test set
# FOR: CS 4210- Assignment #2
# TIME SPENT: 1.5 hr
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard vectors and arrays

#importing some Python libraries
from sklearn import tree
import csv

dataSets = ['contact_lens_training_1.csv', 'contact_lens_training_2.csv', 'contact_lens_training_3.csv']

for ds in dataSets:

    dbTraining = []
    X = []
    Y = []

    #reading the training data in a csv file
    with open(ds, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for i, row in enumerate(reader):
            if i > 0: #skipping the header
                dbTraining.append (row)

    #transform the original training features to numbers and add to the 4D array X. For instance Young = 1, Prepresbyopic = 2, Presbyopic = 3, so X = [[1, 1, 1, 1], [2, 2, 2, 2], ...]]
    #--> add your Python code here
    # X =
    num_dict = {
        "Young" : 1,
        "Prepresbyopic" : 2,
        "Presbyopic" : 3,
        "Myope" : 1,
        "Hypermetrope" : 2,
        "Yes" : 1,
        "No" : 2,
        "Normal" : 1,
        "Reduced" : 2
    }

    for line in dbTraining:
        tmp = []
        for value in line[:-1]:
            tmp.append(num_dict[value])
        X.append(tmp)

    #transform the original training classes to numbers and add to the vector Y. For instance Yes = 1, No = 2, so Y = [1, 1, 2, 2, ...]
    #--> add your Python code here
    # Y =
    for line in dbTraining:
        Y.append(num_dict[line[4]])

    lowest_accuracy = 1
    #loop your training and test tasks 10 times here
    for i in range (10):

        #fitting the decision tree to the data setting max_depth=3
        clf = tree.DecisionTreeClassifier(criterion = 'entropy', max_depth=3)
        clf = clf.fit(X, Y)

        #read the test data and add this data to dbTest
        #--> add your Python code here
        # dbTest =
        dbTest = []
        with open("contact_lens_test.csv", 'r') as csvfile:
            reader = csv.reader(csvfile)
            for i, row in enumerate(reader):
                if i > 0: #skipping the header
                    dbTest.append (row)

            correct_prediction = 0
            dbTest_size = len(dbTest)

        for data in dbTest:
            #transform the features of the test instances to numbers following the same strategy done during training, and then use the decision tree to make the class prediction. For instance:
            #class_predicted = clf.predict([[3, 1, 2, 1]])[0]           -> [0] is used to get an integer as the predicted class label so that you can compare it with the true label
            #--> add your Python code here
            data = [num_dict[value] for value in data]         
            class_predicted = clf.predict([data[:-1]])[0]

            #compare the prediction with the true label (located at data[4]) of the test instance to start calculating the accuracy.
            #--> add your Python code here          
            if class_predicted == data[4]:
                correct_prediction = correct_prediction + 1

        #find the lowest accuracy of this model during the 10 runs (training and test set)
        #--> add your Python code here
        accuracy = correct_prediction / dbTest_size
        if accuracy < lowest_accuracy:
            lowest_accuracy = accuracy

    #print the lowest accuracy of this model during the 10 runs (training and test set) and save it.
    #your output should be something like that: final accuracy when training on contact_lens_training_1.csv: 0.2
    #--> add your Python code here
    print("final accuracy when training on {}: {}".format(ds, lowest_accuracy))
