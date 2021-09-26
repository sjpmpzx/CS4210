#-------------------------------------------------------------------------
# AUTHOR: Zhenxiang Peng
# FILENAME: naive_bayes.py
# SPECIFICATION: Compute and ouput the classification result of each test instance with confidenve score >= 0.75 using Naïve Bayes
# FOR: CS 4210- Assignment #2
# TIME SPENT: 1 hr
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard vectors and arrays

#importing some Python libraries
from sklearn.naive_bayes import GaussianNB
import csv

#reading the training data
#--> add your Python code here
db = []
with open("weather_training.csv", "r") as csvfile:
    reader = csv.reader(csvfile)
    for i, row in enumerate(reader):
        if i > 0: #skipping the header
            db.append (row)

#transform the original training features to numbers and add to the 4D array X. For instance Sunny = 1, Overcast = 2, Rain = 3, so X = [[3, 1, 1, 2], [1, 3, 2, 2], ...]]
#--> add your Python code here
# X =
num_dict = {
    "Sunny" : 1,
    "Overcast" : 2,
    "Rain" : 3,
    "Hot" : 1,
    "Mild" : 2,
    "Cool" : 3,
    "High" : 1,
    "Normal" : 2,
    "Weak" : 1,
    "Strong" : 2,
    "Yes" : 1,
    "No" :2
}
X = []
for instance in db:
    X.append([num_dict[value] for value in instance[1:-1]])

#transform the original training classes to numbers and add to the vector Y. For instance Yes = 1, No = 2, so Y = [1, 1, 2, 2, ...]
#--> add your Python code here
# Y =
Y = [num_dict[instance[-1]] for instance in db]

#fitting the naive bayes to the data
clf = GaussianNB()
clf.fit(X, Y)

#reading the data in a csv file
#--> add your Python code here
db_test = []
db_test_num = []
with open("weather_test.csv", "r") as csvfile:
    reader = csv.reader(csvfile)
    for i, row in enumerate(reader):
        if i > 0: #skipping the header
            db_test.append (row)
for instance in db_test:
    db_test_num.append([num_dict[value] for value in instance[1:-1]])
    
#printing the header os the solution
print ("Day".ljust(15) + "Outlook".ljust(15) + "Temperature".ljust(15) + "Humidity".ljust(15) + "Wind".ljust(15) + "PlayTennis".ljust(15) + "Confidence".ljust(15))

#use your test samples to make probabilistic predictions.
#--> add your Python code here
#-->predicted = clf.predict_proba([[3, 1, 2, 1]])[0]
for i, instance in enumerate(db_test):
    predicted = clf.predict_proba([db_test_num[i]])[0]
    if predicted[0] >= 0.75 or predicted[1] >= 0.75:
        predicted_class = "Yes" if predicted[0] > 0.75 else "No"
        print(
            str(instance[0]).ljust(15) + str(instance[1]).ljust(15) + str(instance[2]).ljust(15) + str(instance[3]).ljust(15) + str(instance[4]).ljust(15) + \
            predicted_class.ljust(15) + str(predicted[0]).ljust(15))
