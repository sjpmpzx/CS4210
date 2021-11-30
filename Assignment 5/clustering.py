#-------------------------------------------------------------------------
# AUTHOR: Zhenxiang Peng
# FILENAME: clustering.py
# SPECIFICATION: description of the program
# FOR: CS 4210- Assignment #5
# TIME SPENT: 1 hr
#-----------------------------------------------------------*/

#importing some Python libraries
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn import metrics

df = pd.read_csv('training_data.csv', sep=',', header=None) #reading the data by using Pandas library

#assign your training data to X_training feature matrix
X_training = df

#run kmeans testing different k values from 2 until 20 clusters
     #Use:  kmeans = KMeans(n_clusters=k, random_state=0)
     #      kmeans.fit(X_training)
     #--> add your Python code
max_sil_score = 0
max_k = 0
k_values = []
sil_scores = []

for k in range(2, 21):
     kmeans = KMeans(n_clusters=k, random_state=0)
     kmeans.fit(X_training)

     #for each k, calculate the silhouette_coefficient by using: silhouette_score(X_training, kmeans.labels_)
     #find which k maximizes the silhouette_coefficient
     #--> add your Python code here
     sil_score = silhouette_score(X_training, kmeans.labels_)
     k_values.append(k)
     sil_scores.append(sil_score)
     if sil_score > max_sil_score:
          max_sil_score = sil_score
          max_k = k

#plot the value of the silhouette_coefficient for each k value of kmeans so that we can see the best k
#--> add your Python code here
print(f"The best k value is: {max_k}")
plt.plot(k_values, sil_scores)
plt.xlabel("K value")
plt.ylabel("Silhouette coefficient")
plt.show()

#reading the validation data (clusters) by using Pandas library
#--> add your Python code here
df = pd.read_csv('testing_data.csv', sep=',', header=None)

#assign your data labels to vector labels (you might need to reshape the row vector to a column vector)
# do this: np.array(df.values).reshape(1,<number of samples>)[0]
#--> add your Python code here
labels = np.array(df.values).reshape(1, len(df))[0]

#Calculate and print the Homogeneity of this kmeans clustering
print("K-Means Homogeneity Score = " + metrics.homogeneity_score(labels, kmeans.labels_).__str__())
#--> add your Python code here

#rung agglomerative clustering now by using the best value o k calculated before by kmeans
#Do it:
#agg = AgglomerativeClustering(n_clusters=<best k value>, linkage='ward')
#agg.fit(X_training)
agg = AgglomerativeClustering(n_clusters=max_k, linkage='ward')
agg.fit(X_training)

#Calculate and print the Homogeneity of this agglomerative clustering
print("Agglomerative Clustering Homogeneity Score = " + metrics.homogeneity_score(labels, agg.labels_).__str__())
