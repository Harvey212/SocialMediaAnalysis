import numpy as np
import csv
from numpy.linalg import eig
###############################################

import matplotlib.pyplot as plt
import random
###########################################3
##############################################3
arr=[]
#node=[]
with open('adjacent_matrix-1-1.csv', newline='') as csvfile:
    rows = csv.reader(csvfile)
#    print(rows)
    x=0
    for row in rows:
        if x!=0:            
            temp=[]
            for k in range(1,len(row)):
                temp.append(int(eval(row[k])))
            #print(temp)
            arr.append(temp)
#        else:
#            node=row[1:len(row)]
        x+=1
#####################################
#print(len(arr))
#print(node)
#print(arr)
A=np.array(arr)
#print(A.shape)
#########################################################



#################################################################################
D=np.zeros((A.shape[0], A.shape[1]))
for i in range(A.shape[0]):
    r=arr[i]
    s=0
    for j in range(len(r)):
        if r[j] ==1:
            s+=1
#    print(r)
    D[i,i] = s

L=D-A

#print(D)
#print(A)
#print(L.shape)
#print(L)
#####################################3
eigenValues, eigenVectors=eig(L)
#print('E-value:', w)

#print('E-vector', v)
idx = eigenValues.argsort()#[::-1]   
eigenValues = eigenValues[idx]
eigenVectors = eigenVectors[:,idx]

#print(eigenValues)
#print(eigenVectors)
#print(eigenVectors.shape)

topv=eigenValues[0:3]
topvec=eigenVectors[0:3]
#print(topvec.shape)
############################################
#count=0
with open("eigen.csv", 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    for i in range(topvec.shape[0]):
        csvwriter.writerow(topvec[i])
########################################################
#########################################
class kMeans():
    def __init__(self, k=3, max_iter=1000, tol=0.0001):
        self.k= k 
        self.max_iter = max_iter
        self.tol = tol
    
    def euclidean(self, v1, v2):
        return np.sqrt(np.sum((v1-v2)**2))
    
    def fit(self, X_train):
        self.X_train = X_train

        idx = np.random.randint(len(X_train), size=self.k)
        self.centroids = X_train[idx,:]
        self.clusters = np.zeros(len(X_train))
        
        for i in range(self.max_iter):
            self.update_clusters()
            early_stop = self.update_centroids()

            if early_stop==True:
                print(f'Early stopping occured after {i} iterations')
                break
    # Calculate which cluster each point belongs to
    def update_clusters(self):
        for row_idx, train_row in enumerate(self.X_train):
            dist = []
            for i in range(self.k):
                dist.append(self.euclidean(train_row, self.centroids[i]))
            
            self.clusters[row_idx] = np.argmin(np.array(dist))
        
    # Calculate center of each cluster
    def update_centroids(self):
        # Loop over k clusters
        new_centroids = np.copy(self.centroids)
        for i in range(self.k):
            new_centroids[i] = np.mean(self.X_train[self.clusters==i], axis=0)
        
        # Check for convergence
        if np.linalg.norm(new_centroids-self.centroids)>self.tol:
            self.centroids = new_centroids
            return False
        else:
            self.centroids = new_centroids
            return True

    # Make predictions
    def predict(self, X_test):
        predictions = np.zeros(len(X_test))
        for row_idx, test_row in enumerate(X_test):
            dist = []
            for i in range(self.k):
                dist.append(self.euclidean(test_row, self.centroids[i]))
            predictions[row_idx] = np.argmin(np.array(dist))
        return predictions


topvec2=topvec.transpose()
# Apply our k-Means algorithm
model = kMeans(k=3)
model.fit(topvec2)
preds = model.predict(topvec2)

print(preds)

with open("result.csv", 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(preds)
#####################################################






###################################
#fig = plt.figure(figsize=(12, 12))
#ax = fig.add_subplot(projection='3d')

#sequence_containing_x_vals = []
#sequence_containing_y_vals = []
#sequence_containing_z_vals = []

#print(topvec2.shape)

#for m in range(topvec2.shape[0]):
#    vv=topvec2[m]
#    sequence_containing_x_vals.append(vv[0])
#    sequence_containing_y_vals.append(vv[1])
#    sequence_containing_z_vals.append(vv[2])

#ax.scatter(sequence_containing_x_vals, sequence_containing_y_vals, sequence_containing_z_vals)
#plt.show()
#####################################################


