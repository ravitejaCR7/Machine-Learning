import pandas as pd
import numpy as np


Train_data = pd.read_csv('sonar_train.csv',header = None)



def computeCovarianceMatrix(data):
    mat = data
    m = len(data)
    mean = (data.sum(axis=0))/m
    n = len(data)
    for i in range(n):
        mat[i] = mat[i] - mean
    mat = mat.transpose()
    mat = np.matmul(mat, mat.transpose())
    return mat
    
    
def eigen(matrix):
    eigenValues = np.linalg.eig(matrix)[0]
    m = len(eigenValues)
    idx = eigenValues.argsort()[m-6:m]  
    print(idx)
    eigenValues = eigenValues[idx]
    return eigenValues
    


m = len(Train_data.columns)-1
data = np.array(Train_data.iloc[:,0:m])
CovarianceMatrix = computeCovarianceMatrix(data)



eigenValues = eigen(CovarianceMatrix)
print('Top 6 eigen values are', eigenValues)
























