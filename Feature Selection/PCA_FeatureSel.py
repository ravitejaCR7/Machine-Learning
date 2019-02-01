import pandas as pd
import numpy as np
import math


Train_data = pd.read_csv('sonar_train.csv',header = None)
Test_data = np.array(pd.read_csv('sonar_test.csv',header = None))

def calculateProbability(x, mean, stdev):
    exponent = math.exp(-(math.pow(x-mean,2)/(2*math.pow(stdev,2))))
    return (1 / (math.sqrt(2*math.pi) * stdev)) * exponent

        
#Accuracy on Test Data
def accuracy(features):
    n = len(Test_data)
    count = 0
    for i in range(n):
        r = Test_data[i,:]
        p1 = 1
        p2 = 1
        for j in features:
            mean = data_summary[1][j]['mean']
            std = data_summary[1][j]['sd']
            p1 = p1 * calculateProbability(r[j], mean, std)
            mean = data_summary[2][j]['mean']
            std = data_summary[2][j]['sd']
            p2 = p2 * calculateProbability(r[j], mean, std)
        if p1 > p2:
            if r[m] == 1:
                count += 1
        else:
            if r[m] == 2:
                count += 1
            
    test_accuracy = count/n * 100
    return test_accuracy 


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
    
    
def eigen(matrix): #Method for calculating Eigen Values and Eigen Vectors
    eigenValues, eigenVectors = np.linalg.eig(matrix) 
    idx = eigenValues.argsort()[::-1][0:10]     #[0:6] selects the top 6 eigen values
    eigenVectors = eigenVectors[:, idx]
    return eigenVectors
    
    

def ProbDis(vector):   #Define Probability distribution
    pd = []
    m, n = vector.shape
    for i in range(m):
        v = 0
        for j in range(n):
            v = v+(vector[i, j]**2)
        pd.append(v/n)
    return pd




m = len(Train_data.columns)-1
data = {}
data[1] = np.array(Train_data.loc[Train_data[m] == 1])
data[2] = np.array(Train_data.loc[Train_data[m] == 2])

data_summary = {}
for key, value in data.items():
    data_k = value
    data_summary[key] = {}
    for i in range(m):
        tree = {}
        tree['mean'] = np.mean(data_k[:, i])
        tree['sd'] = np.std(data_k[:, i])
        data_summary[key][i] = tree

trainData = np.array(Train_data.iloc[:,0:m])
CovarianceMatrix = computeCovarianceMatrix(trainData)
eigenVectors = eigen(CovarianceMatrix)



k = 10
s = 20
for i in range(k):
    eg = eigenVectors[:,0:i+1] #Selects the top K eigen Vectors
    pd = ProbDis(eg) #For Calculating Probability Distribution
    for j in range(s):
        sum_acc = 0
        for it in range(100):  #Running over through 100 iterations to determine average accuracy
            f = np.random.choice(m, j+1, p=pd) #Sampling with repetition with a given probability
            features = np.unique(f) #Selecting the unique features
            test_acc = accuracy(features)
            sum_acc += test_acc #Summing up to calculate average accuracy
        avg_testAccuracy = sum_acc/100 #Average accuracy
        print('Error on test data for value of k =',i+1,'and s = ',j+1,'is',100-avg_testAccuracy)
    print('\n')
            
            
            

            



















