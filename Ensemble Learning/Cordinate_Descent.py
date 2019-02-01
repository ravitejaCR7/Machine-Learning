import pandas as pd
import numpy as np
import math
Train_data = pd.read_csv('heart_train.data',header = None)
Test_data = pd.read_csv('heart_test.data',header = None)
Train_data.loc[Train_data[0] == 0, 0] = -1
Test_data.loc[Test_data[0] == 0, 0] = -1



attrs = len(Train_data.columns)  #No of Attributes

rows = len(Train_data)   #No of Training Data Points


Train_data = np.array(Train_data.iloc[:,:])
Test_data = np.array(Test_data.iloc[:,:])


def output(tree, lst):
    while(True):
        for key, value in tree.items():
            if value == 1 or value == -1:
                return value
            else:
                k = lst[key]
                tree = tree[key][k]
            if tree == 1 or tree == -1:
                return tree

def Hypothesis():
    hypo = []
    lst = [1, -1]
    for attr in range(1, attrs):#attribute iteration
        for i in lst:
            for j in lst: # 4 cases for 2 possible choices of labelling 1, -1
                tree = {}
                tree[attr] = {}
                tree[attr][0] = i
                tree[attr][1] = j
                hypo.append(tree)
    return hypo
                
              
def Coordinate_Descent(alpha, data, labels):
    length = len(alpha)
    it = 0
    diff = 1
    while(diff > 0.01):
        diff = 0
        for i in range(length):#Updating alphas in Round Robin way
            n = 0
            d = 0
            for r in range(rows):
                k = data[r, 0]
                p = 0
                for j in range(length):
                    if i != j:
                        p = p + alpha[j] * labels[j][r]
                p = np.exp(-1 * k * p)
                if k == labels[i][r]:
                    n += p
                else:
                    d += p
            Palpha = alpha[i]
            alpha[i] = 1/2 * math.log(n/d)
            diff += abs(alpha[i] - Palpha)
        #print(alpha)
        #print('diff',diff)
        it += 1
    return alpha
        

hypo = Hypothesis()   #hypo mean Total hypothesis space
space = len(hypo)
alpha = [.5] * space


labels = []
for tree in hypo:
    lst = []
    for i in range(rows):
        row = Train_data[i, :]
        #print(tree)
        #print(row)
        k = output(tree, row)
        lst.append(k)
    labels.append(lst)
        
#print(labels)     
alpha = Coordinate_Descent(alpha, Train_data, labels) #method for getting Hypothesis space and Alpha values



#Exponential Loss on Training set
expLoss = 0
for r in range(rows):
    y = Train_data[r, 0]
    k = 0
    for h in range(space):
        k += alpha[h] * labels[h][r]
    k = np.exp(-1 * y * k)
    expLoss += k
print('Exponential Loss on Training data set is', expLoss)

#accuracy calculation

length = len(Test_data)

accuracy = 0
for i in range(length):
    p = 0
    row = Test_data[i, :]
    for r in range(space):
        k = output(hypo[r], row)
        p = p + alpha[r] * k
    if p >= 0:
        if row[0] == 1:
            accuracy += 1
    else:
        if row[0] == -1:
            accuracy += 1

accuracy = accuracy/length * 100
print("Accuracy on the test data set is", accuracy)
        
print('The values of alpha are')     
print(alpha)