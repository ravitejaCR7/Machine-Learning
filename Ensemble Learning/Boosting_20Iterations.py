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
            #print('key', key)
            #print('value', value)
            if value == 1 or value == -1:
                return value
            else:
                k = lst[key]
                tree = tree[key][k]
            if tree == 1 or tree == -1:
                return tree

def AdaBoost(rounds, model, data):
    weightedError = 1
    weight = [1/rows]*rows #Initial weight matrix
    alpha = []
    HSpace = [] #Total hypothesis space
    predictions = []
    
    for r in range(rounds):
        weightedError = 1
        for mod in model:
            w = 0
            lst = []
            for i in range(rows):
                row = data[i, :]
                k = output(mod, row)
                lst.append(k)
                if row[0] != k:
                    w = w + weight[i]
            if w < weightedError:
                weightedError = w
                bTree = mod #bTree indicates it is a best tree
                bestP = lst
        HSpace.append(bTree)
        t = 1/2 * math.log((1-weightedError)/weightedError)
        alpha.append(t)
        #Weight Updation
        sum1 = 0
        predictions.append(bestP)
        for i in range(rows):
            prediction = bestP[i]
            actual = data[i, 0]
            #print(prediction, actual)
            weight[i] = (weight[i] * (np.exp(-1 * prediction * actual * t)))/(2*np.sqrt(weightedError * (1-weightedError)))
            sum1 += weight[i]
        #print("sum", sum1)
        #print(weight)
        #print(alpha)
        print('weightedError on round',r+1,'is',weightedError)
                                        
    return alpha, HSpace, predictions
                                        
                    
                
        
        
def Hypothesis():
    hypo = []
    lst = [1, -1]
    for attr in range(1, attrs):#attribute iteration
        for i in lst:
            for j in lst: # 4 cases for 2 possible choices of labelling
                tree = {}
                tree[attr] = {}
                tree[attr][0] = i
                tree[attr][1] = j
                hypo.append(tree)
    #print(hypo)
    print('length', len(hypo))
    return hypo
                

rounds = 20
Thypo = Hypothesis()   #Thypo mean Total hypothesis space
alpha, hypo, predictions = AdaBoost(rounds, Thypo, Train_data) #method for getting Hypothesis space and Alpha values

#accuracy calculation
length = len(Test_data)

accuracy = 0
for i in range(length):
    p = 0
    row = Test_data[i, :]
    for r in range(rounds):
        k = output(hypo[r], row)
        p = p + alpha[r] * k
    if p > 0:
        if Test_data[i, 0] == 1:
            accuracy += 1
    else:
        if Test_data[i, 0] == -1:
            accuracy += 1


accuracy = accuracy/length * 100
print('Accuracy on the test data set after' ,rounds,'rounds is', accuracy)
        
print('The values of alpha are')     
print(alpha)