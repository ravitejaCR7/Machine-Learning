import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import time
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
        
            
def accur(data, rounds, alpha, hypo):#Accuracy Calculation for a given data, Number of Rounds, Alpha and 
    accuracy = 0                     #Hypothesis space
    length = len(data)
    for i in range(length):
        p = 0
        row = data[i, :]
        y = row[0]
        for r in range(rounds):
            k = output(hypo[r], row)
            p = p + alpha[r] * k
        if p >= 0:
            if y == 1:
                accuracy += 1
        else:
            if y == -1:
                accuracy += 1
    accuracy = accuracy/length * 100
    return accuracy

def AdaBoost(rounds, model, data):
    test = Test_data
    weightedError = 1
    weight = [1/rows]*rows #Initial weight matrix
    alpha = []
    HSpace = [] #Total hypothesis space
    predictions = []
    Train_accuracies = []
    Test_accuracies = []
    eps = []    #Weighted errors
    for r in range(rounds):
        t1 = time.time()
        print('round',r+1)
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
                if w > weightedError or w > 0.51:#Optimization
                    break
            if w < weightedError:
                weightedError = w
                bTree = mod #bTree indicates it is a best tree
                bestP = lst
        HSpace.append(bTree)
        eps.append(weightedError)
        t = 1/2 * math.log((1-weightedError)/weightedError)
        alpha.append(t)
        #Train and Test Accuracy Calculation
        acc1 = accur(data, r+1, alpha, HSpace)
        acc2 = accur(test, r+1, alpha, HSpace)
        Train_accuracies.append(acc1)
        Test_accuracies.append(acc2)
        #Weight Updation
        sum1 = 0
        predictions.append(bestP)
        for i in range(rows):
            prediction = bestP[i]
            actual = data[i, 0]
            #print(prediction, actual)
            weight[i] = (weight[i] * (np.exp(-1 * prediction * actual * t)))/(2*np.sqrt(weightedError * (1-weightedError)))
            sum1 += weight[i]
        #print(weight)
        #print(alpha)
        print(weightedError) 
        t2 = time.time()
        print('round', r+1, 'end')
        print(t2-t1, 'secs')                      
    return eps, alpha, HSpace, predictions, Train_accuracies, Test_accuracies
                                        
                    
                
        
        
def Hypothesis():
    hypo = []
    lst = [1, -1]
    for i in range(1, attrs):#attribute iteration
        for j in range(1, attrs):
            for k in range(1, attrs):#5 different cases for 3 attribute splits
                for a in lst:
                    for b in lst:
                        for c in lst:
                            for d in lst:
                                tree = {}                #case 1 LLL
                                tree[i] = {}
                                tree[i][1] = a
                                tree1 = {}
                                tree1[j] = {}
                                tree1[j][1] = b
                                tree2 = {}
                                tree2[k] = {}
                                tree2[k][0] = c
                                tree2[k][1] = d
                                tree1[j][0] = tree2
                                tree[i][0] = tree1
                                hypo.append(tree)
                                
                                tree = {}                #case 2 LLR
                                tree[i] = {}
                                tree[i][1] = a
                                tree1 = {}
                                tree1[j] = {}
                                tree1[j][0] = b
                                tree2 = {}
                                tree2[k] = {}
                                tree2[k][0] = c
                                tree2[k][1] = d
                                tree1[j][1] = tree2
                                tree[i][0] = tree1
                                hypo.append(tree)
                                
                                tree = {}                #case 3 
                                tree[i] = {}
                                tree1 = {}
                                tree1[j] = {}
                                tree1[j][0] = a
                                tree1[j][1] = b
                                tree2 = {}
                                tree2[k] = {}
                                tree2[k][0] = c
                                tree2[k][1] = d
                                tree[i][0] = tree1
                                tree[i][1] = tree2
                                hypo.append(tree)
                                
                                tree = {}                #case 4
                                tree[i] = {}
                                tree[i][0] = a
                                tree1 = {}
                                tree1[j] = {}
                                tree1[j][1] = b
                                tree2 = {}
                                tree2[k] = {}
                                tree2[k][0] = c
                                tree2[k][1] = d  
                                tree1[j][0] = tree2
                                tree[i][1] = tree1
                                hypo.append(tree)
                                
                                tree = {}                #case 5
                                tree[i] = {}
                                tree[i][0] = a
                                tree1 = {}
                                tree1[j] = {}
                                tree1[j][0] = b
                                tree2 = {}
                                tree2[k] = {}
                                tree2[k][0] = c
                                tree2[k][1] = d
                                tree1[j][1] = tree2
                                tree[i][1] = tree1
                                hypo.append(tree)  
    return hypo
                

data = Train_data
rounds = 10
print('starting time')
t1 = time.time()
Thypo = Hypothesis()   #Thypo mean Total hypothesis space
print('end')
t2 = time.time()
print(t2-t1, 'secs')
print('Generated Hypothesis spaces')

epsilon, alpha, hypo, predictions, Train_accuracies, Test_accuracies = AdaBoost(rounds, Thypo, data) #method for getting Hypothesis space and Alpha values
t2 = time.time()
print(t2-t1, 'secs')

print('Hypothesis Spaces are', hypo)

print('Alpha values are')
print(alpha)


print('Epsilon values are')
print(epsilon)


print("Accuracies on the training data set are", Train_accuracies)


print("Accuracies on the test data set are", Test_accuracies)

t2 = time.time()
print(t2-t1, 'secs')


#plot reference : https://www.geeksforgeeks.org/graph-plotting-in-python-set-1/
x1 = []
for i in range(rounds):
    x1.append(i+1)

# plotting the Training accuracies
plt.plot(x1, Train_accuracies, label = "Train Accuracies") 
  
# plotting the Testing accuracies
plt.plot(x1, Test_accuracies, label = "Test Accuracies") 
  
# naming the x axis 
plt.xlabel('Numbe of Rounds') 
# naming the y axis 
plt.ylabel('Accuracy') 
# giving a title to my graph 
plt.title('Boosting') 
  
# show a legend on the plot 
plt.legend() 
  
# function to show the plot 
plt.show() 


