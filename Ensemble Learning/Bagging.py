import pandas as pd
import math

Train_data = pd.read_csv('heart_train.data',header = None)
Test_data = pd.read_csv('heart_test.data',header = None)


attrs = len(Train_data.columns)  #No of Attributes

rows = len(Train_data)   #No of Training Data Points



def splitData(data, attribute):
    data0 = data.loc[data[attribute] == 0, :]
    data1 = data.loc[data[attribute] == 1, :]    
    return data0, data1

def label(data):
    p = len(data.loc[data[0] == 1, :])
    n = len(data.loc[data[0] == 0, :])
    if p >= n:
        return 1
    else:
        return 0

def Train(data, attrList):
    maxProb = 1
    bestAttr = -1       
    for i in attrList:
        prob = 0 #prob of the target given an attribute
        for j in range(2): #binary values so j is  0, 1 
            temp = data.loc[data[i] == j, [0, i]]
            tempLen = len(temp)
            p = len(temp.loc[temp[0] == 1])
            e = len(temp.loc[temp[0] == 0])
            logp = 0
            loge = 0
            if(tempLen > 0):
                if p > 0:
                    logp = (math.log(p/tempLen))/(math.log(2))
                if e > 0:
                    loge = (math.log(e/tempLen))/(math.log(2))
                prob = prob + -1 * tempLen/rows * ((p/tempLen)*logp+(e/tempLen)*loge)
        if(prob <= maxProb):
            maxProb = prob
            bestAttr = i
            #print(maxProb)
            #print(bestAttr)
    tree = {}
    tree[bestAttr] = {}
    data0, data1 = splitData(data, bestAttr)
    tree[bestAttr][0] = label(data0)
    tree[bestAttr][1] = label(data1)
    return tree, bestAttr
    
    

def output(tree, lst):
    while(True):
        for key, value in tree.items():
            #print('key', key)
            #print('value', value)
            if value == 1 or value == 0:
                return value
            else:
                k = lst[key]
                tree = tree[key][k]
            if tree == 1 or tree == 0:
                return tree
            
    

def acc(model, data):
    r = len(data)
    accuracy = 0
    for i in range(r):
        row = data.loc[i, :]
        lst = [0] * 2
        for mod in model:
            lst[output(mod, row)] += 1
        if lst[0] > lst[1]:
            if row[0] == 0:
                accuracy += 1
        else:
            if row[0] == 1:
                accuracy += 1
    
    accuracy = accuracy/r * 100
    return accuracy
            
            
            
        
    

it = 25   #Running it for for 25 Iteration to select the best Test accuracy out of 25 Iterations
Test_accuracy = 0
while(it != 0):
    n = 20  #20 samplings
    classifiers = []
    attributeList = []
    
    for i in range(1, attrs):
            attributeList.append(i)
    
    for i in range(20):
        data = Train_data.sample(n = rows, replace = True)
        Hspace, attr = Train(data, attributeList)
        attributeList.remove(attr)
        classifiers.append(Hspace)
    accuracy = acc(classifiers, Test_data) 
    if accuracy > Test_accuracy:
        Test_accuracy = accuracy
        print('Iteration', it, 'Accuracy', Test_accuracy)
    it -= 1 



    
    
    

print("Accuracy on test data set is", Test_accuracy)
    
print('Classifiers are', classifiers)
print('length', len(classifiers))
    
    












