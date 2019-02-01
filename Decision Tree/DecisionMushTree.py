
import pandas as pd
import math
Training_data = pd.read_csv('mush_train.data',header = None)
Test_data = pd.read_csv('mush_test.data',header = None)


attrLen = len(Training_data.columns)-1

#print(Training_data.loc[Training_data[0] == 'p'] )
    

def split(data, dupAttrList):
    rows = len(data)    
    maxProb = 1
    bestAttr = -1
    for i in dupAttrList:
        prob = 0 #prob of the target given an attribute
        for j in attrValues[i]:
            temp = data.loc[data[i+1] == j, [0,i+1]]
            tempLen = len(temp)
            p = len(temp.loc[temp[0] == 'p'])
            e = len(temp.loc[temp[0] == 'e'])
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
    return bestAttr
            

            
            
            
            

def build_tree(Training_data, root):
    dupAttrList = []
    for i in range(attrLen):
        dupAttrList.append(i)
    max_height = 0;
    attr = split(Training_data, dupAttrList)
    root = attr
    dupAttrList.remove(attr)
    lst = [] 
    i = len(attrValues[attr]) - 1
    while i >= 0:
        lst.append([attrValues[attr][i],attr,0])
        i = i-1
    data[attr] = Training_data;
    i = 0 
    Node_Count = 0
    while(len(lst) != 0):
        entry = lst.pop()
        Node_Count = Node_Count + 1; 
        attrToSplitOn = entry[1]
        valueToSplitOn = entry[0]
        DataToSplitOn = data[attrToSplitOn]
        SplitData = DataToSplitOn.loc[DataToSplitOn[attrToSplitOn+1] == valueToSplitOn]
        p = len(SplitData.loc[SplitData[0] == 'p'])
        e = len(SplitData.loc[SplitData[0] == 'e'])
        height = entry[2]+1
        if height > max_height:
            max_height = height
        lenSplitData = len(SplitData)
        if(lenSplitData == 0):
            p = len(DataToSplitOn.loc[DataToSplitOn[0] == 'p'])
            e = len(DataToSplitOn.loc[DataToSplitOn[0] == 'e'])
            if p > e:
                tree[attrToSplitOn][valueToSplitOn] = 'p'
            else:
                tree[attrToSplitOn][valueToSplitOn] = 'e'
        else:
            if p == lenSplitData:
                tree[attrToSplitOn][valueToSplitOn] = 'p'
            elif e == lenSplitData:
                tree[attrToSplitOn][valueToSplitOn] = 'e'
            else:
                attr = split(SplitData, dupAttrList)
                dupAttrList.remove(attr)
                data[attr] = SplitData;
                tree[attrToSplitOn][valueToSplitOn] = attr
                i = len(attrValues[attr]) - 1
                while i >= 0:
                    lst.append([attrValues[attr][i],attr,height])
                    i = i-1
    for i in dupAttrList:
        del tree[i]
    return root, max_height+1, Node_Count;
              
    

attrValues = []

attrValues.append(['b', 'c', 'x', 'f', 'k', 's'])
attrValues.append(['f', 'g', 'y', 's'])
attrValues.append(['n','b','c','g','r','p','u','e','w','y'])
attrValues.append(['t','f'])
attrValues.append(['a','l','c','y','f','m','n','p','s'])#4
attrValues.append(['a','d','f','n'])
attrValues.append(['c','w','d'])
attrValues.append(['b','n'])
attrValues.append(['k','n','b','h','g','r','o','p','u','e','w','y'])
attrValues.append(['e','t'])
attrValues.append(['b','c','u','e','z','r','m'])
attrValues.append(['f','y','k','s'])
attrValues.append(['f','y','k','s'])
attrValues.append(['n','b','c','g','o','p','e','w','y'])
attrValues.append(['n','b','c','g','o','p','e','w','y'])
attrValues.append(['p','u'])
attrValues.append(['n','o','w','y'])
attrValues.append(['n','o','t'])
attrValues.append(['c','e','f','l','n','p','s','z'])
attrValues.append(['k','n','b','h','r','o','u','w','y'])
attrValues.append(['a','c','n','s','v','y'])
attrValues.append(['g','l','m','p','u','w','d'])



attrList = []


data = {}

target = ['p','e']

dupAttrList = attrList


tree = {}

for i in range(attrLen):
    attrList.append(i)

def Intialize(tree):
    for i in attrList:
        tree[i] = {}
        data[i] = {}
        for j in attrValues[i]:
            tree[i][j] = -1
        
 #duplicate Attribute List for not repeating the attributes



root = None
Intialize(tree)
root, height, Node_count = build_tree(Training_data, root) #build tree


#deleting the entries of the attributes that are not used

    
print(tree)

print('Height of the tree is',height)

print('Number of the nodes in the tree are',Node_count)

Training_DataLength = len(Training_data)
i = 0


def accuracy(tree, data, root):
    dataLength = len(data)
    acc = 0
    i = 0
    while i < dataLength:
        result = root
        while True:
            result = tree[result][data.iloc[i][result+1]]
            if result == 'e' or result == 'p':
                break
        if data.iloc[i][0] == result:
            acc = acc + 1
        i = i+1
    return (acc/dataLength)*100;
    

trainingAccuracy = accuracy(tree, Training_data, root)
testingAccuracy = accuracy(tree, Test_data, root)
print('Testing accuracy is', testingAccuracy)

print('Training accuracy is', trainingAccuracy)

#Merging datasets for testing accuracy based on dataset split

mushroom_data = pd.concat([Training_data, Test_data])
lenMushroom = len(mushroom_data)
#for problem 2.7

i = 10 #starting with 50% of training data and remaing of test data
while i < 100:
    trainLength = math.ceil((i/100)*lenMushroom)
    testLength = lenMushroom - trainLength
    #print(trainLength, testLength)
    Training_data = mushroom_data.iloc[0:trainLength]
    Test_data = mushroom_data.iloc[trainLength+1:lenMushroom]
    tree = {}
    root = None
    Intialize(tree)
    #print(tree)
    root, height, Node_count  = build_tree(Training_data, root)
    #print(tree)
    trainingAccuracy = accuracy(tree, Training_data, root)
    #print('height', height)
    #print('Node_count',Node_count)
    testingAccuracy = accuracy(tree, Test_data, root)
    print(i)
    print('with',i,'% split of training and',100-i,'% split of testing data')
    print('Testing accuracy is', testingAccuracy)
    print('Training accuracy is', trainingAccuracy)
    i = i + 10
        
        
