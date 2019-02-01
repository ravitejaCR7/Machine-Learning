# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd

data = pd.read_csv('perceptron.data',header = None)




x = data[[0, 1, 2, 3]]
print(x)
learningStep = 1

w = [0, 0, 0, 0]
b = 0

    


def predicted(x):
    wx = 0
    for i in range(len(x)):
        wx = wx + (w[i]*(x.iloc[i]))
    wx = wx + b
    return wx
      
    

def perceptronLoss():
    loss = 0
    for i in range(len(data)):
        f = predicted(x.iloc[i])
        z = -1*(f)*(data.iloc[i][4])
        loss = loss + max(0, z)
    return loss
            
   
def subGradient():
    lossw = [0, 0, 0, 0]
    lossb = 0
    for i in range(len(data)):
        z = -(data.iloc[i][4])*predicted(x.iloc[i])
        if(z >= 0):
            lossw = lossw + data.iloc[i][4]*(x.iloc[i])
            lossb = lossb + data.iloc[i][4]
    return (lossw, lossb) 



    
iterations = 10
it= 1
print("Weight and Bias for the first 3 iterations are ")
while(True):
    learningStep = 1/(1+it); 
    Deltaw, Deltab = subGradient()
    #print("Deltaw ",Deltaw)
    #print("Deltab ",Deltab)
    w = w + learningStep*Deltaw
    b = b + learningStep*Deltab
    pLoss = perceptronLoss()
    print(pLoss)
    if(it <= 3):
        print("Weight\n",w)
        print("bias\n",b)
     #stopping condition subgradient should be zer0 
    if(all(v == 0 for v in Deltaw) and Deltab==0):    
        break
    print(it)
    it = it+1
    
print("Total number of Iterations is ",it)
pLoss = perceptronLoss()
print("Loss ",pLoss)
print("Final weight ",w)
print("Final bias ",b)

