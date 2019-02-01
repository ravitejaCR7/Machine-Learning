#stochastic
import pandas as pd

data = pd.read_csv('perceptron.data',header = None)




x = data[[0, 1, 2, 3]]

learningStep = 1
#[149.277140, 52.533473, 1.67163, -172.891940] w
#-322.0 b
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
            
   
def subGradient(i):
    lossw = [0, 0, 0, 0]
    lossb = 0
    z = -(data.iloc[i][4])*predicted(x.iloc[i])
    if(z >= 0):
        lossw = lossw + data.iloc[i][4]*(x.iloc[i])
        lossb = lossb + data.iloc[i][4]
    return (lossw, lossb) 



    
iterations = 1000
i = 0 
k= 0

print("Weight and Bias for the first 3 iterations are ")
while(True):
#    print("i",i)
    Deltaw, Deltab = subGradient(i)
    w = w + learningStep*Deltaw
    b = b + learningStep*Deltab
    if(i <= 2 and k<1):
        print("Weight\n",w)
        print("bias\n",b)
    if(i%1000 == 0):
        pLoss = perceptronLoss()
        #print("pLoss \t",pLoss)
        if(pLoss == 0):
            print(k)
            print(i)
            break
    i = i+1
    if(i == 1000):
        i=0
        k = k + 1
        print(k)
print("Total number of iterations is ",(k+1)*1000)
print("Loss ",pLoss)
print("Final weight ",w)
print("Final bias ",b)

"""i = 0
while(i<100):
    print(predicted(x.iloc[i]))
    i = i + 1"""
