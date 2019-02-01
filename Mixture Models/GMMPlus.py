import numpy as np
import pandas as pd
import math
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import fowlkes_mallows_score


def getClusterCenters(data, k):
    rows, cols = data.shape
    centers = np.zeros((k, cols))
    r = np.random.choice(rows, 1) #Initial Choice
    centers[0, :] = data[r, :]
    k_center = 1
    while k_center < k:
        pd = [] #pd represents probability distribution
        dist_sum = 0
        for i in range(rows):
            max_dist = 1000
            for j in range(k_center):
                dist = np.linalg.norm(data[i] - centers[j]) ** 2
                if dist < max_dist:
                    max_dist = dist
            d = max_dist*max_dist
            dist_sum += d
            pd.append(d)
        for i in range(rows):
            pd[i] = pd[i]/dist_sum
        r = np.random.choice(rows, 1, p=pd) 
        centers[k_center, :] = data[r, :]
        k_center = k_center + 1
    return centers

# Log Likelihood 
def logLikelihood(data, mean, covariance, lambda1, k, m):
    log = 0
    for i in range(len(data)):
        sum1 = 0
        for j in range(k):
            sum1 += lambda1[j] * guassian(data[i], mean[j], covariance[j], m)
        log += np.log(sum1)
    return log

def gmmPred(data, mean, covariance, lambda1, k, m):
    pred = []
    for i in range(len(data)):
        best_likelihood = None
        best_cluster = None
        for j in range(k):
            likelihood = lambda1[j] * guassian(data[i], mean[j], covariance[j], m)
            if best_likelihood is None or best_likelihood <= likelihood:
                best_likelihood = likelihood
                best_cluster = j
        pred.append(best_cluster)
    return pred

def preprocessData(data): #Preprocessing Data
    rows, cols = data.shape
    for i in range(cols):
        mean = np.mean(data[:, i])
        std = np.std(data[:, i])
        data[:, i] = (data[:, i]-mean)/std
    return data

def guassian(row, mean, covariance, m): # Multivariate Gaussian
    diff_data_mean = np.array(row - mean).reshape(1, m)
    exp = np.exp(-0.5 * np.dot(np.dot(diff_data_mean, np.linalg.inv(covariance)), diff_data_mean.T))
    return (1 / np.sqrt(((2 * math.pi) ** m) * np.linalg.det(covariance))) * exp


train_data = np.array(pd.read_csv('leaf.data',header = None)) #Train data
trainLength = len(train_data)
train_features = train_data[:, 0]
train_data = train_data[:, 1:]
cols = len(train_data[0])

scaledData = preprocessData(train_data) #Scaling data

# K array
kArray = [12, 18, 24, 36, 42]

# Get GMM objective loss array and compute mean and variance
lossArray = []

meanArray = []
covarianceArray = []
lambdaArray = []

# For each K
for k in kArray:
    print('K-value',k)
    for i in range(20): #20 random Intializations
        print("Iter",i)
        mean_arr = getClusterCenters(train_data, k)
        covMatrix = np.empty((k, cols, cols))
        for j in range(k):
            covMatrix[j] = np.identity(n=cols, dtype=np.float64)

        lambda_arr = np.empty((k, 1), dtype=np.float64)
        for j in range(k):
            lambda_arr[j] = 1/k

        log_like_val = logLikelihood(scaledData, mean_arr, covMatrix, lambda_arr, k, cols)
        iteration_counter = 1
        while True:
            q_array = np.empty((trainLength, k), dtype=np.float64)             #E Step
            for x in range(trainLength):
                den_sum = 0
                for k_val in range(k):
                    q_array[x, k_val] = lambda_arr[k_val] * guassian(scaledData[x], mean_arr[k_val], covMatrix[k_val], cols)
                    den_sum += q_array[x, k_val]

                q_array[x] = q_array[x] / den_sum

            for k_val in range(k):             #M Step
                num_total = 0
                den_total = 0
                for m in range(trainLength):
                    num_total += q_array[m, k_val] * scaledData[m]
                    den_total += q_array[m, k_val]

                mean_arr[k_val] = num_total / den_total

            for k_val in range(k):
                num_total = 0
                den_total = 0
                for m in range(trainLength):
                    diff_vector = scaledData[m] - mean_arr[k_val]
                    diff_vector = np.array(diff_vector).reshape((1, cols))
                    num_total += q_array[m, k_val] * np.dot(diff_vector.T, diff_vector)
                    den_total += q_array[m, k_val]

                covMatrix[k_val] = num_total / den_total
                covMatrix[k_val] += np.identity(n=cols)

            for k_val in range(k):
                num_total = 0
                for m in range(trainLength):
                    num_total += q_array[m, k_val]

            lambda_arr[k_val] = num_total / trainLength

            pLogLikelihood = log_like_val
            log_like_val = logLikelihood(scaledData, mean_arr, covMatrix, lambda_arr, k, cols)

            # Convergence Check
            if pLogLikelihood >= log_like_val:
                lossArray.append(log_like_val)
                meanArray.append(mean_arr)
                covarianceArray.append(covMatrix)
                lambdaArray.append(lambda_arr)
                break

# Mean and variance of GMM objective
index = 0                
while index < 5:
    k = index * 20
    print("The mean and Variance of the GMM Objective for k =",kArray[index],"is", np.mean(lossArray[k:k+20]), "and:", np.var(lossArray[k:k+20]))
    index += 1


# Predict clusters with k = 36
adjRand = 0
fms = 0
temp_data = np.append(scaledData, np.array(train_features - 1).reshape((trainLength, 1)), axis=1)
for i in range(20):
    predict_array = gmmPred(scaledData, meanArray[60+i], covarianceArray[60+i], lambdaArray[60+i], 36, cols)
    adjRand += adjusted_rand_score(train_features-1, predict_array)
    fms += fowlkes_mallows_score(train_features-1, predict_array)

print("Adjusted Rand Index of the GMM model with k = 36 is", adjRand/20)
print("Fowkes Mallows Score of the GMM model with k = 36 is", fms/20)
