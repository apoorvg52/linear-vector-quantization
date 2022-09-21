from math import sqrt
from random import randrange
from random import seed

# calculate the Euclidean distance between two vectors
def euclidean_distance(row1, row2):
	distance = 0.0
	for i in range(len(row1)-1):
		distance += (row1[i] - row2[i])**2
	return sqrt(distance)
 
# Locate the best matching unit
def get_best_matching_unit(codebooks, test_row):
	distances = list()
	for codebook in codebooks:
		dist = euclidean_distance(codebook, test_row)
		distances.append((codebook, dist))
	distances.sort(key=lambda tup: tup[1])
	return distances[0][0]
 
# Create a random codebook vector
def random_codebook(train):
	n_records = len(train)
	n_features = len(train[0])
	codebook = [train[randrange(n_records)][i] for i in range(n_features)]
	return codebook
 
# Train a set of codebook vectors
def train_codebooks(train, n_codebooks, lrate, epochs):
    
	codebooks = [train[0],train[5]]
	for epoch in range(epochs):
		rate = lrate * (1.0-(epoch/float(epochs)))
		sum_error = 0.0
		for row in train:
			bmu = get_best_matching_unit(codebooks, row)
			for i in range(len(row)-1):
				error = row[i] - bmu[i]
				sum_error += error**2
				if bmu[-1] == row[-1]:
					bmu[i] += rate * error
				else:
					bmu[i] -= rate * error
		print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, rate, sum_error))
	return codebooks
 
    
# Make a prediction with codebook vectors
def predict(codebooks, test_row):
	bmu = get_best_matching_unit(codebooks, test_row)
	return bmu[-1]


def accuracy_metric(actual, predicted):
	correct = 0
	for i in range(len(actual)):
		if actual[i] == predicted[i]:
			correct += 1
	return correct / float(len(actual)) * 100.0



# Test the training function.

import pandas as pd
import numpy as np
df = pd.read_csv('g:/data/blood-donation.csv')

x1=np.array(df["Months since Last Donation"])
x2=np.array(df['Number of Donations'])
x3=np.array(df['Total Volume Donated (c.c.)'])
x4=np.array(df['Months since First Donation'])
y=np.array(df['Made Donation in March 2007'])
x_train=[]
x_test=[]



#20% train data and 80% test data    
for i in range(int(len(x1)/4)):
    x_train.append([x1[i],x2[i],x3[i],x4[i],y[i]])
    
for j in range(int(len(x1)/4),int(len(x1)-10)):
    x_test.append([x1[j],x2[j],x3[j],x4[j],y[j]])

# for list conversion print df.iloc[:, 0].tolist()
u=x_train[0]
v=x_train[5]
print(u,v)

#train dataset
learn_rate = 0.3
n_epochs = 10
n_codebooks = 2
codebooks = train_codebooks(x_train, n_codebooks, learn_rate, n_epochs)
print('Codebooks: %s' % codebooks)


#predicting the values
predictions = list()
actual=[]
for row in x_test:
	output = predict(codebooks, row)
	predictions.append(output)
print(predictions)


#calculating accuracy and printing the accuracy by comapring actual and prediction values
actual = [row[-1] for row in x_test]
print("\n\n\naccuracy:",accuracy_metric(actual,predictions))

