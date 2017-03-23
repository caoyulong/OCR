# this program is used to decide the number of hidden nodes and the number of training
# number of hidden nodes 10~50
# number of training 1~10

import numpy as np
from ocr import OCRNeuralNetwork
from sklearn.cross_validation import train_test_split

def test(data_matrix, data_labels, test_indices, nn):
    cnt = 0
    for i in test_indices:
        test = data_matrix[i]
        prediction = nn.predict(test)
        if data_labels[i] == prediction:
            cnt += 1
    return cnt / float(len(test_indices))

# Load data samples and labels into matrix
data_matrix = np.loadtxt(open('mydata.csv', 'rb'), delimiter = ',').tolist()
data_labels = np.loadtxt(open('mydataLabels.csv', 'rb')).tolist()

# Create training and testing sets.
train_indices, test_indices = train_test_split(list(range(len(data_matrix))))

print "PERFORMANCE"
print "-----------"

maxi,maxj = 10,1
maxnn = OCRNeuralNetwork(400, 10, 10, data_matrix, data_labels, train_indices, 1)
maxp = test(data_matrix, data_labels, test_indices, maxnn)

for i in xrange(10,50):
    for j in xrange(1,10):
        nn = OCRNeuralNetwork(400, i, 10, data_matrix, data_labels, train_indices, j)
        p = test(data_matrix, data_labels, test_indices, nn)
        if p > maxp:
            maxi,maxj,maxp = i,j,p
        performance = str(p)
        print "{i} Hidden Nodes, {j} trainings: {val}".format(i=i, j=j, val=performance)

print 'max:', maxi, 'Hidden Nodes,', maxj, 'trainings:', str(maxp)