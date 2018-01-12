''' Plot reference: http://ogrisel.github.io/scikit-learn.org/sklearn-tutorial/auto_examples/cluster/plot_dbscan.html '''
import sys
import csv
import random
import math
import operator

from optparse import OptionParser


def auto_execute(fname):
	inFile = dataFromFile(fname)
	
	trainingSet=[]
	testSet=[]
	split = 0.66

	loadDataset(inFile, split, trainingSet, testSet)
	print 'Train set: ' + repr(len(trainingSet))
	print 'Test set: ' + repr(len(testSet))

	predictions=[]
	k = 3
	for x in range(len(testSet)):
		neighbors = getNeighbors(trainingSet, testSet[x], k)
		result = getResponse(neighbors)
		predictions.append(result)
		# print('> predicted=' + repr(result) + ', actual=' + repr(testSet[x][-1]))
	accuracy = getAccuracy(testSet, predictions)
	print('Accuracy: ' + repr(accuracy) + '%')
	
	return accuracy



def loadDataset(inFile, split, trainingSet=[] , testSet=[]):
	# print (inFile[0])
	for i in range(1,len(inFile)):
		for j in range(len(inFile[i])):
			try:
				inFile[i][j] = float(inFile[i][j])
			except ValueError:
				if inFile[0][j] == 'met' or inFile[0][j] == 'met_o':
					inFile[i][j] = 2
					inFile[i][j] = float(inFile[i][j])
				elif inFile[0][j] == 'amb_o' or inFile[0][j] == 'shar_o' or inFile[0][j] == 'attr_o' or inFile[0][j] == 'prob_o':
					inFile[i][j] = 0
					inFile[i][j] = float(inFile[i][j])
				else:
					print(inFile[0][j])

		randValue = random.random()
		if randValue < split:
			# print("random value: %f" %randValue)
			trainingSet.append(inFile[i])
		else:
			testSet.append(inFile[i])

		# print(trainingSet[0])
		# print(testSet[0])


def dataFromFile(fname):

	with open(fname, 'rb') as f:
		data = [row for row in csv.reader(f.read().splitlines())]
	
	return data

# Gender?
def euclideanDistance(instance1, instance2, length):
	distance = 0
	for x in range(1, length):
		# print ("instance1[x]: %f, instance2[x]: %f" %(instance1[x],instance2[x]))
		# diff = instance1[x] - instance2[x]
		# print ("instance1[x]: %f, instance2[x]: %f, diff: %f" %(instance1[x],instance2[x],diff))
		distance += pow(instance1[x] - instance2[x], 2)

	return math.sqrt(distance)

def corrDistance(instance1, instance2, length):

	print



def getNeighbors(trainingSet, testInstance, k):
	distances = []
	length = len(testInstance)-1
	for x in range(len(trainingSet)):
		print "Test Instance: ", testInstance, " and trainingSet[", x, "] = ", trainingSet[x]
		dist = euclideanDistance(testInstance, trainingSet[x], length)
		print "Distance: ", dist, "\n"
		distances.append((trainingSet[x], dist))
	
	distances.sort(key=operator.itemgetter(1))
	neighbors = []
	for x in range(k):
		neighbors.append(distances[x][0])
	return neighbors

def getResponse(neighbors):
	classVotes = {}
	for x in range(len(neighbors)):
		response = neighbors[x][-1]
		if response in classVotes:
			classVotes[response] += 1
		else:
			classVotes[response] = 1
	# print("classVotes: ", repr(classVotes))
	sortedVotes = sorted(classVotes.iteritems(), key=operator.itemgetter(1), reverse=True)
	return sortedVotes[0][0]
 
def getAccuracy(testSet, predictions):
	correct = 0

	for x in range(len(testSet)):
		if testSet[x][-1] == predictions[x]:
			correct += 1
	return (correct/float(len(testSet))) * 100.0


def getConfusionMatrix(testSet, predictions):

	TP = 0.00
	FP = 0.00
	TN = 0.00
	FN = 0.00
	
	for x in range(len(testSet)):
		if testSet[x][-1] == predictions[x] == 1:
			TP += 1
		elif predictions[x] == 1 and (testSet[x][-1] != predictions[x]):
			FP += 1
		elif testSet[x][-1] == predictions[x] == 0:
			TN += 1
		elif predictions[x] == 0 and (testSet[x][-1] != predictions[x]):
			FN += 1

	precision = float(TP/(TP+FP))
	recall = float(TP/(TP+FN))
	F1 = (2*TP) / (2*TP+FP+FN)
	print(TP, TP+FP, TP/(TP+FP))

	return(TP, FP, TN, FN, precision, recall, F1)


if __name__ == "__main__":

	# Load file
	optparser = OptionParser()
	optparser.add_option('-f', '--inputFile',
						 dest='input',
						 help='filename containing csv',
						 default=None)
	(options, args) = optparser.parse_args()

	inFile = None
	if options.input is None:
		inFile = sys.stdin
	elif options.input is not None:
		inFile = dataFromFile(options.input)
	else:
		print 'No dataset filename specified, system with exit\n'
		sys.exit('System will exit')

	# print(inFile)
	trainingSet=[]
	testSet=[]
	split = 0.66

	loadDataset(inFile, split, trainingSet, testSet)
	print 'Train set: ' + repr(len(trainingSet))
	print 'Test set: ' + repr(len(testSet))

	predictions=[]
	k = 3
	for x in range(len(testSet)):
		neighbors = getNeighbors(trainingSet, testSet[x], k)
		result = getResponse(neighbors)
		predictions.append(result)
		print('> predicted=' + repr(result) + ', actual=' + repr(testSet[x][-1]))
	accuracy = getAccuracy(testSet, predictions)
	cMatrix = getConfusionMatrix(testSet, predictions)

	print('Accuracy: ' + repr(accuracy) + '%')
	print('Confusion Matrix, Precision, Recall, F1: ' + repr(cMatrix))
