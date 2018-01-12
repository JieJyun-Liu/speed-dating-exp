
import pylab as pl
import numpy as np
import math

from numpy import array

from sklearn.decomposition import PCA
from sklearn.datasets import load_iris

from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler


def dataFromFile(fname):

    with open(fname, 'rb') as f:
        data = [row for row in f.read().splitlines()]
    
    return data

def fn(x):
    try:
        return float(x)

    except (ValueError, TypeError):
        return 0.0

if __name__ == "__main__":

	inFile = dataFromFile("corr_dec.csv")

	label_set = set()
	data = []
	target_id = []

	miss_data_ind = []

	# Construct Attributes Array (label_str)
	for tid, trans in enumerate(inFile):

		if tid == 0:
			continue

		items = trans.split(",")
		label_str = items[len(items)-1]
		
		# data
		del items[-1]

		for ind, item in enumerate(items):
			item = fn(item)
			items[ind] = item

		# print items
		data.append(items)
		label_set.add(label_str)

	label_array = list(sorted(label_set))

	print("\n -------------------------------------------- \n")
	print "Length of label array: " , len(label_array), label_array
	print "Total Data: ", len(inFile)
	print("\n -------------------------------------------- \n")

	print("\nBuild Data and Target id Array ======= \n")
	for tid, trans in enumerate(inFile):

		if tid == 0:
			continue

		items = trans.split(",")
		target_id.append(int(items[-1]))
		
	# print (target_id)

	# data, target, target_names
	dataDict = {}
	dataDict['target_names'] = label_array
	dataDict['data'] = data
	dataDict['target'] = target_id
	
	total = len(target_id)
	print "Data length: ", len(dataDict['data']), ", target length: ", len(dataDict['target'])

	# Translate data to ndarray
	labels_true = array(dataDict['target'])
	X = array(dataDict['data'])
	print "Type of Target: ", type(labels_true)
	print labels_true

	
	colors = np.random.rand(len(dataDict['data']))
	plot_data = StandardScaler().fit_transform(X)
	print "X: ", X
	print "plot data: ", plot_data

	# # for ind, p in enumerate(plot_data):
	# # 	if ind < 10:
	# # 		print X[ind], ", ", p

	db = DBSCAN(eps=0.5, min_samples=5).fit(plot_data)
	print "db = ", db
	core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
	core_samples_mask[db.core_sample_indices_] = True
	labels = db.labels_

	n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

	print('Estimated number of clusters: %d' % n_clusters_)
	print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
	print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
	print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
	print("Adjusted Rand Index: %0.3f"
	      % metrics.adjusted_rand_score(labels_true, labels))
	print("Adjusted Mutual Information: %0.3f"
	      % metrics.adjusted_mutual_info_score(labels_true, labels))
	# print("Silhouette Coefficient: %0.3f"
	#       % metrics.silhouette_score(X, labels))
	# print("Silhouette Coefficient: %0.3f"
	#       % metrics.silhouette_score(X, labels))

	##############################################################################
	# Plot result
	import matplotlib.pyplot as plt

	# Black removed and is used for noise instead.
	unique_labels = set(labels)
	colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
	for k, col in zip(unique_labels, colors):
	    if k == -1:
	        # Black used for noise.
	        col = 'k'

	    class_member_mask = (labels == k)

	    xy = X[class_member_mask & core_samples_mask]
	    plt.plot(xy[:, 0], 'o', markerfacecolor=col, markeredgecolor='k', markersize=7)
	    # plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col,
	             # markeredgecolor='k', markersize=14)


	    xy = X[class_member_mask & ~core_samples_mask]
	    print(len(xy))

	    if len(xy) > 0:
	    	plt.plot(xy[:, 0], 'x', markerfacecolor=col,
	             markeredgecolor='k', markersize=6)

	plt.title('Number of clusters (int_corr, dec, dec_o): %d' % n_clusters_)
	plt.show()
