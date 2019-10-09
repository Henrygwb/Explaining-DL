import os
import collections
import numpy as np
from scipy import io
from matplotlib import pyplot as plt
from keras.models import load_model
import rpy2.robjects as robjects

"""
1. Highlight the feature importance in heat map.

2. Fidelity tests:
	1. Booststrapped positive.
	2. Booststrapped negative.
	3. New testing cases.
"""

class fidelity_test(object):
	def __init__(self, digit, data_file_1, data_file_2, param_file, bs_rate, num_fea):

		self.X = io.loadmat(data_file_1)['X']
		feature = io.loadmat(data_file_2)['var_idx']
		robjects.r['load'](param_file)
		self.Z = np.asarray((robjects.r['final_params'][0]))
		self.Beta = np.asarray((robjects.r['final_params'][1]))
		self.model = load_model('../model/fashion_mnist_cnn.h5')

		mixture_idx = collections.Counter(self.Z).most_common()[0][0]

		data = []
		for i in xrange(self.X.shape[0]):
			if (self.Z[i] == mixture_idx):
				data.append(self.X[i,])
		data = np.asarray(data)
		idx = np.random.permutation(data.shape[0])[0:int(bs_rate * data.shape[0])]
		self.X = data[idx,]

		self.Y = np.zeros((self.X.shape[0], 10))
		self.Y[:, digit] = 1

		## get all possible mixture components.
		self.feature_all = []
		for i in np.unique(self.Z):
			self.beta = self.Beta[:, i - 1]
			self.sorted_idx = np.argsort(-abs(self.beta))
			self.feature_all.append(feature[0, self.sorted_idx])

		## get the most common mixture component.
		self.beta = self.Beta[:, mixture_idx - 1]
		self.sorted_idx = np.argsort(-abs(self.beta))
		self.feature = feature[0, self.sorted_idx]

		data_1 = np.zeros_like(data[idx,])
		self.idx = []
		fea = [num_fea]
		data_1[:, self.feature[0:fea[0]]] = self.X[:, self.feature[0:fea[0]]]
		p = self.model.predict(data_1.reshape(data_1.shape[0], 28, 28, 1))[:, digit]
		self.idx.append(np.argmax(p))

	def visualize_feature_importance(self, file_name, num_fea=150):

		i = 1
		for feature in self.feature_all:
			selected_feature = feature[0:num_fea]
			print selected_feature.shape

			a = []
			for iii in xrange(num_fea):
				tmp = (1 / float(num_fea)) * (iii + 1)
				a.append(tmp)

			b_1 = np.zeros((1, 784))
			b_1[0, selected_feature] = a
			b_1 = b_1.reshape(28, 28)
			plt.imsave( file_name+'_'+str(i)+'.pdf', b_1, cmap='hot')
			i = i+1

	def fidelity_test_nullification(self, num_fea, option, pp):
		########### Select important features ###########
		X_select = self.X.copy()
		for i in xrange(self.X.shape[0]):
			tmp = self.X[i, self.feature].copy().reshape(1, 300)
			j = 0
			for k in xrange(300):
				if j == num_fea:
					break
				idx_tmp = self.sorted_idx[k]
				if tmp[0, idx_tmp] != 0:
					j = j + 1
					if option == 0:
						if self.beta[idx_tmp] < 0:
							tmp[0, idx_tmp] = self.X[self.idx[pp], self.feature[idx_tmp]]
						elif self.beta[idx_tmp] > 0:
							tmp[0, idx_tmp] = 0
					elif option == 1:
						tmp[0, idx_tmp] = -self.X[self.idx[pp], self.feature[idx_tmp]] * 0.5
			X_select[i, self.feature] = tmp

		score_sel = self.model.evaluate(X_select.reshape(self.X.shape[0], 28, 28, 1), self.Y, verbose=0)

		########### Random pick up features ###########
		X_ran = self.X.copy()
		for i in xrange(X_ran.shape[0]):
			idx = np.random.permutation(784)[0:num_fea]
			X_ran[i, idx] = 0
		score_ran = self.model.evaluate(X_ran.reshape(self.X.shape[0], 28, 28, 1), self.Y, verbose=0)
		return score_sel, score_ran

	def fidelity_test_filling(self, data, num_fea, bs_rate, pp):
		############ Boostrap step ######################
		idx = np.random.permutation(data.shape[0])[0:int(self.X.shape[0])]
		data = data[idx,]
		# print data.shape
		Y = self.Y[0:data.shape[0], ]

		########### Select important features ###########
		X_select = data.copy()
		X_select[:, self.feature[0:num_fea]] = self.X[self.idx[pp], self.feature[0:num_fea]] * 5

		score_sel = self.model.evaluate(X_select.reshape(data.shape[0], 28, 28, 1), Y, verbose=0)

		########### Random pick up features ###########
		X_ran = data.copy()
		for i in xrange(X_ran.shape[0]):
			idx = np.random.permutation(784)[0:num_fea]
			X_ran[i, idx] = 1

		score_ran = self.model.evaluate(X_ran.reshape(data.shape[0], 28, 28, 1), Y, verbose=0)
		# prob_ran = self.model.predict(X_ran.reshape(data.shape[0], 28, 28, 1), verbose=0)
		return score_sel, score_ran

	def fidelity_test_pathological(self, num_fea, mean, var, pp):
		########### Select important features ##########
		X_select = np.zeros_like(self.X)
		for kk in xrange(X_select.shape[0]):
			X_select[kk,] = np.random.normal(mean, var, 784)
		X_select[:, self.feature[0:num_fea]] = self.X[self.idx[pp], self.feature[0:num_fea]] * 3

		score_sel = self.model.evaluate(X_select.reshape(self.X.shape[0], 28, 28, 1), self.Y, verbose=0)
		# prob_sel = self.model.predict(X_select.reshape(data.shape[0], 28, 28, 1), verbose = 0)

		########### Random pick up features ###########
		X_ran = np.zeros_like(self.X)
		for i in xrange(X_ran.shape[0]):
			X_ran[i,] = abs(np.random.normal(mean, var, 784))
			idx = np.random.permutation(784)[0:num_fea]
			X_ran[i, idx] = 1
		score_ran = self.model.evaluate(X_ran.reshape(self.X.shape[0], 28, 28, 1), self.Y, verbose=0)
		return score_sel, score_ran,


def fidelity_bootstrap(digit, data_file_1, data_file_2, data_file_3, param_file, boost_time, num_fea, mean,
					  var, bs_rate, pp):
	fidelity_test_0 = fidelity_test(digit, data_file_1, data_file_2, param_file, bs_rate, num_fea)
	fidelity_test_0.visualize_feature_importance(file_name='../results/heatmap')
	null_result_score_sel = []
	null_result_score_ran = []

	for i in xrange(boost_time):
		score_sel, score_ran = fidelity_test_0.fidelity_test_nullification(num_fea, 1, pp)
		null_result_score_sel.append(score_sel)
		null_result_score_ran.append(score_ran)
	null = dict(score_sel=null_result_score_sel, score_ran=null_result_score_ran)

	data = io.loadmat(data_file_3)['X']
	fill_result_score_sel = []
	fill_result_score_ran = []

	for i in xrange(boost_time):
		score_sel, score_ran = fidelity_test_0.fidelity_test_filling(data, num_fea, bs_rate, pp)
		fill_result_score_sel.append(score_sel)
		fill_result_score_ran.append(score_ran)

	fill = dict(score_sel=fill_result_score_sel, score_ran=fill_result_score_ran)

	path_result_score_sel = []
	path_result_score_ran = []

	for i in xrange(boost_time):
		score_sel, score_ran = fidelity_test_0.fidelity_test_pathological(num_fea, mean, var, pp)
		path_result_score_sel.append(score_sel)
		path_result_score_ran.append(score_ran)

	path = dict(score_sel=path_result_score_sel, score_ran=path_result_score_ran)

	result = dict(null=null, path=path, fill=fill)
	return result

def accuracy_statistics(name):
	score_sel = name['score_sel']
	score_ran = name['score_ran']
	sel_score = []
	ran_score = []

	for i in xrange(len(score_sel)):
		sel_score.append(score_sel[i][1])
		ran_score.append(score_ran[i][1])

	print '*******************************************'
	print "Results of selected important features"
	# print "Lowest: "
	# print min(sel_score)
	# print "Highest: "
	# print max(sel_score)
	# print "Average:"
	print np.mean(sel_score)
	print '*******************************************'
	print "Results of random selected features"
	# print "Lowest:"
	# print min(ran_score)
	# print "Highest:"
	# print max(ran_score)
	# print "Average:"
	print np.mean(ran_score)
	print '*******************************************'

if __name__ == '__main__':

	class_exp = 1

	data_file_1 = '../data/data_1.mat'
	data_file_2 = '../data/data_1_300.mat'
	data_file_3 = '../data/data_3.mat'
	param_file = '../results/final_parameters_1_300.RData'

	# number of bootstrap
	boost_time = 10
	# bootstrap percentage
	bs_rate = 0.3
	# number of selected features
	num_fea = 75
	# mean and variance of the random noise
	mean = 0.03
	var = 0.01

	result = fidelity_bootstrap(class_exp, data_file_1, data_file_2, data_file_3, param_file, boost_time,
								num_fea, mean, var, bs_rate, 0)

	print 'Result of Nullification...'
	null = result['null']
	accuracy_statistics(null)
	print ' '
	print 'Result of Pathological...'
	path = result['path']
	accuracy_statistics(path)
	print ' '
	print 'Result of Filling...'
	fill = result['fill']
	accuracy_statistics(fill)

