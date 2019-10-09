import os
#os.environ["THEANO_FLAGS"] = "device=gpu,floatX=float32"
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from scipy import io
import rpy2.robjects as robjects
from keras.models import load_model
np.random.seed(1234)
from keras.utils import np_utils
import sklearn
import scipy as sp
from sklearn.linear_model import Ridge
from skimage.segmentation import slic
import argparse

def distance_fn(x, distance_metric='cosine'):
    return sklearn.metrics.pairwise.pairwise_distances(x, x[0], metric=distance_metric).ravel()

def perf_measure(y_true, y_pred):
    TP_FN = np.count_nonzero(y_true)
    FP_TN = y_true.shape[0]  - TP_FN
    FN = np.where((y_true - y_pred) == 1)[0].shape[0]
    TP = TP_FN - FN
    FP = np.count_nonzero(y_pred) - TP
    TN = FP_TN - FP
    Precision = float(float(TP) / float(TP + FP + 1e-9))
    Recall = float(float(TP) / float((TP + FN + 1e-9)))
    accuracy = float(float(TP + TN) / float((TP_FN + FP_TN + 1e-9)))
    F1 =  2*((Precision * Recall) / (Precision + Recall))
    return Precision, Recall, accuracy, F1

def forward_selection(data, labels, weights, num_features):
    clf = sklearn.linear_model.Ridge(alpha=0, fit_intercept=True)
    used_features = []
    for _ in range(min(num_features, data.shape[1])):
        max_ = -100000000
        best = 0
        for feature in range(data.shape[1]):
            if feature in used_features:
                continue
            clf.fit(data[:, used_features + [feature]], labels,
                    sample_weight=weights)
            score = clf.score(data[:, used_features + [feature]],
                              labels,
                              sample_weight=weights)
            if score > max_:
                best = feature
                max_ = score
        used_features.append(best)
    return np.array(used_features)

def feature_selection(data, labels, weights, num_features, method = 'highest_weights'):
    if method == 'none':
        return np.array(range(data.shape[1]))
    elif method == 'forward_selection':
        return forward_selection(data, labels, weights, num_features)
    elif method == 'highest_weights':
        clf = sklearn.linear_model.Ridge(alpha=0, fit_intercept=True)
        clf.fit(data, labels, sample_weight=weights)
        feature_weights = sorted(zip(range(data.shape[0]),
                                     clf.coef_ * data[0]),
                                 key=lambda x: np.abs(x[1]),
                                 reverse=True)
        return np.array([x[0] for x in feature_weights[:num_features]])
    elif method == 'auto':
        if num_features <= 6:
            n_method = 'forward_selection'
        else:
            n_method = 'highest_weights'
        return feature_selection(data, labels, weights, num_features, n_method)


def LIME_xai(data_explain, num_samples, num_features):
    n_features = 135
    data = np.random.randint(0, 2, num_samples*n_features).reshape((num_samples, n_features))
    data[0, :] = 1
    malware = []
    for row in data:
        temp = np.copy(data_explain)
        zeros = np.where(row == 0)[0]
        for x in zeros:
            temp[0, x] = 0
        temp = temp.reshape(135, )
        malware.append(temp)

    labels = model.predict(np.array(malware))[:, 1]

    distances = distance_fn(sp.sparse.csr_matrix(data))

    neighborhood_data = data
    neighborhood_labels = labels
    distances = distances

    kernel_width = 25
    weights = np.sqrt(np.exp(-(distances ** 2) / kernel_width ** 2))

    used_features = forward_selection(data, labels, weights, num_features)
    labels_column = neighborhood_labels
    easy_model = Ridge(alpha=0, fit_intercept=True)
    easy_model.fit(neighborhood_data[:, used_features], labels_column, sample_weight=weights)
    prediction_score = easy_model.score(neighborhood_data[:, used_features], labels_column,
                                        sample_weight=weights)

    local_pred = easy_model.predict(neighborhood_data[0, used_features].reshape(1, -1))
    fea = used_features[np.argsort(np.abs(easy_model.coef_))[::-1]]

    return fea

class xai_mlp(object):
    """class for explaining the rnn prediction"""
    def __init__(self, model, data, fea, n_fea_select):
        """
        Args:
            model: target rnn model.
            data: data sample needed to be explained.
            label: label of the data sample.
            start: value of function start.
        """
        self.model = model
        self.data = data
        self.pred = self.model.predict(self.data, verbose = 0)[0, 1]
        self.fea = fea.reshape(1, n_fea_select)
        self.n_features = n_fea_select

## three types of fidelity test method.
class fid_test(object):
    def __init__(self, xai_mlp):
        self.xai_mlp = xai_mlp

    ## feature deduction testing
    def fea_deduct_test(self, num_fea):
        test_data = np.copy(self.xai_mlp.data)
        selected_fea = self.xai_mlp.fea[0, 0:num_fea]
        test_data[0, selected_fea] = 1
        P1 = self.xai_mlp.model.predict(test_data, verbose=0)[0, 1]

        random_fea = np.random.choice(135, num_fea, replace=False)
        test_data_1 = np.copy(self.xai_mlp.data)
        test_data_1[0, random_fea] = 1
        P2 = self.xai_mlp.model.predict(test_data_1, verbose=0)[0, 1]
        return test_data, P1, P2

    ## feature augmentation testing
    def fea_aug_test(self, test_seed, num_fea):
        test_seed = test_seed.reshape(1, 135)
    
        test_data = np.copy(test_seed)
        selected_fea = self.xai_mlp.fea[0, 0:num_fea]
        test_data[0, selected_fea] = self.xai_mlp.data[0,selected_fea]
        P_neg_1 = self.xai_mlp.model.predict(test_data, verbose=0)[0, 1]

        random_fea = np.random.choice(135, num_fea, replace=False)
        test_data_1 = np.copy(test_seed)
        test_data_1[0, random_fea] = self.xai_mlp.data[0,random_fea]
        P_neg_2 = self.xai_mlp.model.predict(test_data_1, verbose=0)[0, 1]

        return test_data, P_neg_1, P_neg_2

    ## Synthetic testing
    def syn_test(self, num_fea):

        test_data = np.ones_like(self.xai_mlp.data)
        selected_fea = self.xai_mlp.fea[0, 0:num_fea]
        test_data[0, selected_fea] = self.xai_mlp.data[0, selected_fea]
        P_test_1 = self.xai_mlp.model.predict(test_data, verbose=0)[0, 1]

        random_fea = np.random.choice(135, num_fea, replace=False)
        test_data_1 = np.ones_like(self.xai_mlp.data)
        test_data_1[0, random_fea] = self.xai_mlp.data[0, random_fea]
        P_test_2 = self.xai_mlp.model.predict(test_data_1, verbose=0)[0, 1]

        return test_data, P_test_1, P_test_2

def dimensional_reduction(X):
    var = []
    for i in xrange(X.shape[1]):
        var_tmp = np.var(X[:,i])
        var.append(var_tmp)
    var = np.asarray(var)
    sort_var = np.argsort(var)
    return sort_var

if __name__ == "__main__":
    
    ## specify the number of features selected.
    parser = argparse.ArgumentParser()
    parser.add_argument('-nf', '--num_feature', help='number of features', type=str, default='5')
    args = parser.parse_args()
    n_fea_select = int(args.num_feature)

    ## load the model and data
    model = load_model('../model/pdf_mlp.h5')
    PATH_TEST_DATA = '../data/train_pdf.npz'

    train = np.load(PATH_TEST_DATA)
    X_train = train['X_train']
    y_train = train['Y_train']
    Y_train = np_utils.to_categorical(y_train)

    print('evaluating test data....')
    P_test = model.predict_classes(X_train, verbose=0)
    (precision, recall, accuracy, F1) = perf_measure(y_true=y_train, y_pred=P_test)
    print("Precision: %s Recall: %s Accuracy: %s F1: %s" % (precision, recall, accuracy, F1))

    ## load the mixture regression coefficients.
    param_file = 'final_parameters.RData'
    robjects.r['load'](param_file)
    Z = np.asarray((robjects.r['final_params'][0]))
    Beta = np.asarray((robjects.r['final_params'][1]))
    pp = model.predict(X_train)[:, 1]
    a = np.max(pp[np.where(pp<0.5)])
    seed_idx = np.where(pp==a)[0]

    ## print the positive important features
    for i in np.unique(Z):
        print (i)
        fea = np.argsort(Beta[:, (Z[i]-1)])
        # fea = np.argsort(np.abs(Beta[:, (Z[i]-1)])) ## top positive and negative important features
        print fea

    ## fidelity test
    n_pos = 0
    n_new = 0
    n_neg = 0

    n_pos_rand = 0
    n_new_rand = 0
    n_neg_rand = 0
    n = 0

    ## test on all the malware
    idx = np.nonzero(y_train)[0]
    for i in idx:
        if n%100 ==0:
            print n
        data_for_explain = X_train[i,:].reshape(1, 135)
        n = n + 1

        fea = fea[0:n_fea_select]
        xai_test = xai_mlp(model, data_for_explain, fea, n_fea_select)

        fid_tt = fid_test(xai_test)
        test_data, P1, P2 = fid_tt.fea_deduct_test(n_fea_select)
        if P1 > 0.5:
            n_pos = n_pos + 1
        if P2 > 0.5:
            n_pos_rand = n_pos_rand + 1

        test_data, P_test_1, P_test_2 = fid_tt.syn_test(n_fea_select)
        if P_test_1> 0.5:
            n_new = n_new + 1
        if P_test_2 > 0.5:
            n_new_rand = n_new_rand + 1

        test_seed = X_train[seed_idx, ]
        neg_test_data, P_neg_1, P_neg_2 = fid_tt.fea_aug_test(test_seed, n_fea_select)
        if P_neg_1 > 0.5:
            n_neg = n_neg + 1
        if P_neg_2 > 0.5:
            n_neg_rand = n_neg_rand + 1

    print n
    print 'Our method'
    print 'Acc pos:', float(n_pos)/n
    print 'Acc new:', float(n_new)/n
    print 'Acc neg:', float(n_neg)/n

    print 'Random'
    print 'Acc pos:', float(n_pos_rand) / n
    print 'Acc new:', float(n_new_rand) / n
    print 'Acc neg:', float(n_neg_rand) / n
