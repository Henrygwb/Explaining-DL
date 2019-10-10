import os
# os.environ["THEANO_FLAGS"] = "device=gpu,floatX=float32"
import numpy as np
import cPickle as pickle
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from scipy import io
from rpy2 import robjects
from rpy2.robjects.packages import importr
import rpy2.robjects.numpy2ri
from keras.models import load_model
import argparse

np.random.seed(1234)

r = robjects.r
rpy2.robjects.numpy2ri.activate()

importr('genlasso')
importr('gsubfn')

def perf_measure(y_true, y_pred, x):
    TP_FN = np.count_nonzero(y_true)
    FP_TN = y_true.shape[0] * y_true.shape[1] - TP_FN
    FN = np.where((y_true - y_pred) == 1)[0].shape[0]
    TP = TP_FN - FN
    FP = np.count_nonzero(y_pred) - TP
    TN = FP_TN - FP
    Precision = float(float(TP) / float(TP + FP + 1e-9))
    Recall = float(float(TP) / float((TP + FN + 1e-9)))
    accuracy = float(float(TP + TN) / float((TP_FN + FP_TN + 1e-9)))
    F1 = 2 * ((Precision * Recall) / (Precision + Recall + 1e-9))
    FN_1 = FN
    aa = np.where((y_true - y_pred) == 1)
    n_wrong = 0
    for i in np.unique(aa[0]):
        idx_Col = np.where(aa[0] == i)
        idx_Row = aa[1][idx_Col]
        for j in xrange(len(idx_Row)):
            if idx_Row[j] <= 10 or x[i, idx_Row[j]] == 256:
                n_wrong = n_wrong + 1
    FN_2 = FN_1 - n_wrong
    return Precision, Recall, accuracy, TP, FN_1, FN_2, FP, TN


class xai_rnn(object):
    """class for explaining the rnn prediction"""

    def __init__(self, model, data, start_binary, real_start_sp, cls):
        """
        Args:
            model: target rnn model.
            data: data sample needed to be explained.
            start: value of function start.
        """
        self.model = model
        self.data = data
        self.seq_len = data.shape[1]
        self.seq_len = data[0].shape[0]
        self.start = start_binary
        self.real_sp = real_start_sp
        self.pred = self.model.predict(self.data, verbose=0)[0, self.real_sp]
        self.cls = cls

    def truncate_seq(self, trunc_len):
        """ Generate truncated data sample
        Args:
            trun_len: the lenght of the truncated data sample.

        return:
            trunc_data: the truncated data samples.
        """
        self.trunc_data_test = np.zeros((1, self.seq_len), dtype=int)
        # self.trunc_data_test = np.ones((1, self.seq_len),dtype = int)
        self.tl = trunc_len
        cen = self.seq_len / 2
        half_tl = trunc_len / 2

        if self.real_sp < half_tl:
            self.trunc_data_test[0, (cen - self.real_sp):cen] = self.data[0, 0:self.real_sp]
            self.trunc_data_test[0, cen:(cen + half_tl + 1)] = self.data[0, self.real_sp:(self.real_sp + half_tl + 1)]

        elif self.real_sp >= self.seq_len - half_tl:
            self.trunc_data_test[0, (cen - half_tl):cen] = self.data[0, (self.real_sp - half_tl):self.real_sp]
            self.trunc_data_test[0, cen:(cen + (self.seq_len - self.real_sp))] = self.data[0, self.real_sp:self.seq_len]

        else:
            self.trunc_data_test[0, (cen - half_tl):(cen + half_tl + 1)] = self.data[0, (self.real_sp - half_tl):(
                        self.real_sp + half_tl + 1)]

        self.trunc_data = self.trunc_data_test[0, (cen - half_tl):(cen + half_tl + 1)]
        return self.trunc_data

    def xai_feature(self, samp_num, option='None'):
        """extract the important features from the input data
        Arg:
            fea_num: number of features that needed by the user
            samp_num: number of data used for explanation
        return:
            fea: extracted features
        """
        cen = self.seq_len / 2
        half_tl = self.tl / 2
        sample = np.random.randint(1, self.tl + 1, samp_num)
        features_range = range(self.tl + 1)
        data_explain = np.copy(self.trunc_data).reshape(1, self.trunc_data.shape[0])
        data_sampled = np.copy(self.data)
        for i, size in enumerate(sample, start=1):
            inactive = np.random.choice(features_range, size, replace=False)
            tmp_sampled = np.copy(self.trunc_data)
            tmp_sampled[inactive] = 0
            tmp_sampled = tmp_sampled.reshape(1, self.trunc_data.shape[0])
            data_explain = np.concatenate((data_explain, tmp_sampled), axis=0)
            data_sampled_mutate = np.copy(self.data)
            if self.real_sp < half_tl:
                data_sampled_mutate[0, 0:tmp_sampled.shape[1]] = tmp_sampled
            elif self.real_sp >= self.seq_len - half_tl:
                data_sampled_mutate[0, (self.seq_len - tmp_sampled.shape[1]): self.seq_len] = tmp_sampled
            else:
                data_sampled_mutate[0, (self.real_sp - half_tl):(self.real_sp + half_tl + 1)] = tmp_sampled
            data_sampled = np.concatenate((data_sampled, data_sampled_mutate), axis=0)

        if option == "Fixed":
            print "Fix start points"
            data_sampled[:, self.real_sp] = self.start
        label_sampled = self.model.predict(data_sampled, verbose=0)[:, self.real_sp, self.cls]
        label_sampled = label_sampled.reshape(label_sampled.shape[0], 1)
        X = r.matrix(data_explain, nrow=data_explain.shape[0], ncol=data_explain.shape[1])
        Y = r.matrix(label_sampled, nrow=label_sampled.shape[0], ncol=label_sampled.shape[1])

        n = r.nrow(X)
        p = r.ncol(X)
        results = r.fusedlasso1d(y=Y, X=X)
        result = np.array(r.coef(results, np.sqrt(n * np.log(p)))[0])[:, -1]

        importance_score = np.argsort(result)[::-1]
        self.fea = (importance_score - self.tl / 2) + self.real_sp
        self.fea = self.fea[np.where(self.fea < 200)]
        self.fea = self.fea[np.where(self.fea >= 0)]
        return self.fea


if __name__ == "__main__":

    n_fea_select = 15
    num_data_fn = 3
    num_data_fp = 4
    epoch = 10
    batch_size = 100

    pret_fea_fn = 2
    pret_fea_fp = 2

    PATH_TRAIN_DATA = "../data/elf_x86_32_gcc_O1_train.pkl"
    PATH_TEST_DATA = "../data/elf_x86_32_gcc_O1_test.pkl"

    print 'load data ...'
    print('**************************')
    data_train = pickle.load(file(PATH_TRAIN_DATA))

    data_num_train = len(data_train[0])
    seq_len = 200
    x_train = pad_sequences(data_train[0], maxlen=seq_len, dtype='int32', padding='post', truncating='post', value=0)
    x_train = x_train + 1
    y = pad_sequences(data_train[1], maxlen=seq_len, dtype='int32', padding='post', truncating='post', value=0)

    y_train = np.zeros((data_num_train, seq_len, 2), dtype=y.dtype)
    for train_id in xrange(data_num_train):
        y_train[train_id, np.arange(seq_len), y[train_id]] = 1

    data_test = pickle.load(file(PATH_TEST_DATA))

    data_num_t = len(data_test[0])
    seq_len = 200
    x_test = pad_sequences(data_test[0], maxlen=seq_len, dtype='int32', padding='post', truncating='post', value=0)
    x_test = x_test + 1
    y_t = pad_sequences(data_test[1], maxlen=seq_len, dtype='int32', padding='post', truncating='post', value=0)
    y_test = np.zeros((data_num_t, seq_len, 2), dtype=y.dtype)
    for test_id in xrange(data_num_t):
        y_test[test_id, np.arange(seq_len), y_t[test_id]] = 1

    print '[Load model...]'
    model = load_model("../model/O1_Bi_Rnn.h5")

    print'evaluating train data....'
    P_train = model.predict_classes(x_train, verbose=0)
    (precision, recall, accuracy, TP, FN_1, FN_2, FP, TN) = perf_measure(y_true=y, y_pred=P_train, x=x_train)
    print("Precision: %s Recall: %s Accuracy: %s TP: %s FN_all: %s FN_true: %s FP: %s TN: %s" %
          (precision, recall, accuracy, TP, FN_1, FN_2, FP, TN))

    print "evaluating test data..."
    P_test = model.predict_classes(x_test, verbose=0)
    (precision, recall, accuracy, TP, FN_1, FN_2, FP, TN) = perf_measure(y_true=y_t, y_pred=P_test, x=x_test)
    print("Precision: %s Recall: %s Accuracy: %s TP: %s FN_all: %s FN_true: %s FP: %s TN: %s" %
          (precision, recall, accuracy, TP, FN_1, FN_2, FP, TN))

    print 'fixing testing error for False negative...................'
    idx = np.nonzero(y_t)[0]
    start_points = np.nonzero(y_t)[1]

    n_FN = 0
    n_wrong = 0
    crafted_data = []
    crafted_label = []

    for i in np.unique(idx):
        idx_Col = np.where(idx == i)
        idx_Row = start_points[idx_Col]
        binary_func_start = x_test[i][idx_Row]
        x_exp = x_test[i:i + 1]

        for j in xrange(len(idx_Row)):
            P = model.predict(x_exp.reshape(1, 200), verbose=0)[0, idx_Row[j], 1]
            if P < 0.5:
                if idx_Row[j] <= 10 or binary_func_start[j] == 256:
                    n_wrong = n_wrong + 1
                else:
                    n_FN = n_FN + 1
                    xai_test = xai_rnn(model, x_exp, binary_func_start[j], idx_Row[j], 0)
                    truncate_seq_data = xai_test.truncate_seq(20)
                    xai_fea = xai_test.xai_feature(500)[0:n_fea_select]
                    for k in xrange(num_data_fn):
                        sample_crafed = np.copy(x_exp)
                        sample_crafed = sample_crafed.reshape(200, )
                        sample_crafed[xai_fea[0:pret_fea_fn]] = [1, 1]
                        sample_crafed[idx_Row[j]] = binary_func_start[j]
                        label_crafted = np.copy(y_test[i,])
                        label_crafted[xai_fea[0:pret_fea_fn], 0] = 1
                        label_crafted[xai_fea[0:pret_fea_fn], 1] = 0
                        crafted_data.append(sample_crafed)
                        crafted_label.append(label_crafted)

    print 'Number of false negative due to sequence truncation.'
    print n_wrong
    print 'Number of ture false negative.'
    print n_FN

    crafted_data = np.array(crafted_data)
    crafted_label = np.array(crafted_label)

    crafted_data = np.concatenate((x_train, crafted_data))
    crafted_label = np.concatenate((y_train, crafted_label))

    print crafted_data.shape
    print crafted_label.shape

    print 'fixing testing error for False positive...................'
    crafted_fp_data = []
    crafted_fp_label = []

    y_p = model.predict_classes(x_test)
    y_er = y_p - y_t
    idx = np.where(y_er == 1)[0]
    wrong_points = np.where(y_er == 1)[1]
    print '>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>'
    print 'Number of false positive.'
    print wrong_points.shape[0]
    print 'OOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO'
    n_neg = 0
    for i in np.unique(idx):
        idx_Col = np.where(idx == i)
        idx_Row = wrong_points[idx_Col]
        binary_func_start = x_test[i][idx_Row]
        x_test_d = x_test[i:i + 1, ].reshape(1, 200)

        for j in xrange(len(idx_Row)):
            P = model.predict(x_test_d.reshape(1, 200), verbose=0)[0, idx_Row[j], 1]
            if idx_Row[j] >= 0:
                n_neg = n_neg + 1
                xai_test = xai_rnn(model, x_test_d, binary_func_start[j], idx_Row[j], 1)

                truncate_seq_data = xai_test.truncate_seq(20)
                xai_fea = xai_test.xai_feature(500)[0:n_fea_select]
                for k in xrange(num_data_fp):
                    sample_crafed = np.copy(x_test_d)
                    sample_crafed = sample_crafed.reshape(200, )
                    sample_crafed[xai_fea[0:pret_fea_fp]] = np.zeros((pret_fea_fp,))
                    label_crafted = np.copy(y_test[i,])
                    label_crafted[xai_fea[0:pret_fea_fp], 0] = 1
                    label_crafted[xai_fea[0:pret_fea_fp], 1] = 0
                    crafted_fp_data.append(sample_crafed)
                    crafted_fp_label.append(label_crafted)

    crafted_fp_data = np.array(crafted_fp_data)
    print crafted_fp_data.shape
    crafted_fp_label = np.array(crafted_fp_label)

    crafted_data = np.concatenate((crafted_data, crafted_fp_data))
    crafted_label = np.concatenate((crafted_label, crafted_fp_label))

    print crafted_data.shape
    print crafted_label.shape

    model.compile('adam', 'categorical_crossentropy', metrics=['accuracy'])
    model.fit(crafted_data, crafted_label, batch_size=batch_size,
              epochs=epoch, verbose=1,
              validation_data=[crafted_data, crafted_label])

    # model.save('../results/O1_Bi_Rnn_fixed.h5')
    print'evaluating train data....'
    P_train = model.predict_classes(x_train, verbose=0)
    (precision, recall, accuracy, TP, FN_1, FN_2, FP, TN) = perf_measure(y_true=y, y_pred=P_train, x=x_train)
    print("Precision: %s Recall: %s Accuracy: %s TP: %s FN_all: %s FN_true: %s FP: %s TN: %s" %
          (precision, recall, accuracy, TP, FN_1, FN_2, FP, TN))

    print "evaluating test data..."
    P_test = model.predict_classes(x_test, verbose=0)
    (precision, recall, accuracy, TP, FN_1, FN_2, FP, TN) = perf_measure(y_true=y_t, y_pred=P_test, x=x_test)
    print("Precision: %s Recall: %s Accuracy: %s TP: %s FN_all: %s FN_true: %s FP: %s TN: %s" %
          (precision, recall, accuracy, TP, FN_1, FN_2, FP, TN))
