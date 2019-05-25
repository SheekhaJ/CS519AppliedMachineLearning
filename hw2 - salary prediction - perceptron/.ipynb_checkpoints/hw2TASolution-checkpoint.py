#!/usr/bin/env python

from __future__ import division
import sys
import numpy as np
import time
from collections import defaultdict
from itertools import combinations

def normalization(data_X, mean = False, variance = False, norm_para=None):
    data_X = np.array(data_X)
    if norm_para is None:    # calc for training data
        feat_mean = np.mean(data_X,0)   # mean along 0 axis (vertical)
        feat_std  = np.std(data_X,0)
        feat_std[feat_std < 1e-10] = 1.0    # in case some features not appear in dev or test data, resulting 0-std in normalization
    else:               # use scales from training data
        feat_mean, feat_std = norm_para

    if mean:
        data_X = data_X - feat_mean
    if variance:
        data_X = data_X / feat_std
    data_X[:,0] = 1.0    # resume the bias feature to 1

    return data_X, feat_mean, feat_std

def map_data(filename, feature2index, mean = False, variance = False, norm_para = None):
    data_X = []
    data_Y, last = (None, None) if 'blind' in filename else ([], -1)    # last is the position of last valid column
    dimension = len(feature2index)
    for j1, line in enumerate(open(filename)):
        line = line.strip()
        features = line.split(", ")
        feat_vec = np.zeros(dimension)
        for i, fv in enumerate(features[:last]):    # train/dev: last one is target; test: last is also a col in feature
            if (i, fv) in feature2index:            # ignore unobserved features
                feat_vec[feature2index[i, fv]] = 1.0

        # engineering for feature combination
        for i, j in combinations(comb, 2):
            if (i, features[i], j, features[j]) in feature2index:     # ignore unobserved combined features
                feat_vec[feature2index[i, features[i], j, features[j]]] = 1.0

        # engineering for 2 numerical features addition
        if numerical: 
            feat_vec = np.append(feat_vec, [float(features[0]), float(features[7])])  # numerical age, numerical hours  
        data_X.append(feat_vec)
        if last is not None: 
            data_Y.append(1 if features[-1] == ">50K" else -1)

    # normalization part
    data_X, feat_mean, feat_std = normalization(data_X, mean = mean, variance = variance, norm_para=norm_para)

    return data_X, data_Y, (feat_mean, feat_std)
    

def train(train_data, dev_data, it = 5, check_freq = 5000, smart_avg = False):
    train_size = len(train_data)
    dimension = len(train_data[0][0])
    model = np.zeros(dimension)
    totmodel = np.zeros(dimension)
    c, smart_tot = 0, np.zeros(dimension)
    best_err_rate = best_err_rate_avg = best_positive = best_positive_avg = 1
    t = time.time()
    for i in range(it):
        updates = 0
        for j, (vecx, y) in enumerate(train_data, start = 1):
            c += 1
            if model.dot(vecx) * y <= 0:
                updates += 1
                model += y * vecx
                if smart_avg:
                    smart_tot += c * y * vecx
            if not smart_avg:
                totmodel += model
            if (j+i*train_size) % check_freq == 0:
                dev_err_rate, positive = test_dev(dev_data, model)
                dev_err_rate_avg, positive_avg = test_dev(dev_data,  model - smart_tot/c if smart_avg else totmodel)
                epoch_position = i + j/train_size

                if dev_err_rate < best_err_rate:        # update a better error
                    best_err_rate = dev_err_rate
                    best_err_pos = epoch_position #(i, j)
                    best_positive = positive
                if dev_err_rate_avg < best_err_rate_avg:
                    best_err_rate_avg = dev_err_rate_avg
                    best_err_pos_avg = epoch_position #(i, j)
                    best_positive_avg = positive_avg
                    best_avg_model = model - smart_tot/c if smart_avg else totmodel.copy() #copy() is important
                print("unavg, epoch {} updates {} ({:.1%}) dev_err {:.1%} (+:{:.1%});   ".format(i+1,
                                                            updates,
                                                            updates/train_size,
                                                            dev_err_rate,
                                                            positive), \
                "avg, epoch {} dev_err {:.1%} (+:{:.1%})".format(i+1,
                                                            dev_err_rate_avg,
                                                            positive_avg))
    print("training time {:.5f} s".format(time.time()-t))
    return best_avg_model/it/len(train_data)

def test_dev(data, model):
    errors = sum(model.dot(vecx) * y <= 0 for vecx, y in data)
    positives = sum(model.dot(vecx) > 0 for vecx, _ in data)
    return errors / len(data), positives / len(data)

def predict(test_data, model):
    return [">50K" if model.dot(vecx) > 0 else "<=50K" for vecx in test_data ]

def create_feature_map(train_file):
    column_values = defaultdict(set)
    for line in open(train_file):
        features = line.strip().split(", ")[:-1] # last field is target.
        for i, fv in enumerate(features):
            # if i in numerical: continue    # uncommand to keep only binarized/numerical features for age and hours, command out to keep both
            column_values[i].add(fv)
    feature2index = {(-1, 'bias'): 0} # bias
    index2feature = {0: ('col-1', 'bias')}
    for i, values in column_values.items():
        for v in values:
            feature2index[i, v] = len(feature2index)
            index2feature[len(feature2index) - 1] = ('col'+str(i),v)
    # engineering for feature combination
    for i, j in combinations(comb, 2):
        for v1 in column_values[i]:
            for v2 in  column_values[j]:
                feature2index[i, v1, j, v2] = len(feature2index)
                index2feature[len(feature2index) - 1] = ('col'+str(i),v1,'col'+str(j),v2)
    dimension = len(feature2index) + len(numerical)
    print("dimensionality: ", dimension)
    return feature2index, index2feature


def experiment(train_file, dev_file, test_file = '', it = 1, check_freq = 5000, feat_detail = False, mean = False, variance = False):
    feature2index,index2feature = create_feature_map(train_file)
    X1, Y1, norm = map_data(train_file, feature2index, mean, variance)
    train_data   = list(zip(X1, Y1))
    X2, Y2, _    = map_data(dev_file, feature2index, mean, variance, norm_para=norm)
    dev_data     = list(zip(X2, Y2))
    
    model = train(train_data, dev_data, it, check_freq)

    print("train_err {:.2%} (+:{:.1%})".format(*test_dev(train_data, model)))

    # display feature details (top 5 pos weights, top5 neg weights, etc)
    if feat_detail: 
        print([(index2feature[i], '{:.5}'.format(model[i])) for i in model.argsort()[-5:][::-1]])
        print([(index2feature[i], '{:.5}'.format(model[i])) for i in model.argsort()[:5]])
        print([(i, model[feature2index[-1,i]]) for i in ['bias']])
        print([(i, model[feature2index[6,i]]) for i in ['Male', 'Female']])

    # predict blind test if test_file name is assigned
    if test_file != '':
        test_data_X, _, _  = map_data(test_file, feature2index, mean, variance, norm_para=norm)
        labels = predict(test_data_X, model)
        positive = sum([1 for x in labels if x == '>50K'])/len(labels)
        fout = open(test_file+'_pred','w')
        for j, line in enumerate(open(test_file)):
            fout.write(line[:-1]+', '+labels[j]+'\n')
        fout.close()
        print('(+:{:.1%}), prediction written to {}'.format(positive, test_file+'_pred'))
 
def exp():
    print("{}\nPerceptron and Averaged Perceptron".format('-'*N_dash))
    experiment(train_file, dev_file,it=it,feat_detail=False,mean=mean_,variance=variance_)

if __name__ == "__main__":

    if len(sys.argv) > 1:
        train_file, dev_file = sys.argv[1], sys.argv[2]
    else:
        train_file, dev_file = "./hw1-data/income.train.txt.5k", "./hw1-data/income.dev.txt"
    test_file = "./hw1-data/income.test.blind"

    mean_ = True
    variance_ = False
    it = 5

    N_dash = 40

    # here are tunnung on numerical features and combinations
    numerical = []#[0, 7]
    # numerical = [0,7]
    comb = []
    # comb=[4,6]
    comb = [4,5,8]
    # comb = [0,5,6,7,8]
    # comb = [0,1,2,3,4,5,6,7,8]
    exp()
