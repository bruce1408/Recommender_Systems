"""
Preprocess Criteo dataset. This dataset was used for the Display Advertising
Challenge (https://www.kaggle.com/c/criteo-display-ad-challenge).
"""
import os
# import sys
# import click
import random
import collections

# There are 13 integer features and 26 categorical features
import pandas as pd

continous_features = range(1, 14)
categorial_features = range(14, 40)

# Clip integer features. The clip point for each integer feature
# is derived from the 95% quantile of the total values in each feature
continous_clip = [20, 600, 100, 50, 64000, 500, 100, 50, 500, 10, 10, 10, 50]


class CategoryDictGenerator:
    """
    Generate dictionary for each of the categorical features
    """

    def __init__(self, num_feature):
        self.dicts = []
        self.cout = 0
        self.num_feature = num_feature #  26
        for i in range(0, num_feature):
            self.dicts.append(collections.defaultdict(int))

    def build(self, datafile, categorial_features, cutoff=0):
        with open(datafile, 'r') as f:
            for line in f:
                self.cout +=1
                features = line.rstrip('\n').split('\t')
                for i in range(0, self.num_feature):
                    if features[categorial_features[i]] != '':
                        self.dicts[i][features[categorial_features[i]]] += 1 # 所有类别数据出现的次数
        for i in range(0, self.num_feature):
            self.dicts[i] = filter(lambda x: x[1] >= cutoff, self.dicts[i].items()) # 大于等于数字cutoff进行排序
            self.dicts[i] = sorted(self.dicts[i], key=lambda x: (-x[1], x[0]))
            vocabs, _ = list(zip(*self.dicts[i])) # bug 问题
            self.dicts[i] = dict(zip(vocabs, range(1, len(vocabs) + 1)))  # 按照出现次数进行排序之后然后重新进行构建，value是1-n的名次
            self.dicts[i]['<unk>'] = 0

    def gen(self, idx, key):
        if key not in self.dicts[idx]:
            res = self.dicts[idx]['<unk>']
        else:
            res = self.dicts[idx][key]
        return res

    def dicts_sizes(self):
        return [len(self.dicts[idx]) for idx in range(0, self.num_feature)]


class ContinuousFeatureGenerator:
    """
    Clip continuous features.
    """

    def __init__(self, num_feature):
        self.num_feature = num_feature

    def build(self, datafile, continous_features):
        with open(datafile, 'r') as f:
            for line in f:
                features = line.rstrip('\n').split('\t')
                for i in range(0, self.num_feature):
                    val = features[continous_features[i]]
                    if val != '':
                        val = int(val)
                        if val > continous_clip[i]:
                            val = continous_clip[i]


    def readCsv(self, csvpath, continous_features):
        df = pd.read_csv(csvpath)
        for index, i in enumerate(df.columns[1: 14]):
            df.iloc[df[i] > continous_clip[index]] = continous_clip[index]
            df.iloc[:, 1: 14].astype('int')

    def gen(self, idx, val):
        if val == '':
            return 0.0
        val = float(val)
        return val


def preprocess(datadir, outdir, num_train_sample = 900, num_test_sample = 900):
    """
    All the 13 integer features are normalzied to continous values and these
    continous features are combined into one vecotr with dimension 13.
    Each of the 26 categorical features are one-hot encoded and all the one-hot
    vectors are combined into one sparse binary vector.
    """
    dists = ContinuousFeatureGenerator(len(continous_features))
    dists.build(os.path.join(datadir, 'train.txt'), continous_features)

    dicts = CategoryDictGenerator(len(categorial_features))
    dicts.build(os.path.join(datadir, 'train.txt'), categorial_features, cutoff=2)

    dict_sizes = dicts.dicts_sizes()
    with open(os.path.join(outdir, 'feature_sizes.txt'), 'w') as feature_sizes:
        sizes = [1] * len(continous_features) + dict_sizes
        sizes = [str(i) for i in sizes]
        feature_sizes.write(','.join(sizes))

    random.seed(0)

    # Saving the data used for training.
    with open(os.path.join(outdir, 'train.txt'), 'w') as out_train:
        with open(os.path.join(datadir, 'train.txt'), 'r') as f:
            for line in f.readlines()[:num_train_sample]:
                features = line.rstrip('\n').split('\t')

                continous_vals = []
                for i in range(0, len(continous_features)):
                    val = dists.gen(i, features[continous_features[i]])
                    continous_vals.append("{0:.6f}".format(val).rstrip('0').rstrip('.'))
                categorial_vals = []
                for i in range(0, len(categorial_features)):
                    val = dicts.gen(i, features[categorial_features[i]])
                    categorial_vals.append(str(val))

                continous_vals = ','.join(continous_vals)
                categorial_vals = ','.join(categorial_vals)
                label = features[0]
                out_train.write(','.join([continous_vals, categorial_vals, label]) + '\n')
                    

    with open(os.path.join(outdir, 'test.txt'), 'w') as out:
        with open(os.path.join(datadir, 'test.txt'), 'r') as f:
            for line in f.readlines()[:num_test_sample]:
                features = line.rstrip('\n').split('\t')

                continous_vals = []
                for i in range(0, len(continous_features)):
                    val = dists.gen(i, features[continous_features[i] - 1])
                    continous_vals.append("{0:.6f}".format(val).rstrip('0').rstrip('.'))
                categorial_vals = []
                for i in range(0, len(categorial_features)):
                    val = dicts.gen(i, features[categorial_features[i] - 1])
                    categorial_vals.append(str(val))

                continous_vals = ','.join(continous_vals)
                categorial_vals = ','.join(categorial_vals)
                out.write(','.join([continous_vals, categorial_vals]) + '\n')

if __name__ == "__main__":
    preprocess('../data/raw', '../data')