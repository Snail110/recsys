import numpy as np
import os
import pickle
from collections import Counter

"""
Data Process for Wide-Deep network
https://github.com/busesese/Wide_Deep_Model
https://github.com/aviraj-sinha/ML5/blob/master/10.%20Keras%20Wide%20and%20Deep.ipynb
"""
def get_train_test_file(file_path, feat_dict_, split_ratio=0.9):
    #定义训练集与测试集
    train_label_fout = open(file_path+'traincross_label', 'w')
    train_value_fout = open(file_path+'traincross_value', 'w')
    train_idx_fout = open(file_path+'traincross_idx', 'w')
    train_num_fout = open(file_path + 'traincross_num', 'w')

    continuous_range_ = range(1, 14)
    categorical_range_ = range(14, 52)

    def process_line_(line):
        features = line.rstrip('\n').split('\t')
        feat_idx, feat_value, label= [], [], []
        # 自己获取每列特征中的最大值，最小值
        cont_min_ = [0.0, -2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        cont_max_ = [95.0,7864,8457.0,87.0,1015215.0,4638.0,1658.0,547.0,5637.0,4.0,37.0,98.0,770.0]
        cont_diff_ = [cont_max_[i] - cont_min_[i] for i in range(len(cont_min_))]
        # MinMax Normalization
        for idx in continuous_range_:
            if features[idx] == '':
                feat_idx.append(0)
                feat_value.append(0.0)
            else:
                feat_idx.append(feat_dict_[idx])
                feat_value.append(round((float(features[idx]) - cont_min_[idx - 1]) / cont_diff_[idx - 1], 6))
        # 获取数值型特征
        num = feat_value[:]
        # 处理分类型数据
        for idx in categorical_range_:
            if features[idx] == '' or features[idx] not in feat_dict_:
                feat_idx.append(0)
                feat_value.append(0.0)
            else:
                feat_idx.append(feat_dict_[features[idx]])
                feat_value.append(1.0)
        return feat_idx, feat_value, [int(features[0])], num

    with open(file_path+'traincross.txt', 'r') as fin:
        for line_idx, line in enumerate(fin):
            feat_idx, feat_value, label, num = process_line_(line)

            feat_value = '\t'.join([str(v) for v in feat_value]) + '\n'
            feat_idx = '\t'.join([str(idx) for idx in feat_idx]) + '\n'
            label = '\t'.join([str(idx) for idx in label]) + '\n'
            feat_num = '\t'.join([str(idx) for idx in num]) + '\n'

            train_label_fout.write(label)
            train_idx_fout.write(feat_idx)
            train_value_fout.write(feat_value)
            train_num_fout.write(feat_num)

        fin.close()

    train_label_fout.close()
    train_idx_fout.close()
    train_value_fout.close()
    train_num_fout.close()


def cross_feature(file_path,cross_range):
    # 构建交叉特征数据集
    traincross = open(file_path+'traincross.txt', 'w')
    with open(file_path+'train.txt', 'r') as fin:
        for line_idx, line in enumerate(fin):
            features = line.rstrip('\n').split('\t')
            for i in cross_range:
                features.append('_'.join([features[i[0]], features[i[1]]]))
            string_features = '\t'.join(features) + '\n'
            traincross.write(string_features)
        fin.close()
    traincross.close()

def get_feat_dict(file_path):

    freq_ = 10
    # pkl2格式用来保存字典形式的wide-deep数据pickle
    dir_feat_dict_ = file_path+'cross_feat_dict_' + str(freq_) + '.pkl2'
    continuous_range_ = range(1, 14)
    categorical_range_ = range(14, 52)

    if os.path.exists(dir_feat_dict_):
        feat_dict = pickle.load(open(dir_feat_dict_, 'rb'))
    else:
        # print('generate a feature dict')
        # Count the number of occurrences of discrete features
        feat_cnt = Counter()
        with open(file_path+'traincross.txt', 'r') as fin:
            for line_idx, line in enumerate(fin):
                features = line.rstrip('\n').split('\t')
                for idx in categorical_range_:
                    if features[idx] == '': continue
                    feat_cnt.update([features[idx]])
            fin.close()
        # Only retain discrete features with high frequency
        dis_feat_set = set()
        for feat, ot in feat_cnt.items():
            if ot >= freq_:
                dis_feat_set.add(feat)

        # Create a dictionary for continuous and discrete features
        feat_dict = {}
        tc = 1
        # Continuous features
        for idx in continuous_range_:
            feat_dict[idx] = tc
            tc += 1
        # Discrete features
        cnt_feat_set = set()
        with open(file_path+'traincross.txt', 'r') as fin:
            for line_idx, line in enumerate(fin):
                features = line.rstrip('\n').split('\t')

                for idx in categorical_range_:
                    if features[idx] == '' or features[idx] not in dis_feat_set:
                        continue
                    if features[idx] not in cnt_feat_set:
                        cnt_feat_set.add(features[idx])
                        feat_dict[features[idx]] = tc
                        tc += 1
            # Save dictionary
            fin.close()
        with open(dir_feat_dict_, 'wb') as fout:
            pickle.dump(feat_dict, fout)
        print('args.num_feat ', len(feat_dict) + 1)
    return feat_dict


if __name__ == '__main__':
    file_path = '../data/Criteo/'
    # 交叉特征
    cross_range = [[14, 15], [16, 17], [18, 19], [20, 21], [22, 23], [24, 25], [26, 27], [28, 29], [30, 31],
                   [32, 33], [34, 35], [36, 37], [38, 39]]
    cross_feature(file_path,cross_range)
    feat_dict = get_feat_dict(file_path)
    get_train_test_file(file_path, feat_dict)
    print('Done!')