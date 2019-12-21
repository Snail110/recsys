import numpy as np
import os
import pickle
from collections import Counter

"""
Data Process for FM, PNN, and DeepFM.
"""
def get_train_test_file(file_path, feat_dict_, split_ratio=0.9):
    #定义训练集与测试集
    train_label_fout = open('train_label', 'w')
    train_value_fout = open('train_value', 'w')
    train_idx_fout = open('train_idx', 'w')
    test_label_fout = open('test_label', 'w')
    test_value_fout = open('test_value', 'w')
    test_idx_fout = open('test_idx', 'w')

    continuous_range_ = range(1, 14)
    categorical_range_ = range(14, 40)

    def process_line_(line):
        features = line.rstrip('\n').split('\t')
        feat_idx, feat_value, label = [], [], []
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

        # 处理分类型数据
        for idx in categorical_range_:
            if features[idx] == '' or features[idx] not in feat_dict_:
                feat_idx.append(0)
                feat_value.append(0.0)
            else:
                feat_idx.append(feat_dict_[features[idx]])
                feat_value.append(1.0)
        return feat_idx, feat_value, [int(features[0])]

    with open(file_path, 'r') as fin:
        for line_idx, line in enumerate(fin):
            feat_idx, feat_value, label = process_line_(line)

            feat_value = '\t'.join([str(v) for v in feat_value]) + '\n'
            feat_idx = '\t'.join([str(idx) for idx in feat_idx]) + '\n'
            label = '\t'.join([str(idx) for idx in label]) + '\n'

            if np.random.random() <= split_ratio:
                train_label_fout.write(label)
                train_idx_fout.write(feat_idx)
                train_value_fout.write(feat_value)
            else:
                test_label_fout.write(label)
                test_idx_fout.write(feat_idx)
                test_value_fout.write(feat_value)

        fin.close()

    train_label_fout.close()
    train_idx_fout.close()
    train_value_fout.close()
    test_label_fout.close()
    test_idx_fout.close()
    test_value_fout.close()


def get_feat_dict(file_path):
    freq_ = 10
    # pkl2格式用来保存字典形式的数据pickle
    dir_feat_dict_ = 'feat_dict_' + str(freq_) + '.pkl2'
    continuous_range_ = range(1, 14)
    categorical_range_ = range(14, 40)

    if os.path.exists(dir_feat_dict_):
        feat_dict = pickle.load(open(dir_feat_dict_, 'rb'))
    else:
        # print('generate a feature dict')
        # Count the number of occurrences of discrete features
        feat_cnt = Counter()
        with open(file_path, 'r') as fin:
            for line_idx, line in enumerate(fin):
                features = line.rstrip('\n').split('\t')
                for idx in categorical_range_:
                    if features[idx] == '': continue
                    feat_cnt.update([features[idx]])

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
        with open(file_path, 'r') as fin:
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
        with open(dir_feat_dict_, 'wb') as fout:
            pickle.dump(feat_dict, fout)
        print('args.num_feat ', len(feat_dict) + 1)

    return feat_dict


if __name__ == '__main__':
    file_path = './train.txt'
    feat_dict = get_feat_dict(file_path)
    get_train_test_file(file_path, feat_dict)
    print('Done!')