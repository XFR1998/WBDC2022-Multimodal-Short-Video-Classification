#%%
import json
import zipfile
import torch
import numpy as np
from io import BytesIO
import random
from tqdm import tqdm
import pandas as pd
from sklearn.model_selection import (train_test_split,
									KFold,
									StratifiedKFold,
                                    StratifiedShuffleSplit,
                                    ShuffleSplit,
                                    GroupShuffleSplit,
                                    GroupKFold,
                                    TimeSeriesSplit)

#%%

# load annotations
with open('./data/annotations/labeled.json', 'r', encoding='utf8') as f:
    anns = json.load(f)
print('总数据长度：', len(anns))

#%%
labels = []
for info in anns:
    labels.append(info['category_id'])
print('总标签长度：', len(labels))
#%%

#%%
# x = [[1,1],
#      [2,2],
#      [3,3],
#      [1,1],
#      [1,1],
#      [1,1],
#      [2,2],
#      [3,3],
#      [1,1],
#      [1,1]]
# y = ['1','2','3','1','1','1','2','3','1','1']

x = np.array(anns)
# x = anns
y = np.array(labels)

skf = StratifiedKFold(n_splits=5)
idx = 0
for train_index, test_index in skf.split(x, y):
    idx += 1
    # 得到的train_index, test_index是索引
    x_train, x_test = x[train_index], x[test_index]
    x_train = x_train.tolist()
    x_test = x_test.tolist()
    # kfold_data = np.vstack((x_train, x_test)).tolist()
    with open('./data/kfold_data/{}_fold_train.json'.format(idx), 'w') as f:
        json.dump(x_train, f)
    with open('./data/kfold_data/{}_fold_val.json'.format(idx), 'w') as f:
        json.dump(x_test, f)




#%%
print('输出检测下切分的k折数据是否合理:   ')
for idx in range(1,6):
    print('{} fold数据长度'.format(idx))
    with open('./data/kfold_data/{}_fold_train.json'.format(idx), 'r') as f:
        kfold_anns = json.load(f)
    print('train data len: ', len(kfold_anns))

    with open('./data/kfold_data/{}_fold_val.json'.format(idx), 'r') as f:
        kfold_anns = json.load(f)
    print('val data len: ', len(kfold_anns))
