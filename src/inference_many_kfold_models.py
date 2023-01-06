import os
#os.environ["CUDA_VISIBLE_DEVICES"]="1"
import torch
from torch.utils.data import SequentialSampler, DataLoader
import numpy as np
from config import parse_args
from data_helper import MultiModalDataset
from category_id_map import lv2id_to_category_id
from model import MultiModal111
from tqdm import tqdm


def inference():
    use_5fold = True
    use_1fold = False



    args = parse_args()
    # 1. load data
    dataset = MultiModalDataset(args, args.test_annotation, args.test_zip_feats, test_mode=True)
    sampler = SequentialSampler(dataset)
    dataloader = DataLoader(dataset,
                            batch_size=args.test_batch_size,
                            sampler=sampler,
                            drop_last=False,
                            pin_memory=True,
                            num_workers=args.num_workers,
                            prefetch_factor=args.prefetch)

    if use_5fold:
        # ----------------------------5折------------------------------------
        logits_path = './kfold_logits/5折/'
        logits_list = os.listdir(logits_path)
        print('5折一共有如下logits：', logits_list)


        res = torch.zeros((25000, 200))

        for logits_npy in tqdm(logits_list):
            logits = np.load(logits_path+logits_npy)
            print('logits shape: ', logits.shape)
            logits = logits.tolist()
            logits = [torch.tensor(i) for i in logits]

            for i in range(5):
                res += logits[i]

        # res = res/(len(logits_list)*5)
    # -------------------------------------------------------------------

    # ---------------------------单模----------------------------------------------
    if use_1fold:

        single_path = './kfold_logits/单模/'
        single_list = os.listdir(single_path)
        print('单模一共有如下logits：', single_list)

        single_res = torch.zeros((25000, 200))
        for logits_npy in tqdm(single_list):
            logits = np.load(single_path + logits_npy)
            logits = torch.tensor(logits)
            print('logits shape: ', logits.shape)
            # logits = logits.tolist()
            # logits = [torch.tensor(i) for i in logits]

            single_res += logits[i]
        # single_res = single_res / len(single_list)

    # --------------------------------------------------------------------------

    if use_1fold and use_5fold:
        res = (res+single_res)/((len(logits_list)*5)+len(single_list))
    elif use_5fold:
        res = res/(len(logits_list)*5)
    elif use_1fold:
        res = single_res / len(single_list)








    print('res.shape: ', res.shape)
    pred_label_id = torch.argmax(res, dim=1)
    predictions = []
    predictions.extend(pred_label_id.numpy())
    test_output_csv = f'../data/result.csv'
    with open(test_output_csv, 'w') as f:
        for pred_label_id, ann in zip(predictions, dataset.anns):
            video_id = ann['id']
            category_id = lv2id_to_category_id(pred_label_id)
            f.write(f'{video_id},{category_id}\n')



if __name__ == '__main__':
    print('-'*10, '开始用many折模型做预测' ,'-'*10)
    inference()
    print('预测结束')

