import os
#os.environ["CUDA_VISIBLE_DEVICES"]="2,3"
import torch
from torch.utils.data import SequentialSampler, DataLoader
import numpy as np
from config import parse_args
from data_helper import MultiModalDataset
from category_id_map import lv2id_to_category_id
from model import MultiModal111
from model_pretrain import QQUniModel
from tqdm import tqdm

def inference():
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

    all_kfold_logits = []

    for kfold_idx in tqdm(range(1, 6)):
        kfold_logits = []
        kfold_model_path = f'./save/kfold_model/{kfold_idx}fold_model/'
        model_list = os.listdir(kfold_model_path)
        model_list = [(i, int(i.split('_')[2])) for i in model_list]
        model_list.sort(key=lambda x: x[1], reverse=True)
        best_model_path = kfold_model_path + model_list[0][0]


        # 2. load model
        #model = MultiModal111(args)
        model = QQUniModel(args, task=['tag'], init_from_pretrain=True)
        print('使用模型：', best_model_path)
        checkpoint = torch.load(best_model_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        if torch.cuda.is_available():
            model = torch.nn.parallel.DataParallel(model.cuda())
        model.eval()

        # 3. inference
        # predictions = []
        with torch.no_grad():
            for batch in tqdm(dataloader):
                logits, pred_label_id = model(batch, inference=True, kfold_inference=True)
                kfold_logits.append(logits)
        
        kfold_logits = torch.vstack(kfold_logits)
        print(kfold_logits.shape)
        all_kfold_logits.append(kfold_logits)


        # 4. dump results
        # test_output_csv = f'./data/result_{kfold_idx}fold.csv'
        # with open(test_output_csv, 'w') as f:
        #     for pred_label_id, ann in zip(predictions, dataset.anns):
        #         video_id = ann['id']
        #         category_id = lv2id_to_category_id(pred_label_id)
        #         f.write(f'{video_id},{category_id}\n')


    save_logits = np.array([i.cpu().numpy() for i in all_kfold_logits])
    print('save_logits shape: ', save_logits.shape)
    np.save("../kfold_logits/kfold_logits_reberta_mlm_mfm_20ep_4_testb.npy", save_logits)


    print('-' * 10, '开始用5折模型预测logits取平均', '-' * 10)
    print('结束')

    res = torch.zeros((25000, 200)).to(kfold_logits.device)
    for i in range(5):
        res += all_kfold_logits[i]

    res = res/5
    print('res.shape: ', res.shape)
    pred_label_id = torch.argmax(res, dim=1)
    predictions = []
    predictions.extend(pred_label_id.cpu().numpy())
    test_output_csv = f'../../data/result_logits_avg_testb.csv'
    with open(test_output_csv, 'w') as f:
        for pred_label_id, ann in zip(predictions, dataset.anns):
            video_id = ann['id']
            category_id = lv2id_to_category_id(pred_label_id)
            f.write(f'{video_id},{category_id}\n')




if __name__ == '__main__':
    print('-'*10, '开始用5折模型做预测' ,'-'*10)
    inference()
    print('预测结束')

