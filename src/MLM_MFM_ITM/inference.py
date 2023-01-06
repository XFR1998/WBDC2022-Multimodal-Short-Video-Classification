import torch
from torch.utils.data import SequentialSampler, DataLoader

from myconfig.config import parse_args
from data.pre_data_helper import MultiModalDataset
from myconfig.category_id_map import lv2id_to_category_id
from pretrain_model.my_nui_model import MultiModal
from tqdm import tqdm
import os
import numpy as np
#os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"

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

    # 2. load model
    model = MultiModal(args, task=['tag'])
    checkpoint = torch.load(args.ckpt_file, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    if torch.cuda.is_available():
        model = torch.nn.parallel.DataParallel(model.cuda())
    model.eval()

    # 3. inference
    predictions = []
    with torch.no_grad():
        logits = []
        for batch in tqdm(dataloader):
            logit, pred_label_id = model(batch, inference=True, return_logits=True)
            logits.append(logit)
            predictions.extend(pred_label_id.cpu().numpy())
        logits = torch.vstack(logits)
        print(logits.shape)
        savelogits = np.array(logits.cpu().numpy())
        print(savelogits.shape)
        # np.save("6698_jjj.npy", savelogits)

    # 4. dump results
    with open(args.test_output_csv, 'w') as f:
        for pred_label_id, ann in zip(predictions, dataset.anns):
            video_id = ann['id']
            category_id = lv2id_to_category_id(pred_label_id)
            f.write(f'{video_id},{category_id}\n')


if __name__ == '__main__':
    inference()
