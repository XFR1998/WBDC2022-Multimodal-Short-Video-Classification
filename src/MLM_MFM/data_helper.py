import json
import random
import zipfile
from io import BytesIO
from functools import partial

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from transformers import BertTokenizer

from category_id_map import category_id_to_lv2id
from tqdm import tqdm
import jieba
import re
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import PCA


def create_kfold_dataloaders(args=None, fold_idx=None):
    train_data_path = args.kfold_data_path + '{}_fold_train.json'.format(fold_idx)
    val_data_path = args.kfold_data_path + '{}_fold_val.json'.format(fold_idx)
    print('训练集数据路径： ', train_data_path)
    print('验证集数据路径： ', val_data_path)
    train_dataset = MultiModalDataset(args, train_data_path, args.train_zip_feats)
    val_dataset = MultiModalDataset(args, val_data_path, args.train_zip_feats)
    print('训练集数据大小： ', len(train_dataset))
    print('验证集数据大小： ', len(val_dataset))
    # size = len(dataset)
    # val_size = int(size * args.val_ratio)
    # train_dataset, val_dataset = torch.utils.data.random_split(dataset, [size - val_size, val_size],
    #                                                            generator=torch.Generator().manual_seed(args.seed))

    if args.num_workers > 0:
        dataloader_class = partial(DataLoader, pin_memory=True, num_workers=args.num_workers, prefetch_factor=args.prefetch)
    else:
        # single-thread reading does not support prefetch_factor arg
        dataloader_class = partial(DataLoader, pin_memory=True, num_workers=0)

    train_sampler = RandomSampler(train_dataset)
    val_sampler = SequentialSampler(val_dataset)
    train_dataloader = dataloader_class(train_dataset,
                                        batch_size=args.batch_size,
                                        sampler=train_sampler,
                                        drop_last=True)
    val_dataloader = dataloader_class(val_dataset,
                                      batch_size=args.val_batch_size,
                                      sampler=val_sampler,
                                      drop_last=False)
    return train_dataloader, val_dataloader








def create_dataloaders(args):
    dataset = MultiModalDataset(args, args.train_annotation, args.train_zip_feats)
    size = len(dataset)
    val_size = int(size * args.val_ratio)
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [size - val_size, val_size],
                                                               generator=torch.Generator().manual_seed(args.seed))

    if args.num_workers > 0:
        dataloader_class = partial(DataLoader, pin_memory=True, num_workers=args.num_workers, prefetch_factor=args.prefetch)
    else:
        # single-thread reading does not support prefetch_factor arg
        dataloader_class = partial(DataLoader, pin_memory=True, num_workers=0)

    train_sampler = RandomSampler(train_dataset)
    val_sampler = SequentialSampler(val_dataset)
    train_dataloader = dataloader_class(train_dataset,
                                        batch_size=args.batch_size,
                                        sampler=train_sampler,
                                        drop_last=True)
    val_dataloader = dataloader_class(val_dataset,
                                      batch_size=args.val_batch_size,
                                      sampler=val_sampler,
                                      drop_last=False)
    return train_dataloader, val_dataloader






class MultiModalDataset(Dataset):
    """ A simple class that supports multi-modal inputs.
    For the visual features, this dataset class will read the pre-extracted
    features from the .npy files. For the title information, it
    uses the BERT tokenizer to tokenize. We simply ignore the ASR & OCR text in this implementation.
    Args:
        ann_path (str): annotation file path, with the '.json' suffix.
        zip_feats (str): visual feature zip file path.
        test_mode (bool): if it's for testing.
    """

    def __init__(self,
                 args,
                 ann_path: str,
                 zip_feats: str,
                 test_mode: bool = False):
        self.max_frame = args.max_frames
        self.bert_seq_length = args.bert_seq_length
        self.test_mode = test_mode

        self.zip_feat_path = zip_feats
        self.num_workers = args.num_workers
        if self.num_workers > 0:
            # lazy initialization for zip_handler to avoid multiprocessing-reading error
            self.handles = [None for _ in range(args.num_workers)]
        else:
            self.handles = zipfile.ZipFile(self.zip_feat_path, 'r')
        # load annotations
        with open(ann_path, 'r', encoding='utf8') as f:
            self.anns = json.load(f)

        # initialize the text tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(args.bert_dir, use_fast=True, cache_dir=args.bert_cache)




        # 拼接ocr
        print('---------------------拼接ocr-----------------------')
        for i in tqdm(range(len(self.anns))):
            ocr_list = self.anns[i]['ocr']
            ocr_str = '。'.join([i['text'] for i in ocr_list])
            self.anns[i]['ocr'] = ocr_str


        # 去除噪音数据
        # print('-----------------去除噪音数据---------------------------')
        # def clean(text):
        #     # 合并正⽂中过多的空格
        #     text = re.sub(r"\嗯嗯嗯(.*?)", '', text)  #
        #     text = re.sub(r"\哦哦哦(.*?)", '', text)  #
        #     text = re.sub(r"\啊啊啊(.*?)", '', text)  #
        #     text = re.sub(r"\噢噢噢(.*?)", '', text)  #
        #     text = re.sub(r"\呀呀呀(.*?)", '', text)  #
        #     text = re.sub(r"\哟哟哟(.*?)", '', text)  #
        #     return text.strip()
        # for i in tqdm(range(len(self.anns))):
        #     self.anns[i]['title'] = clean(self.anns[i]['title'])
        #     self.anns[i]['asr'] = clean(self.anns[i]['asr'])



        # # 构造tf-idf特征
        # print('---------------------构造tf-idf特征-----------------------')
        #
        # title_text = [jieba_separate_sentence(info['title']) for info in tqdm(self.anns)]
        # # 该类会将文本中的词语转换为词频矩阵，矩阵元素a[i][j] 表示j词在i类文本下的词频
        # vectorizer = CountVectorizer(max_features=128)  # 列数为128
        # # 该类会统计每个词语的tf-idf权值
        # tf_idf_transformer = TfidfTransformer()
        # # 将文本转为词频矩阵并计算tf-idf
        # tf_idf = tf_idf_transformer.fit_transform(vectorizer.fit_transform(title_text))
        #
        # # 将tf-idf矩阵抽取出来，元素a[i][j]表示j词在i类文本中的tf-idf权重
        # title_text_weight = tf_idf.toarray()
        # print('tf-idf特征维度：', title_text_weight.shape)
        #
        # # 使用PCA将tf-idf特征降维
        # print('---------------------使用PCA将tf-idf特征降维-----------------------')
        # pca_weight = PCA(n_components=32).fit_transform(title_text_weight)
        # print('pca特征维度：', pca_weight.shape)
        #
        # for i in tqdm(range(len(self.anns))):
        #     self.anns[i]['tf-idf'] = torch.LongTensor(pca_weight[i])

        # print(len(title_text))
        #
        # print(title_text[:5])
        # assert 1==2



    def __len__(self) -> int:
        return len(self.anns)

    def get_visual_feats(self, idx: int) -> tuple:
        # read data from zipfile
        vid = self.anns[idx]['id']
        if self.num_workers > 0:
            worker_id = torch.utils.data.get_worker_info().id
            if self.handles[worker_id] is None:
                self.handles[worker_id] = zipfile.ZipFile(self.zip_feat_path, 'r')
            handle = self.handles[worker_id]
        else:
            handle = self.handles
        raw_feats = np.load(BytesIO(handle.read(name=f'{vid}.npy')), allow_pickle=True)
        raw_feats = raw_feats.astype(np.float32)  # float16 to float32
        num_frames, feat_dim = raw_feats.shape
        feat = np.zeros((self.max_frame, feat_dim), dtype=np.float32)
        mask = np.ones((self.max_frame,), dtype=np.int32)
        if num_frames <= self.max_frame:
            feat[:num_frames] = raw_feats
            mask[num_frames:] = 0
        else:
            # if the number of frames exceeds the limitation, we need to sample
            # the frames.
            if self.test_mode:
                # uniformly sample when test mode is True
                step = num_frames // self.max_frame
                select_inds = list(range(0, num_frames, step))
                select_inds = select_inds[:self.max_frame]
            else:
                # randomly sample when test mode is False
                select_inds = list(range(num_frames))
                random.shuffle(select_inds)
                select_inds = select_inds[:self.max_frame]
                select_inds = sorted(select_inds)
            for i, j in enumerate(select_inds):
                feat[i] = raw_feats[j]
        feat = torch.FloatTensor(feat)
        mask = torch.LongTensor(mask)
        return feat, mask

    def tokenize_text(self, text: str, max_len=512) -> tuple:

        encoded_inputs = self.tokenizer(text, max_length=max_len, padding='max_length', truncation=True)
        input_ids = torch.LongTensor(encoded_inputs['input_ids'])
        mask = torch.LongTensor(encoded_inputs['attention_mask'])
        return input_ids, mask

    def __getitem__(self, idx: int) -> dict:
        # Step 1, load visual features from zipfile.
        frame_input, frame_mask = self.get_visual_feats(idx)


        # Step 2, load title tokens
        # title_input, title_mask = self.tokenize_text(self.anns[idx]['title'], max_len=64)
        # # asr
        # asr_input, asr_mask = self.tokenize_text(self.anns[idx]['asr'], max_len=64)
        # asr_input = asr_input[1:]
        # asr_mask = asr_mask[1:]
        # # ocr
        # ocr_input, ocr_mask = self.tokenize_text(self.anns[idx]['ocr'], max_len=106)
        # ocr_input = ocr_input[1:]
        # ocr_mask = ocr_mask[1:]
        # title_input = torch.cat([title_input, asr_input, ocr_input], dim=-1)
        # title_mask = torch.cat([title_mask, asr_mask, ocr_mask], dim=-1)


        title_maxlen = 156
        asr_maxlen = 156
        ocr_maxlen = 156
        title_text = self.anns[idx]['title']
        asr_text = self.anns[idx]['asr']
        ocr_text = self.anns[idx]['ocr']

        if len(title_text)>title_maxlen:
            title_text = title_text[:title_maxlen//2]+title_text[-(title_maxlen//2):]

        if len(asr_text)>asr_maxlen:
            asr_text = asr_text[:asr_maxlen//2]+asr_text[-(asr_maxlen//2):]

        if len(ocr_text)>ocr_maxlen:
            ocr_text = ocr_text[:ocr_maxlen//2]+ocr_text[-(ocr_maxlen//2):]

        # frame_input, frame_mask = self.get_visual_feats(idx)
        #
        # # Step 2, load title tokens
        # title_input, title_mask = self.tokenize_text(self.anns[idx]['title'], max_len=64)
        #
        #
        #
        #
        # # asr
        # asr_input, asr_mask = self.tokenize_text(self.anns[idx]['asr'], max_len=64)
        # asr_input = asr_input[1:]
        # asr_mask = asr_mask[1:]
        # # for info in tqdm(anns):
        # #     ocr_list = info['ocr']
        # #     ocr_str = '。'.join([i['text'] for i in ocr_list])
        #
        # # ocr
        # ocr_input, ocr_mask = self.tokenize_text(self.anns[idx]['ocr'], max_len=100)
        # ocr_input = ocr_input[1:]
        # ocr_mask = ocr_mask[1:]

        # cat_input = torch.cat([title_input, asr_input, ocr_input], dim=-1)
        # cat_mask = torch.cat([title_mask, asr_mask, ocr_mask], dim=-1)






        cat_text = title_text+asr_text+ocr_text
        cat_input, cat_mask = self.tokenize_text(cat_text, max_len=466)
        cat_input = torch.cat((torch.tensor([101]), torch.tensor([102]), cat_input[1:]))
        cat_mask = torch.cat((torch.tensor([1]), cat_mask))


        # print(cat_input)
        # print(cat_mask)
        data = dict(
            frame_input=frame_input,
            frame_mask=frame_mask,
            title_input=cat_input,
            title_mask=cat_mask
        )

        # Step 4, load label if not test mode
        if not self.test_mode:
            label = category_id_to_lv2id(self.anns[idx]['category_id'])
            data['label'] = torch.LongTensor([label])

        return data
