#!/usr/bin/env python
# @Project ：challenge-pretrain 
# @File    ：fineturn.py
# @Author  ：
# @Date    ：2022/6/8 0:48 
# 
# --------------------------------------------------------
import logging
import os
import time
import torch

from myconfig.config import parse_args
from data.pre_data_helper import create_dataloaders
from pretrain_model.my_nui_model import MultiModal, MultiModal_fineturn
from myconfig.util import setup_device, setup_seed, setup_logging, build_optimizer, evaluate, build_optimizer_fineturn
from tqdm import tqdm
from copy import deepcopy
# from torchcontrib.optim import SWA
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"


def validate(model, val_dataloader, args):
    model.eval()
    if args.ema!=False:
        args.ema.apply_shadow()
    predictions = []
    labels = []
    losses = []
    with torch.no_grad():
        for batch in tqdm(val_dataloader):
            loss, _, pred_label_id, label = model(batch)
            loss = loss.mean()
            predictions.extend(pred_label_id.cpu().numpy())
            labels.extend(label.cpu().numpy())
            losses.append(loss.cpu().numpy())
    loss = sum(losses) / len(losses)
    results = evaluate(predictions, labels)

    model.train()
    return loss, results


def train_and_validate(args):
    # 1. load data
    train_dataloader, val_dataloader = create_dataloaders(args)

    # 2. build model and optimizers
    model = MultiModal(args, task=['tag'])
    # 加载预训练权重
    model_new_dict = model.state_dict()
    # for k in model_new_dict:
    #     print(k)

    # 参数对齐
    checkpoint = torch.load(args.fineturn_model, map_location='cpu')
    checkpointcopy = deepcopy(checkpoint)
    for k in checkpoint:
        checkpointcopy[k.replace("module.", '')] = checkpoint[k]
        del checkpointcopy[k]

    for k in checkpointcopy:
        if k not in model_new_dict:
            print(k)
    model.load_state_dict(checkpointcopy, strict=False)

    # =============打印模型学习参数==============================================
    print("=" * 40)
    print("epoch: ", args.max_epochs)
    print("batch size: ", args.batch_size)
    print("max len: ", args.maxlen)
    print("ckpt file: ", args.ckpt_file)
    print("fineturn model: ", args.fineturn_model)
    print("pretrain model path: ", args.pretrain_model_path)
    print("use all data train: ", args.use_all_data)
    num_total_steps = len(train_dataloader) * args.max_epochs
    print("num_total_steps: ", num_total_steps)

    optimizer, scheduler = build_optimizer_fineturn(args, model, num_total_steps)

    if args.device == 'cuda':
        model = torch.nn.parallel.DataParallel(model.to(args.device))

    # ===================== my tricks ========================================
    if args.use_swa:
        print('-' * 10, '采用SWA机制训练', '-' * 10)
        # 采用自动模式
        # swa_start：是一个整数，表示经过swa_start个优化steps之后，将学习率切换到固定值swa_lr；
        # swa_freq：在每swa_freq个steps优化之后，会将对应的权重加到SWA滑动平均的结果上。
        # 最后，使用opt.swap_swa_sgd()之后，模型的权重将会替换为SWA滑动平均权重。
        print("now step:", num_total_steps * 0.75 - len(train_dataloader) * 4)
        optimizer = SWA(optimizer, swa_start=(num_total_steps * 0.75 - len(train_dataloader) * 4), swa_freq=100,
                        swa_lr=5e-5)

    if args.ema==True:
        print('-'*10, '采用EMA机制训练', '-'*10)
        from tricks import EMA
        args.ema = EMA(model, 0.999)
        args.ema.register()

    if args.use_fgm==True:
        print('-' * 10, '采用FGM对抗训练', '-' * 10)
        from tricks import FGM
        # 初始化
        fgm = FGM(model)

    # 3. training
    step = 0
    best_score = args.best_score
    start_time = time.time()
    for epoch in range(args.max_epochs):
        for batch in tqdm(train_dataloader):
            model.train()
            loss, accuracy, _, _ = model(batch)
            loss = loss.mean()
            accuracy = accuracy.mean()
            loss.backward()

            # =========================== 对抗攻击 ======================================
            if args.use_fgm:
                # 对抗训练
                fgm.attack()  # 在embedding上添加对抗扰动
                loss_adv, accuracy, _, _ = model(batch)
                loss_adv = loss_adv.mean()
                loss_adv.backward()  # 反向传播，并在正常的grad基础上，累加对抗训练的梯度
                fgm.restore()  # 恢复embedding参数

            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

            if args.ema!= False:
                args.ema.update()

        # 4. validation
        loss, results = validate(model, val_dataloader, args)
        results = {k: round(v, 4) for k, v in results.items()}
        logging.info(f"Epoch {epoch} step {step}: loss {loss:.3f}, {results}")

        if args.use_all_data:
            if epoch == 3:
                state_dict = model.module.state_dict() if args.device == 'cuda' else model.state_dict()
                torch.save({'epoch': epoch, 'model_state_dict': state_dict},
                           f'{args.savedmodel_path}/model_epoch_{epoch}_mean_f1.bin')
                break
        else:
            # 5. save checkpoint
            mean_f1 = results['mean_f1']
            if mean_f1 > best_score:
                best_score = mean_f1
                state_dict = model.module.state_dict() if args.device == 'cuda' else model.state_dict()
                torch.save({'epoch': epoch, 'model_state_dict': state_dict, 'mean_f1': mean_f1},
                           f'{args.savedmodel_path}/model_epoch_{epoch}_mean_f1_{mean_f1}.bin')

        if args.use_swa:
            optimizer.swap_swa_sgd()
            optimizer.bn_update(train_dataloader, model)

        if args.ema != False:
            args.ema.restore()


def main():
    args = parse_args()
    setup_logging()
    setup_device(args)
    setup_seed(args)

    os.makedirs(args.savedmodel_path, exist_ok=True)
    logging.info("Training/evaluation parameters: %s", args)

    train_and_validate(args)


if __name__ == '__main__':
    main()