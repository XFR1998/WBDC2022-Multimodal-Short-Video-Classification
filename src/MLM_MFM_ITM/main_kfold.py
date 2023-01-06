#!/usr/bin/env python
# @Project ：challenge-pretrain 
# @File    ：main_kfold.py
# @Author  ：
# @Date    ：2022/6/20 17:52 
# 
# --------------------------------------------------------
import os

#os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
import logging

import time
import torch

from myconfig.config import parse_args
from data.pre_data_kfold import create_kfold_dataloaders
from pretrain_model.my_nui_model import MultiModal
from myconfig.util import setup_device, setup_seed, setup_logging, build_optimizer_fineturn, evaluate
from tqdm import tqdm


def validate(model, val_dataloader, args):
    model.eval()
    if args.ema != False:
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
    print('-' * 20)
    print('！！！！！！！！！！！kfold开始冲啊！！！！！！！！！！！')
    print('-' * 20)
    if not os.path.exists(args.saved_kfold_model_path):
        os.mkdir(args.saved_kfold_model_path)
    for fold_idx in range(1, 6):
        args.ema = True
        print('**' * 10, '第', fold_idx, '折', 'ing....', '**' * 10)
        if not os.path.exists(args.saved_kfold_model_path):
            os.mkdir(args.saved_kfold_model_path)
        save_kfold_model_path = args.saved_kfold_model_path + '/{}fold_model'.format(fold_idx)
        if not os.path.exists(save_kfold_model_path):
            os.mkdir(save_kfold_model_path)

        os.makedirs(args.saved_kfold_model_path + '/{}fold_model'.format(fold_idx), exist_ok=True)
        logging.info("Training/evaluation parameters: %s", args)

        # 1. load data
        train_dataloader, val_dataloader = create_kfold_dataloaders(args=args, fold_idx=fold_idx)
        # continue

        # 2. build model and optimizers
        # model = MultiModal111(args)

        # ---------------------------------------------------------------
        model = MultiModal(args, task=['tag'], init_from_pretrain=True)
        model_new_dict = model.state_dict()
        best_model_path = './save/best_pretrain_model.pth'
        print('加载最优预训练模型：', best_model_path)
        checkpoint = torch.load(best_model_path, map_location='cpu')
        import copy
        checkpointcopy = copy.deepcopy(checkpoint)
        for k in checkpoint:
            checkpointcopy[k.replace('module.', '')] = checkpoint[k]
            del checkpointcopy[k]

        print('-' * 20)
        print('*' * 5 + '输出预训练和微调不一样的层' + '*' * 5)
        for k, v in checkpointcopy.items():
            if k not in model_new_dict.keys():
                print(k)
        print('-' * 20)

        model.load_state_dict(checkpointcopy, strict=False)
        # ---------------------------------------------------------------

        num_total_steps = len(train_dataloader) * args.max_epochs
        optimizer, scheduler = build_optimizer_fineturn(args, model, num_total_steps=num_total_steps)
        if args.device == 'cuda':
            model = torch.nn.parallel.DataParallel(model.to(args.device))

        if args.ema == True:
            print('-' * 10, '采用EMA机制训练', '-' * 10)
            from tricks import EMA
            args.ema = EMA(model, 0.999)
            args.ema.register()

        if args.use_fgm == True:
            print('-' * 10, '采用FGM对抗训练', '-' * 10)
            from tricks import FGM
            # 初始化
            fgm = FGM(model)

        if args.use_pgd == True:
            print('-' * 10, '采用PGD对抗训练', '-' * 10)
            from tricks import PGD
            # 初始化
            pgd = PGD(model=model)
            K = 3

        # 3. training
        step = 0
        best_score = args.best_score
        start_time = time.time()

        # -------------------控制早停--------------
        early_stop_epochs = 2
        no_improve_epochs = 0
        # ---------------------------------------
        print('radom seed: ', args.seed)
        print('batch_size: ', args.batch_size)
        print('Epochs nums: ', args.max_epochs)
        print('Max len: ', args.maxlen)
        for epoch in range(args.max_epochs):
            print('-' * 10, 'epoch: ', epoch, '-' * 10)
            for batch in tqdm(train_dataloader):
                model.train()
                loss, accuracy, _, _ = model(batch)
                loss = loss.mean()
                accuracy = accuracy.mean()
                loss.backward()

                # -----------------------------------对抗攻击------------------------------------------------
                if args.use_fgm:
                    # 对抗训练
                    fgm.attack()  # 在embedding上添加对抗扰动
                    # loss_adv = model(batch_input, batch_label)
                    loss_adv, accuracy, _, _ = model(batch)
                    loss_adv = loss_adv.mean()
                    loss_adv.backward()  # 反向传播，并在正常的grad基础上，累加对抗训练的梯度
                    fgm.restore()  # 恢复embedding参数

                if args.use_pgd:
                    pgd.backup_grad()
                    for t in range(K):
                        pgd.attack(is_first_attack=(t == 0))
                        if t != K - 1:
                            model.zero_grad()
                        else:
                            pgd.restore_grad()

                        loss_adv, accuracy, _, _ = model(batch)
                        loss_adv = loss_adv.mean()
                        loss_adv.backward()  # 反向传播，并在正常的grad基础上，累加对抗训练的梯度

                    pgd.restore()

                # ----------------------------------------------------------------------------------------

                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

                if args.ema != False:
                    args.ema.update()

                # step += 1
                # if step % args.print_steps == 0:
                #     time_per_step = (time.time() - start_time) / max(1, step)
                #     remaining_time = time_per_step * (num_total_steps - step)
                #     remaining_time = time.strftime('%H:%M:%S', time.gmtime(remaining_time))
                #     logging.info(f"Epoch {epoch} step {step} eta {remaining_time}: loss {loss:.3f}, accuracy {accuracy:.3f}")

            # 4. validation
            loss, results = validate(model, val_dataloader, args)
            results = {k: round(v, 4) for k, v in results.items()}
            # logging.info(f"Epoch {epoch} step {step}: loss {loss:.3f}, {results}")
            logging.info(f"Epoch {epoch}: loss {loss:.3f}, {results}")
            # 5. save checkpoint
            mean_f1 = results['mean_f1']
            if mean_f1 > best_score:
                print('')
                print('best model saved!!!!!!!!!!!!!')
                print('mean_f1: ', mean_f1)
                best_score = mean_f1
                state_dict = model.module.state_dict() if args.device == 'cuda' else model.state_dict()
                torch.save({'epoch': epoch, 'model_state_dict': state_dict, 'mean_f1': mean_f1},
                           f'{save_kfold_model_path}/model_epoch_{epoch}_mean_f1_{mean_f1}.bin')

                no_improve_epochs = 0
            else:
                no_improve_epochs += 1

            if no_improve_epochs == early_stop_epochs:
                print('no improve score !!! stop train !!!')
                break

            if args.ema != False:
                args.ema.restore()


def main():
    args = parse_args()
    setup_logging()
    setup_device(args)
    setup_seed(args)

    train_and_validate(args)


if __name__ == '__main__':
    main()
