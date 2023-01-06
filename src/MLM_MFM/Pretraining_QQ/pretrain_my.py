import os
#os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"
import os, math, random, time, sys, gc, sys, json, psutil

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import logging
from imp import reload

reload(logging)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt='%H:%M:%S',
    handlers=[
        logging.FileHandler(f"train.log"),
        logging.StreamHandler()
    ]
)

import numpy as np
import pandas as pd
from my_model import QQUniModel
from config.data_cfg import *
from config.model_cfg import *
from config.pretrain_cfg import *





import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, ChainDataset
from transformers import AutoConfig



from data_helper_pretrain import create_dataloaders
from pretrain_config import parse_args
from pretrain_util import setup_device, setup_seed, setup_logging, build_optimizer, evaluate
from tqdm import tqdm







def get_pred_and_loss(model, inputs):
    """Get pred and loss for specific task"""
    pred, loss = model(inputs)
    return pred, loss


def eval(model, data_loader, get_pred_and_loss):
    """Evaluates the |model| on |data_loader|"""
    model.eval()
    loss_l = []

    with torch.no_grad():
        for inputs in tqdm(data_loader):
            pred, loss = get_pred_and_loss(model, inputs)

            if loss is not None:
                loss_l.append(loss.to("cpu"))
    
    loss_l = torch.mean(torch.stack(loss_l)).item()
    return loss_l


def train(args,
          model,
          model_path,
          train_loader,
          val_loader,
          optimizer,
          get_pred_and_loss,
          scheduler=None,
          num_epochs=5):
    best_val_loss, best_epoch, step = None, 0, 0

    for epoch in range(num_epochs):
        for batch in tqdm(train_loader):
            model.train()
            optimizer.zero_grad()
            pred, loss = get_pred_and_loss(model, batch)
            loss = loss.mean()
            loss.backward()

            optimizer.step()
            if scheduler:
                scheduler.step()
            if step == 20:
                val_loss = eval(model, val_loader, get_pred_and_loss=get_pred_and_loss)
                improve_str = ''
                if not best_val_loss or val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(model.state_dict(), f'save_model/best_model_{best_val_loss}.pth')
                    # state_dict = model.module.state_dict() if args.device == 'cuda' else model.state_dict()
                    # torch.save({'epoch': epoch, 'model_state_dict': state_dict, 'best_val_loss': best_val_loss},
                    #            f'save_model/best_model_{best_val_loss}.bin')

                    improve_str = f"|New best_val_loss={best_val_loss:6.4}"

                logging.info(
                    f"Epoch={epoch + 1}/{num_epochs}|step={step:3}|val_loss={val_loss:6.4}" + improve_str)

            step += 1
            # if step == 20 or (step % 3000 == 0 and step > 0):
            #     val_loss = eval(model, val_loader, get_pred_and_loss=get_pred_and_loss)
            #     improve_str = ''
            #     if not best_val_loss or val_loss < best_val_loss:
            #         best_val_loss = val_loss
            #         torch.save(model.state_dict(), f'save_model/best_model_{best_val_loss}.pth')
            #         # state_dict = model.module.state_dict() if args.device == 'cuda' else model.state_dict()
            #         # torch.save({'epoch': epoch, 'model_state_dict': state_dict, 'best_val_loss': best_val_loss},
            #         #            f'save_model/best_model_{best_val_loss}.bin')
            #
            #         improve_str = f"|New best_val_loss={best_val_loss:6.4}"
            #
            #     logging.info(
            #         f"Epoch={epoch + 1}/{num_epochs}|step={step:3}|val_loss={val_loss:6.4}" + improve_str)
            # step += 1

        val_loss = eval(model, val_loader, get_pred_and_loss=get_pred_and_loss)
        improve_str = ''
        if not best_val_loss or val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f'save_model/best_model.pth')
            # state_dict = model.module.state_dict() if args.device == 'cuda' else model.state_dict()
            # torch.save({'epoch': epoch, 'model_state_dict': state_dict, 'best_val_loss': best_val_loss},
            #            f'save_model/best_model_{best_val_loss}.bin')

            improve_str = f"|New best_val_loss={best_val_loss:6.4}"

        logging.info(
            f"Epoch={epoch + 1}/{num_epochs}|step={step:3}|val_loss={val_loss:6.4}" + improve_str)




    return best_val_loss



def pretain_model(args):

    # model
    model = QQUniModel(args, task=['mlm', 'mfm'], init_from_pretrain=True)



    # 1. load data
    train_dataloader, val_dataloader = create_dataloaders(args)
    num_total_steps = len(train_dataloader) * args.max_epochs



    # ---------------------------线性warmup scheduler--------------------------------------
    # optimizer, scheduler = build_optimizer(args, model, num_total_steps=num_total_steps)
    # -------------------------------------------------------------------------------------

    # ---------------------------余弦退火warmup scheduler--------------------------------------
    warmup_steps = int(WARMUP_RATIO * num_total_steps)
    logging.info(f'Total train steps={num_total_steps}, warmup steps={warmup_steps}')

    # model
    # optimizer
    from optim.create_optimizer import create_optimizer
    logging.info(f'LR setting: {LR}')
    optimizer = create_optimizer(model, model_lr=LR, layerwise_learning_rate_decay=LR_LAYER_DECAY)
    # schedueler
    from transformers import get_cosine_schedule_with_warmup
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_training_steps=num_total_steps,
                                                num_warmup_steps=warmup_steps)
    # -------------------------------------------------------------------------------------







    if args.device == 'cuda':
        model = torch.nn.parallel.DataParallel(model.to(args.device))


    print('radom seed: ', args.seed)
    print('batch_size: ', args.batch_size)
    print('Epochs nums: ', args.max_epochs)


    # train
    model_path = f"best_pretrain_model.pth"
    val_loss = train(args,model, model_path, train_dataloader, val_dataloader, optimizer,
                     get_pred_and_loss=get_pred_and_loss,
                     scheduler=scheduler, num_epochs=args.max_epochs)
    print('-'*20)
    logging.info(f"best_val_loss={val_loss}")
    print('-'*20)





def main():
    args = parse_args()
    setup_logging()
    setup_device(args)
    setup_seed(args)

    os.makedirs(args.savedmodel_path, exist_ok=True)
    logging.info("Training/evaluation parameters: %s", args)

    pretain_model(args)



if __name__ == '__main__':
    main()
    logging.info("Pretraining Finish!!!!")
