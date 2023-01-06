import logging
import os
import time
import torch

from myconfig.config import parse_args
from data.pre_data_helper import create_dataloaders
from pretrain_model.my_nui_model import MultiModal
from myconfig.util import setup_device, setup_seed, setup_logging, build_optimizer, evaluate
from tqdm import tqdm
import numpy as np


def eval(model, data_loader, compute_loss=True, eval_max_num=99999):
    """Evaluates the |model| on |data_loader|"""
    model.eval()
    loss_l, emb_l, vid_l = [], [], []

    with torch.no_grad():
        for batch_num, item in enumerate(data_loader):
            loss, mlmloss, mfmloss, itmloss= model(item)

            if loss is not None:
                # print('loss2:',loss.shape)
                loss_l.append(loss.to("cpu"))

    loss_l = torch.mean(torch.stack(loss_l)).item()
    return loss_l #, np.concatenate(vid_l)


def train_and_validate(args):
    # 1. load data
    logging.info("loading data...")
    train_dataloader, val_dataloader = create_dataloaders(args)
    logging.info("load data successfully")

    # 2. build model and optimizers
    logging.info("building model and optimizers...")
    model = MultiModal(args, task=['mlm', 'mfm', 'itm'])


    # for n,p in model.named_parameters():
    #     print(n)
    print("="*40)
    print("epoch: ", args.max_epochs)
    print("batch size: ", args.batch_size)
    print("max len: ", args.maxlen)
    print("ckpt file: ", args.ckpt_file)
    print("pretrain model path: ", args.pretrain_model_path)

    num_total_steps = len(train_dataloader)*args.max_epochs
    print("num_total_steps: ", num_total_steps)
    optimizer, scheduler = build_optimizer(args, model, num_total_steps)
    if args.device == 'cuda':
        model = torch.nn.parallel.DataParallel(model.to(args.device))

    # 3. training
    best_val_loss, best_epoch, step = None, 0, 0
    start_time = time.time()
    for epoch in range(args.max_epochs):
        for batch in tqdm(train_dataloader):
            model.train()
            loss, mlmloss, mfmloss, itmloss = model(batch)
            loss = loss.mean()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

            step += 1
            if step % args.print_steps == 0:
                time_per_step = (time.time() - start_time) / max(1, step)
                remaining_time = time_per_step * (num_total_steps - step)
                remaining_time = time.strftime('%H:%M:%S', time.gmtime(remaining_time))
                logging.info(f"Epoch {epoch} step {step} eta {remaining_time}: loss {loss:.3f}, mlm loss{mlmloss.mean():.5f}, mfm loss{mfmloss.mean():.5f}, itm loss{itmloss.mean():.5f}" )

        # 4. validation
        val_loss = eval(model, val_dataloader, eval_max_num=10000)
        logging.info(f"Epoch {epoch} step {step}: loss {val_loss:.3f}")

        improve_str = ''
        if not best_val_loss or val_loss < best_val_loss:
            best_val_loss = round(val_loss, 5)
            torch.save(model.state_dict(), f'{args.savedmodel_path}/best_pretrain_model.pth')
            improve_str = f"|New best_val_loss={best_val_loss:6.4}"

            logging.info(f"Epoch={epoch + 1}/{epoch}|step={step:3}|val_loss={val_loss:6.4}" + improve_str)


def main():
    args = parse_args()
    print(args)
    setup_logging()
    setup_device(args)
    setup_seed(args)

    os.makedirs(args.savedmodel_path, exist_ok=True)
    logging.info("Training/evaluation parameters: %s", args)

    train_and_validate(args)


if __name__ == '__main__':
    main()
