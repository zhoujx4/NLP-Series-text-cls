"""
@Time : 2020/12/1110:44
@Auth : 周俊贤
@File ：run.py
@DESCRIPTION:

"""
import copy
import json
import time

import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from dataset.dataset import CLSDataset
from utils.adversarial import FGM
from utils.collate_fn import sup_collate_fn
from utils.logger import init_logger, logger
from utils.finetuning_argparse import get_argparse
from utils.progressbar import ProgressBar
from utils.seed_everything import seed_everything

from transformers import BertTokenizer, BertModel, BertConfig, AdamW
from models.model import CLS_model
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

def train(args, dataset, model, device):
    train_iter = DataLoader(
        dataset,
        batch_size=args.per_gpu_train_batch_size,
        shuffle=True,
        collate_fn=sup_collate_fn)

    # 优化器
    no_decay = ["bias", "LayerNorm.weight"]
    bert_param_optimizer = list(model.bert.named_parameters())
    linear_param_optimizer = list(model.fc1.named_parameters())
    linear_param_optimizer.extend(list(model.fc2.named_parameters()))
    optimizer_grouped_parameters = [
        {'params': [p for n, p in bert_param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay,
         'lr': args.learning_rate},
        {'params': [p for n, p in bert_param_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0,
         'lr': args.learning_rate},
        {'params': [p for n, p in linear_param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay,
         'lr': args.linear_learning_rate},
        {'params': [p for n, p in linear_param_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0,
         'lr': args.linear_learning_rate},
    ]
    optimizer = AdamW(optimizer_grouped_parameters,
                      lr=args.learning_rate,
                      eps=args.adam_epsilon)
    # 损失函数
    criterion = nn.CrossEntropyLoss().to(device)
    # 迭代训练
    model.train()
    #
    best_f1 = 0
    early_stop = 0
    # Train!
    logger.info("***** Running train %s *****")
    fgm = FGM(model,epsilon=1,emb_name='word_embeddings.weight')
    for epoch, _ in enumerate(range(int(args.num_train_epochs))):
        batch_loss = 0
        pbar = ProgressBar(n_total=len(train_iter), desc='Training')
        print("****" * 20)
        logger.info(" Begin {}-epoch! ".format(epoch))
        for step, batch in enumerate(train_iter):

            for key in batch.keys():
                batch[key] = batch[key].to(device)

            predictions = model(
                input_ids=batch['all_input_ids'],
                attention_mask=batch['all_attention_mask'],
                token_type_ids=batch['all_token_type_ids']
            )
            # 正常训练
            loss = criterion(predictions, batch['all_labels'])
            # print(nn.functional.softmax(predictions, dim=-1)[range(8), batch['all_labels']])
            loss.backward()
            # 对抗训练
            fgm.attack()  # 在embedding上添加对抗扰动
            predictions_adv = model(
                input_ids=batch['all_input_ids'],
                attention_mask=batch['all_attention_mask'],
                token_type_ids=batch['all_token_type_ids']
            )
            loss_adv = criterion(predictions_adv, batch['all_labels'])
            loss_adv.backward()  # 反向传播，并在正常的grad基础上，累加对抗训练的梯度
            fgm.restore()  # 恢复embedding参数
            #
            batch_loss += loss.item()
            pbar(step, {'batch_loss': batch_loss / (step + 1)})
            # 梯度下降，更新参数
            optimizer.step()
            model.zero_grad()

        # 每轮epoch在验证集上计算分数
        eval_f1 = evaluate(args, model)
        if eval_f1 > best_f1:
            early_stop = 0
            best_f1 = eval_f1
            logger.info(
                "the best f1 is {:.4f}, saving model !!".format(best_f1))
            best_model = copy.deepcopy(
                model.module if hasattr(
                    model, "module") else model)
            torch.save(
                best_model.state_dict(),
                os.path.join(
                    args.output_dir,
                    "best_model.pkl"))
        else:
            early_stop += 1
            if early_stop == args.early_stop:
                logger.info("Early stop in {} epoch!".format(epoch))
                break
        # 每 save_epochs 步保持模型
        if args.save_epochs > 0 and epoch % args.save_epochs == 0:
            if not os.path.exists(args.output_dir):
                os.makedirs(args.output_dir)
            model_to_save = (
                model.module if hasattr(
                    model, "module") else model)
            torch.save(
                model_to_save.state_dict(),
                os.path.join(
                    args.output_dir,
                    "epoch-{}.pkl".format(epoch)))
    test_f1 = evaluate(args, best_model, type="test")
    logger.info("Test F1 is {}!".format(test_f1))

def evaluate(args, model, prefix="", type="dev"):
    if type == "dev":
        eval_dataset = CLSDataset(
            args,
            csv_path="./data/{}/dev.csv".format(args.task),
            tokenizer=args.tokenizer,
            type="dev")
    elif type == "test":
        eval_dataset = CLSDataset(
            args,
            csv_path="./data/{}/test.csv".format(args.task),
            tokenizer=args.tokenizer,
            type="test")

    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=args.per_gpu_eval_batch_size,
        shuffle=False,
        collate_fn=sup_collate_fn,
        )
    # Eval!
    logger.info("***** Running {} {} *****".format(type, prefix))
    eval_loss = 0.0
    eval_steps = 0
    criterion = nn.CrossEntropyLoss().to(args.device)
    pbar = ProgressBar(n_total=len(eval_dataloader), desc="Evaluating")

    pres, trues = [], []
    model.eval()
    for step, batch in enumerate(eval_dataloader):
        for key in batch.keys():
            batch[key] = batch[key].to(args.device)
        with torch.no_grad():
            predictions = model(
                input_ids=batch['all_input_ids'],
                attention_mask=batch['all_attention_mask'],
                token_type_ids=batch['all_token_type_ids']
            )
        loss = criterion(predictions, batch['all_labels'])
        eval_loss += loss.item()
        eval_steps += 1

        pbar(step)
        _, pre = torch.max(predictions, axis=1)
        pre = pre.cpu().numpy().tolist()
        true = batch['all_labels'].cpu().numpy().tolist()
        pres.extend(pre)
        trues.extend(true)

    score_f1 = f1_score(trues, pres, average="micro")
    eval_loss = eval_loss / eval_steps
    logger.info("Micro F1:{:.4f}, Loss:{:.4f}".format(score_f1, eval_loss))
    return score_f1

def main():
    args = get_argparse().parse_args()
    args.datapath = os.path.join("./data", args.task)
    print(json.dumps(vars(args), sort_keys=True, indent=4, separators=(', ',': '), ensure_ascii=False))
    init_logger(log_file="./log/{}.log".format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
    seed_everything(args.seed)
    #
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # model
    model = CLS_model(args.model_name_or_path, 200, 15)
    model.to(args.device)
    #
    tokenizer = BertTokenizer.from_pretrained('/data/zhoujx/prev_trained_model/chinese_roberta_wwm_ext_pytorch')
    args.tokenizer = tokenizer
    #
    dataset = CLSDataset(
        args,
        csv_path="./data/{}/train.csv".format(args.task),
        tokenizer=args.tokenizer,
        type="train",
        is_augement=False)
    # 训练
    train(args, dataset, model, args.device)

if __name__ == "__main__":
    main()