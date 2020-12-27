"""
@Time : 2020/12/1110:44
@Auth : 周俊贤
@File ：run.py
@DESCRIPTION:

"""
import os
import copy
import time
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from dataset.dataset import CLSDataset, UDADataset
from utils.adversarial import FGM
from utils.collate_fn import sup_collate_fn, unsup_collate_fn
from utils.get_tsa_thresh import get_tsa_thresh
from utils.logger import init_logger, logger
from utils.finetuning_argparse import get_argparse
from utils.progressbar import ProgressBar
from utils.seed_everything import seed_everything

from transformers import BertTokenizer, BertModel, BertConfig, AdamW
from models.model import CLS_model
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

def train_UDA(args, label_dataset, unlabel_dataset, model, device):
    # create iteration
    sup_iter = DataLoader(
        label_dataset,
        batch_size=args.per_gpu_train_batch_size,
        shuffle=True,
        collate_fn=sup_collate_fn)
    sup_iter2 = iter(sup_iter)
    unsup_iter = DataLoader(
        unlabel_dataset,
        batch_size=args.per_gpu_train_batch_size * args.unsup_ratio,
        shuffle=True,
        collate_fn=unsup_collate_fn)

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
    sup_criterion = nn.CrossEntropyLoss(reduction="none").to(device)
    unsup_criterion = nn.KLDivLoss(reduction='none').to(device)

    # 迭代训练
    model.train()
    #
    global_step = 0
    best_f1 = 0
    early_stop = 0
    # Train!
    logger.info("***** Running train %s *****")
    batch_final_loss, batch_sup_loss, batch_unsup_loss = 0, 0, 0
    for epoch, _ in enumerate(range(int(args.num_train_epochs))):
        pbar = ProgressBar(n_total=len(unsup_iter), desc='Training')
        print("****" * 20)
        logger.info(" Begin {}-epoch! ".format(epoch))
        for step, unsup_batch in enumerate(unsup_iter):
            for key in unsup_batch.keys():
                unsup_batch[key] = unsup_batch[key].to(device)
            try:
                sup_batch = next(sup_iter2)
            except:
                sup_iter2 = iter(sup_iter)
                sup_batch = next(sup_iter2)
            #### Supervised Loss
            for key in sup_batch.keys():
                sup_batch[key] = sup_batch[key].to(device)
            predictions = model(
                input_ids = sup_batch['all_input_ids'],
                attention_mask = sup_batch['all_attention_mask'],
                token_type_ids = sup_batch['all_token_type_ids']
            )
            #
            sup_loss = sup_criterion(predictions, sup_batch['all_labels'])
            # print(torch.exp(-sup_loss))
            if args.tsa:
                tsa_thresh = get_tsa_thresh(args.tsa,
                                            global_step,
                                            args.total_steps,
                                            start=1. / predictions.shape[-1],
                                            end=1)
                larger_than_threshold = torch.exp(-sup_loss) > tsa_thresh
                loss_mask = torch.ones_like(sup_batch['all_labels'], dtype=torch.float32) * (1 - larger_than_threshold.type(torch.float32))
                sup_loss = torch.sum(sup_loss * loss_mask, dim=-1) / \
                           torch.max(torch.sum(loss_mask, dim=-1), torch.tensor(1.).to(device))
            else:
                sup_loss = torch.mean(sup_loss)

            #### UnSupervised Loss
            uda_softmax_temp = args.uda_softmax_temp if args.uda_softmax_temp > 0 else 1.
            # ori
            len_ori = int(unsup_batch['all_input_ids'].shape[0]/2)
            with torch.no_grad():
                ori_logits = model(
                                    input_ids = unsup_batch['all_input_ids'][:len_ori],
                                    attention_mask = unsup_batch['all_attention_mask'][:len_ori],
                                    token_type_ids = unsup_batch['all_token_type_ids'][:len_ori]
                                    )
                # confidence-based masking
                ori_prob = F.softmax(ori_logits, dim=-1)
                if args.uda_confidence_thresh != -1:
                    unsup_loss_mask = torch.max(ori_prob, dim=-1)[0] > args.uda_confidence_thresh
                    unsup_loss_mask = unsup_loss_mask.type(torch.float32)
                else:
                    unsup_loss_mask = torch.ones(len_ori, dtype=torch.float32)
                unsup_loss_mask = unsup_loss_mask.to(device)
            # Sharpening Predictions
            ori_prob = F.softmax(ori_logits / uda_softmax_temp, dim=-1)
            # aug
            # softmax temperature controlling
            logits = model(
                        input_ids = unsup_batch['all_input_ids'][len_ori:],
                        attention_mask = unsup_batch['all_attention_mask'][len_ori:],
                        token_type_ids = unsup_batch['all_token_type_ids'][len_ori:]
                        )
            aug_log_prob = F.log_softmax(logits, dim=-1)
            # KLdiv loss
            unsup_loss = torch.sum(unsup_criterion(aug_log_prob, ori_prob), dim=-1)
            unsup_loss = torch.sum(unsup_loss * unsup_loss_mask, dim=-1) / torch.max(torch.sum(unsup_loss_mask, dim=-1), torch.tensor(1.).to(device))
            final_loss = sup_loss + args.uda_coeff*unsup_loss
            #
            batch_final_loss += final_loss.item()
            batch_sup_loss += sup_loss.item()
            batch_unsup_loss += unsup_loss.item()
            pbar(step, {'global_final_loss': batch_final_loss / (global_step + 1),
                        'global_sup_loss': batch_sup_loss / (global_step + 1),
                        'global_unsup_loss': batch_unsup_loss / (global_step + 1),
                        'final_loss': final_loss,
                        'sup_loss': sup_loss,
                        'unsup_loss': unsup_loss,
                        'tsa_thresh': tsa_thresh
                        })
            ###
            if global_step % 500 == 0:
                eval_f1 = evaluate(args, model)
            #### 梯度下降，更新参数
            final_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1

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
            csv_path=os.path.join(args.datapath, "dev.csv"),
            tokenizer=args.tokenizer,
            type="dev")
    elif type == "test":
        eval_dataset = CLSDataset(
            args,
            csv_path=os.path.join(args.datapath, "test.csv"),
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
    log_path = "./log/{}_{}.log".format(args.task, time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    init_logger(log_file=log_path)
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
    label_dataset = CLSDataset(
        args,
        csv_path=os.path.join(args.datapath, "train.csv"),
        tokenizer=args.tokenizer,
        type="train",
        is_augement=False)
    unlabel_dataset = UDADataset(
        args,
        csv_path=os.path.join(args.datapath, "unsup.csv"),
        tokenizer=args.tokenizer,
    )
    # 训练
    train_UDA(args, label_dataset, unlabel_dataset, model, args.device)

if __name__ == "__main__":
    main()
