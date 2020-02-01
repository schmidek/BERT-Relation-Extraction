#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 09:37:26 2019

@author: weetee
"""
import os
import math
import torch
import torch.nn as nn
from ..misc import save_as_pickle, load_pickle
from seqeval.metrics import precision_score, recall_score, f1_score, classification_report
import logging
from tqdm import tqdm

logging.basicConfig(format='%(asctime)s [%(levelname)s]: %(message)s', \
                    datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
logger = logging.getLogger(__file__)

def load_state(net, optimizer, scheduler, args, load_best=False):
    """ Loads saved model and optimizer states if exists """
    base_path = "./data/"
    amp_checkpoint = None
    checkpoint_path = os.path.join(base_path,"task_test_checkpoint_%d.pth.tar" % args.model_no)
    best_path = os.path.join(base_path,"task_test_model_best_%d.pth.tar" % args.model_no)
    start_epoch, best_pred, checkpoint = 0, 10000000, None
    if (load_best == True) and os.path.isfile(best_path):
        checkpoint = torch.load(best_path)
        logger.info("Loaded best model.")
    elif os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        logger.info("Loaded checkpoint model.")
    if checkpoint != None:
        start_epoch = checkpoint['epoch']
        best_pred = checkpoint['best_acc']
        net.load_state_dict(checkpoint['state_dict'])
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer'])
        if scheduler is not None:
            scheduler.load_state_dict(checkpoint['scheduler'])
        amp_checkpoint = checkpoint['amp']
        logger.info("Loaded model and optimizer.")    
    return start_epoch, best_pred, amp_checkpoint

def load_results(model_no=0):
    """ Loads saved results if exists """
    losses_path = "./data/task_test_losses_per_epoch_%d.pkl" % model_no
    accuracy_path = "./data/task_train_accuracy_per_epoch_%d.pkl" % model_no
    f1_path = "./data/task_test_f1_per_epoch_%d.pkl" % model_no
    if os.path.isfile(losses_path) and os.path.isfile(accuracy_path) and os.path.isfile(f1_path):
        losses_per_epoch = load_pickle("task_test_losses_per_epoch_%d.pkl" % model_no)
        accuracy_per_epoch = load_pickle("task_train_accuracy_per_epoch_%d.pkl" % model_no)
        f1_per_epoch = load_pickle("task_test_f1_per_epoch_%d.pkl" % model_no)
        logger.info("Loaded results buffer")
    else:
        losses_per_epoch, accuracy_per_epoch, f1_per_epoch = [], [], []
    return losses_per_epoch, accuracy_per_epoch, f1_per_epoch


def evaluate_(output, labels):
    mask = ~torch.eq(labels, torch.tensor(-1.0))
    #print("mask", mask)
    masked_output = torch.masked_select(output, mask)
    l = torch.masked_select(labels, mask).cpu()
    o = torch.sigmoid(masked_output).cpu().round().int()

    if len(l) > 1:
        acc = (l == o).sum().item()/len(l)

    return acc

bce = nn.BCEWithLogitsLoss()
def criterion(logits, labels):
    #print("logits", logits)
    #print("labels", labels)
    mask = ~torch.eq(labels, torch.tensor(-1.0))
    #print("mask", mask)
    masked_logits = torch.masked_select(logits, mask)
    masked_labels = torch.masked_select(labels, mask).double()
    ret = bce(masked_logits, masked_labels)
    #print(ret)
    return ret

def evaluate_results(net, test_loader, pad_id, cuda, num_classes):
    logger.info("Evaluating test samples...")
    acc = 0; out_labels = []; true_labels = []
    loss = 0
    net.eval()
    with torch.no_grad():
        for i, data in tqdm(enumerate(test_loader), total=len(test_loader)):
            x, e1_e2_start, labels, _,_,_ = data
            attention_mask = (x != pad_id).float()
            token_type_ids = torch.zeros((x.shape[0], x.shape[1])).long()

            if cuda:
                x = x.cuda()
                labels = labels.cuda()
                attention_mask = attention_mask.cuda()
                token_type_ids = token_type_ids.cuda()
                
            classification_logits = net(x, token_type_ids=token_type_ids, attention_mask=attention_mask, Q=None,\
                          e1_e2_start=e1_e2_start)
            
            accuracy = evaluate_(classification_logits, labels.squeeze(1))
            out_labels.extend(torch.sigmoid(classification_logits).cpu().numpy())
            true_labels.extend(labels.squeeze(1).cpu().numpy())
            acc += accuracy
            loss += criterion(classification_logits, labels.squeeze(1))

    #print(out_labels)
    tp_total = 0
    fn_total = 0
    fp_total = 0
    for c in range(num_classes):
        prediction_scores = list(map(lambda x: x[c], out_labels))
        labels = list(map(lambda x: x[c], true_labels))
        #print(prediction_scores)
        #print(labels)
        sorted_results = sorted(zip(prediction_scores, labels), key=lambda tup: tup[0], reverse=True)

        best_fscore = 0.0
        best_threshold = 0.95
        tp = 0
        fp = 0
        fn = sum(list(map( lambda x: (1 if x[1] == 1 else 0), sorted_results)))
        tp_at_best = 0
        fn_at_best = 0
        fp_at_best = fn

        if fn == 0:
            print("Skipping", c, "because of no labels")
            continue

        for i in range(len(sorted_results)):
            r = sorted_results[i]
            threshold = r[0]
            label = r[1]

            if label == -1: #unknown
                continue
            if label == 1:
                tp += 1
                fn -= 1
            else:
                fp += 1

            if i+1 < len(sorted_results) and sorted_results[i+1][0] == threshold:
                continue

            precision = tp / (tp+fp)
            recall = tp / (tp+fn)
            if tp == 0:
                continue

            fscore = 2 * precision * recall / (precision + recall)
            if fscore > best_fscore:
                best_fscore = fscore
                best_threshold = (threshold + sorted_results[i+1][0]) / 2.0
                tp_at_best = tp
                fn_at_best = fn
                fp_at_best = fp

        precision_at_best = tp_at_best / (tp_at_best+fp_at_best) if tp_at_best > 0 else 0.0
        recall_at_best = tp_at_best / (tp_at_best+fn_at_best) if tp_at_best > 0 else 0.0
        fscore_at_best = 2 * precision_at_best * recall_at_best / (precision_at_best + recall_at_best) if recall_at_best > 0 else 0.0

        tp_total += tp_at_best
        fn_total += fn_at_best
        fp_total += fp_at_best
        print(c, threshold, precision_at_best, recall_at_best, fscore_at_best)

    accuracy = acc/(i + 1)
    precision = tp_total / (tp_total+fp_total)
    recall = tp_total / (tp_total+fn_total)

    fscore = 2 * precision * recall / (precision + recall)
    results = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": fscore,
        "loss": loss
    }
    logger.info("***** Eval results *****")
    for key in sorted(results.keys()):
        logger.info("  %s = %s", key, str(results[key]))
    
    return results
    