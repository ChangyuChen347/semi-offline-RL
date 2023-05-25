import re
import random
import numpy as np
from configparser import ConfigParser

import torch
from torch import nn
#import rouge
import time

from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED, as_completed, ALL_COMPLETED
from automatic_evaluation_tool import text_normalization
from rouge_score import rouge_scorer
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL', 'rougeLsum'], use_stemmer=True)
scorer_dict = {'12': rouge_scorer.RougeScorer(['rouge1', 'rouge2'], use_stemmer=True), '12l': rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True),'12sum': rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeLsum'], use_stemmer=True)}
from threading import Thread
import torch.nn.functional as F
from nltk.translate.meteor_score import meteor_score
from decode_seq2seq import detokenize

from sklearn.utils.extmath import softmax

from multiprocessing import Pool
import nltk
from access_dict_by_dot import AccessDictByDot
import rouge

from automatic_evaluation_tool.bleu.bleu import Bleu as bleu_squad
bleu_squad = bleu_squad()
from automatic_evaluation_tool.rouge import Rouge as rouge_squad
rouge_squad = rouge_squad()
# from automatic_evaluation_tool.meteor.meteor import Meteor as meteor_squad
# meteor_squad = meteor_squad()
scorer_pool = [rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True) for _ in range(100)]
scorer_12sum_pool = [rouge_scorer.RougeScorer(['rouge1', 'rouge2',  'rougeLsum'], use_stemmer=True) for _ in range(100)]

scorer_12_pool = [rouge_scorer.RougeScorer(['rouge1', 'rouge2'], use_stemmer=True) for _ in range(100)]
evaluator_m_pool = [rouge.Rouge(metrics=['rouge-n', 'rouge-l'], max_n=2,
                                      apply_avg=False) for _ in range(100)]
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from multiprocessing import Array
all_res_dict = Array('d', [float(x) for x in range(32*64)])
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL', 'rougeLsum'], use_stemmer=True)
def get_rouge_sum(zip_output_tgt_id):

    zip_output_tgt, task_id = zip_output_tgt_id

    local_output_lines = [e[0] for e in zip_output_tgt]
    local_tgt = [e[1] for e in zip_output_tgt]
    local_idx = [e[2] for e in zip_output_tgt]

    r1s = []
    rls = []
    r2s = []
    output_lines = [e.strip() for e in local_output_lines]
    raw_tgt = [e.strip() for e in local_tgt]
    for hyp, ref in zip(output_lines, raw_tgt):
        res = scorer_pool[task_id].score(hyp, ref)
        r1 = res['rouge1'].fmeasure
        r1s.append(r1)
        r2 = res['rouge2'].fmeasure
        r2s.append(r2)
        rl = res['rougeL'].fmeasure
        rls.append(rl)
    for i in range(len(r1s)):
        r1s[i] = (r1s[i] + rls[i] + r2s[i]) / 3
    # all_res_dict.append(r1s)
    for i, sc in enumerate(r1s):
        all_res_dict[local_idx[i]] = sc

def get_rouge_sum_12sum(zip_output_tgt_id):

    zip_output_tgt, task_id = zip_output_tgt_id

    local_output_lines = [e[0] for e in zip_output_tgt]
    local_tgt = [e[1] for e in zip_output_tgt]
    local_idx = [e[2] for e in zip_output_tgt]

    r1s = []
    rls = []
    r2s = []
    output_lines = [e.strip() for e in local_output_lines]
    raw_tgt = [e.strip() for e in local_tgt]
    for hyp, ref in zip(output_lines, raw_tgt):
        res = scorer_12sum_pool[task_id].score(hyp, ref)
        r1 = res['rouge1'].fmeasure
        r1s.append(r1)
        r2 = res['rouge2'].fmeasure
        r2s.append(r2)
        rl = res['rougeLsum'].fmeasure
        rls.append(rl)
    for i in range(len(r1s)):
        r1s[i] = (r1s[i] + rls[i] + r2s[i]) / 3

    for i, sc in enumerate(r1s):
        all_res_dict[local_idx[i]] = sc

def get_rouge_sum_12(zip_output_tgt_id):
    # global all_res_dict
    zip_output_tgt, task_id = zip_output_tgt_id
    # print(len(zip_output_tgt))
    local_output_lines = [e[0] for e in zip_output_tgt]
    local_tgt = [e[1] for e in zip_output_tgt]
    local_idx = [e[2] for e in zip_output_tgt]

    r1s = []
    rls = []
    r2s = []
    output_lines = [e.strip() for e in local_output_lines]
    raw_tgt = [e.strip() for e in local_tgt]
    for hyp, ref in zip(output_lines, raw_tgt):
        res = scorer_12_pool[task_id].score(hyp, ref)
        r1 = res['rouge1'].fmeasure
        r1s.append(r1)
        r2 = res['rouge2'].fmeasure
        r2s.append(r2)
    for i in range(len(r1s)):
        r1s[i] = (r1s[i] + r2s[i]) / 2
    # all_res_dict.append(r1s)
    for i, sc in enumerate(r1s):
        all_res_dict[local_idx[i]] = sc
    # print(all_res_dict)




from nltk.util import ngrams
from functools import reduce
import operator
import os
import time
import datasets
# sacrebleu = datasets.load_metric('sacrebleu')
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED, as_completed

from transformers.models.roberta.tokenization_roberta import RobertaTokenizer
from transformers.models.roberta.configuration_roberta import RobertaConfig

def timer(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        res = func(*args, **kwargs)
        print('unirl 共耗时约 {:.2f} 秒'.format(time.time() - start))
        return res

    return wrapper

def timer_r(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        res = func(*args, **kwargs)
        print('reward 共耗时约 {:.4f} 秒'.format(time.time() - start))
        return res
    return wrapper

def RankingLoss(score, summary_score=None, margin=0.001, gold_margin=0, gold_weight=1, no_gold=True, no_cand=False, fix_margin=False, not_normal_score=None):
    ones = torch.ones_like(score)
    loss_func = torch.nn.MarginRankingLoss(0.0)
    TotalLoss = loss_func(score, score, ones)
    #print(TotalLoss)
    # candidate loss
    n = score.size(1)
    # if p_1 - p_2 > 0.001
    #   L = 0
    # else:
    #   L = p_1 - p_2 - 0.001
    if not no_cand:
        for i in range(1, n):
            pos_score = score[:, :-i]
            neg_score = score[:, i:]
            if not_normal_score is not None:
                not_normal_pos_score = not_normal_score[:, :-i]
                not_normal_neg_score = not_normal_score[:, i:]
                not_normal_pos_score = not_normal_pos_score.contiguous().view(-1)  # bs
                not_normal_neg_score = not_normal_neg_score.contiguous().view(-1)  # bs
            pos_score = pos_score.contiguous().view(-1) # bs
            neg_score = neg_score.contiguous().view(-1)
            ones = torch.ones_like(pos_score)
            loss_func = torch.nn.MarginRankingLoss(margin*i)

            loss = loss_func(pos_score, neg_score, ones) * pos_score.shape[0] # some samples are not optimized. those have
            # print(pos_score, neg_score, loss)
            TotalLoss += loss
    if no_gold:
        return TotalLoss
    # gold summary loss

    pos_score = summary_score.unsqueeze(-1).expand_as(score)
    neg_score = score
    pos_score = pos_score.contiguous().view(-1)
    neg_score = neg_score.contiguous().view(-1)
    ones = torch.ones_like(pos_score)
    loss_func = torch.nn.MarginRankingLoss(gold_margin)
    #print(gold_weight)
    TotalLoss += gold_weight * loss_func(pos_score, neg_score, ones)
    return TotalLoss


class PolicyGradientForUniLM(nn.Module):

    def __init__(self, pc_model_recover_path, tokenizer, max_target_seq_len, impose_way, token_replaces=None, special_token_ids=None, mlm_reward=False,
                 mlm_weight=0.0, force_replace=False, reward_type='rouges',
                 metric_type='none', distinct_normalize=False, rewards='pc2', rewards_weight='1', adv_weight=1, rouge_type='l', sample_num=0,
                 punish_non_click_reward=False, local_rank=-1, cand_num=1, loss_type=None, margin=0.001, pred_type=''):
        super().__init__()
        self.distinct_normalize = distinct_normalize
        self.reward_type = reward_type
        self.metric_type = metric_type
        self.adv_weight = adv_weight
        self.rouge_type = rouge_type

        self.sample_num =sample_num
        self.cand_num = cand_num
        self.loss_type = loss_type
        self.pred_type = pred_type

        self.rewards = rewards.split(',')
        self.margin = margin


        if 'fact' in self.rewards:
            from summac.model_summac import SummaCConv, model_map
            model_name = 'anli'
            fact_model = SummaCConv(models=[model_name], device='cuda', granularity="sentence", \
                               start_file='summac/summac_conv_vitc_sent_perc_e.bin', bins="percentile")
            fact_model.load_state_dict(torch.load('summac/ckpt/model.best.pt')['model'])
            fact_model.eval()
            self.fact_model = fact_model
        self.pool = ThreadPoolExecutor(max_workers=16)
        self.cur_rewards_weight = rewards_weight.split(',')
        self.cur_rewards_weight = [float(t) for t in self.cur_rewards_weight]
        self.tokenizer = tokenizer
        self.max_target_seq_len = max_target_seq_len
        self.punish_non_click_reward = punish_non_click_reward
        if token_replaces is None:
            token_replaces = {}
        self.repl = token_replaces
        if special_token_ids is None:
            special_token_ids = []
        self.special_token_ids = special_token_ids
        self.dynamic_tokens = []
        self.dynamic_tokens_lambda = []

    @classmethod
    def from_config(cls, config_path, tokenizer,
                    max_target_seq_len, impose_way, token_replaces, special_token_ids,
                    reward_type='pc', distinct_normalize=False, rewards='pc2', rewards_weight='1',
                    adv_weight=1, rouge_type='l', sample_num=0, punish_non_click_reward=False,
                    local_rank=-1, cand_num=1, loss_type=None, margin=0.001, pred_type=''):
        cfg = ConfigParser()
        cfg.read(config_path)

        return cls(
            tokenizer, max_target_seq_len, impose_way, token_replaces, special_token_ids,
              distinct_normalize=distinct_normalize,
            rewards=rewards, rewards_weight=rewards_weight, adv_weight=adv_weight, rouge_type=rouge_type, sample_num=sample_num, punish_non_click_reward=punish_non_click_reward, local_rank=local_rank, cand_num=cand_num, loss_type=loss_type, margin=margin, pred_type=pred_type)

    def get_rouge_reawrd(self, querys, raw_src, raw_tgt, traces, b_traces, log_probs, raw_tgt_tk, traces_tk, b_traces_tk, tokenizer, zero_traces, b_zero_traces, labels_traces, probs, b_probs, rouge_type=None, base_traces=None):

        base_reward = None
        output_lines = traces
        b_output_lines = b_traces
        base_output_lines = base_traces
        if ((rouge_type is None and self.rouge_type != 2) or (rouge_type is not None and rouge_type != '2')) and 'remove_gt_rouge' in self.metric_type:
            output_lines = zero_traces
            b_output_lines = b_zero_traces
            raw_tgt = labels_traces
        raw_tgt = [output_line.replace('<s>', ' ') for output_line in raw_tgt]
        raw_tgt = [output_line.replace('</s>', ' ') for output_line in raw_tgt]
        raw_tgt = [output_line.replace('.', ' <SEP> ') for output_line in raw_tgt]
        raw_tgt = [output_line.replace(',', ' <SEP2> ') for output_line in raw_tgt]
        raw_tgt = [output_line.lower().strip() for output_line in raw_tgt]
        output_lines = [output_line.replace('</s>', ' ') for output_line in output_lines]
        output_lines = [output_line.replace('<s>', ' ') for output_line in output_lines]
        output_lines = [output_line.replace('<pad>', ' ') for output_line in output_lines]
        output_lines = [output_line.replace('-TitleSep-', ' ') for output_line in output_lines]
        output_lines = [output_line.replace('-Desc-', ' ') for output_line in output_lines]
        output_lines = [output_line.replace('.', ' <SEP> ') for output_line in output_lines]
        output_lines = [output_line.replace(',', ' <SEP2> ') for output_line in output_lines]
        # print(output_lines)
        output_lines = [output_line.lower().strip() for output_line in output_lines]
        if rouge_type == 'xsum' or rouge_type == 'cnn':
            output_lines = ["\n".join(nltk.sent_tokenize(x.strip())) for x in output_lines]
        # output_lines = [" ".join(nltk.word_tokenize(x.strip())) for x in output_lines]
        b_output_lines = [b_output_line.replace('</s>', ' ') for b_output_line in b_output_lines]
        b_output_lines = [b_output_line.replace('<s>', ' ') for b_output_line in b_output_lines]
        b_output_lines = [b_output_line.replace('<pad>', ' ') for b_output_line in b_output_lines]
        b_output_lines = [b_output_line.replace('-TitleSep-', ' ') for b_output_line in b_output_lines]
        b_output_lines = [b_output_line.replace('-Desc-', ' ') for b_output_line in b_output_lines]
        b_output_lines = [b_output_line.replace('.', ' <SEP> ') for b_output_line in b_output_lines]
        b_output_lines = [b_output_line.replace(',', ' <SEP2> ') for b_output_line in b_output_lines]

        b_output_lines = [output_line.lower().strip() for output_line in b_output_lines]
        if base_output_lines is not None:
            base_output_lines = [b_output_line.replace('</s>', ' ') for b_output_line in base_output_lines]
            base_output_lines = [b_output_line.replace('<s>', ' ') for b_output_line in base_output_lines]
            base_output_lines = [b_output_line.replace('<pad>', ' ') for b_output_line in base_output_lines]
            base_output_lines = [b_output_line.replace('-TitleSep-', ' ') for b_output_line in base_output_lines]
            base_output_lines = [b_output_line.replace('-Desc-', ' ') for b_output_line in base_output_lines]
            base_output_lines = [b_output_line.replace('.', ' <SEP> ') for b_output_line in base_output_lines]
            base_output_lines = [b_output_line.replace(',', ' <SEP2> ') for b_output_line in base_output_lines]
            base_output_lines = [output_line.lower() for output_line in base_output_lines]

        if self.sample_num != 0 or self.cand_num != 1:
            sample_tgt = [t for t in raw_tgt for _ in range(self.cand_num*(self.sample_num+1))]
            rouge_list = self.get_rouge(output_lines, sample_tgt, rouge_type=rouge_type)
        else:
            rouge_list = self.get_rouge(output_lines,  raw_tgt, rouge_type=rouge_type)

        if len(b_output_lines) != len(raw_tgt):
            rep_num = len(b_output_lines) // len(raw_tgt)
            sample_tgt = [t for t in raw_tgt for _ in range(rep_num)]
            b_rouge_list = self.get_rouge(b_output_lines, sample_tgt, rouge_type=rouge_type)
        else:
            b_rouge_list = self.get_rouge(b_output_lines,  raw_tgt, rouge_type=rouge_type)

        if self.sample_num != 0 or self.cand_num != 1:
            sample_q = [t for t in querys for _ in range(self.cand_num*(self.sample_num+1))]
            sample_t = [t for t in raw_tgt for _ in range(self.cand_num*(self.sample_num+1))]
            sample_r = [t for t in raw_src for _ in range(self.cand_num*(self.sample_num+1))]
            reward = self.get_rouge_reward_tensor(sample_q, output_lines, sample_r, sample_t, rouge_list)
        else:
            reward = self.get_rouge_reward_tensor(querys, output_lines, raw_src, raw_tgt, rouge_list)
        if len(b_output_lines) != len(raw_tgt):
            sample_q = [t for t in querys for _ in range(len(b_output_lines) // len(querys))]
            sample_t = [t for t in raw_tgt for _ in range(len(b_output_lines) // len(raw_tgt))]
            sample_r = [t for t in raw_src for _ in range(len(b_output_lines) // len(raw_src))]
            b_reward = self.get_rouge_reward_tensor(sample_q, b_output_lines, sample_r, sample_t, b_rouge_list)
        else:
            b_reward = self.get_rouge_reward_tensor(querys, b_output_lines, raw_src, raw_tgt, b_rouge_list)

        return reward, b_reward, base_reward


    def get_squad_rouge_bleu_reawrd(self, querys, raw_src, raw_tgt, traces, b_traces, log_probs, raw_tgt_tk, traces_tk, b_traces_tk, tokenizer, zero_traces, b_zero_traces, labels_traces, probs, b_probs, rouge_type=None, refs=None):

        output_lines = traces
        b_output_lines = b_traces
        if ((rouge_type is None and self.rouge_type != 2) or (rouge_type is not None and rouge_type != '2')) and 'remove_gt_rouge' in self.metric_type:
            output_lines = zero_traces
            b_output_lines = b_zero_traces
            raw_tgt = labels_traces

        raw_tgt = refs

        raw_tgt = [[t.lower() for t in e.split('<#REF#>')] for e in raw_tgt]
        output_lines = [output_line.replace('</s>', ' ') for output_line in output_lines]
        output_lines = [output_line.replace('<s>', ' ') for output_line in output_lines]
        output_lines = [output_line.replace('<pad>', ' ') for output_line in output_lines]
        output_lines = [output_line.replace('-TitleSep-', ' ') for output_line in output_lines]
        output_lines = [output_line.replace('-Desc-', ' ') for output_line in output_lines]
        output_lines = [output_line.replace('.', ' <SEP> ') for output_line in output_lines]
        output_lines = [output_line.replace(',', ' <SEP2> ') for output_line in output_lines]
        # print(output_lines)
        output_lines = [output_line.lower() for output_line in output_lines]
        if rouge_type == 'xsum' or rouge_type == 'cnn':
            output_lines = ["\n".join(nltk.sent_tokenize(x.strip())) for x in output_lines]
        # output_lines = [" ".join(nltk.word_tokenize(x.strip())) for x in output_lines]
        b_output_lines = [b_output_line.replace('</s>', ' ') for b_output_line in b_output_lines]
        b_output_lines = [b_output_line.replace('<s>', ' ') for b_output_line in b_output_lines]
        b_output_lines = [b_output_line.replace('<pad>', ' ') for b_output_line in b_output_lines]
        b_output_lines = [b_output_line.replace('-TitleSep-', ' ') for b_output_line in b_output_lines]
        b_output_lines = [b_output_line.replace('-Desc-', ' ') for b_output_line in b_output_lines]
        b_output_lines = [b_output_line.replace('.', ' <SEP> ') for b_output_line in b_output_lines]
        b_output_lines = [b_output_line.replace(',', ' <SEP2> ') for b_output_line in b_output_lines]

        # b_output_lines = [" ".join(nltk.word_tokenize(x.strip())) for x in b_output_lines]
        b_output_lines = [output_line.lower() for output_line in b_output_lines]

        base_output_lines = [b_output_line.replace('</s>', ' ') for b_output_line in base_output_lines]
        base_output_lines = [b_output_line.replace('<s>', ' ') for b_output_line in base_output_lines]
        base_output_lines = [b_output_line.replace('<pad>', ' ') for b_output_line in base_output_lines]
        base_output_lines = [b_output_line.replace('-TitleSep-', ' ') for b_output_line in base_output_lines]
        base_output_lines = [b_output_line.replace('-Desc-', ' ') for b_output_line in base_output_lines]
        base_output_lines = [b_output_line.replace('.', ' <SEP> ') for b_output_line in base_output_lines]
        base_output_lines = [b_output_line.replace(',', ' <SEP2> ') for b_output_line in base_output_lines]

        # base_output_lines = [" ".join(nltk.word_tokenize(x.strip())) for x in base_output_lines]
        base_output_lines = [output_line.lower() for output_line in base_output_lines]

        if rouge_type == 'xsum' or rouge_type == 'cnn':
            b_output_lines = ["\n".join(nltk.sent_tokenize(x.strip())) for x in b_output_lines]
            raw_tgt = ["\n".join(nltk.sent_tokenize(x.strip())) for x in raw_tgt]

        if self.sample_num != 0 or self.cand_num != 1:
            sample_tgt = [t for t in raw_tgt for _ in range(self.cand_num*(self.sample_num+1))]
            rouge_list = self.get_squad_rouge_bleu(output_lines, sample_tgt, rouge_type=rouge_type)
        else:
            rouge_list = self.get_squad_rouge_bleu(output_lines,  raw_tgt, rouge_type=rouge_type)
        if len(b_output_lines) != len(raw_tgt):
            rep_num = len(b_output_lines) // len(raw_tgt)
            sample_tgt = [t for t in raw_tgt for _ in range(rep_num)]
            b_rouge_list = self.get_squad_rouge_bleu(b_output_lines, sample_tgt, rouge_type=rouge_type)
        else:
            b_rouge_list = self.get_squad_rouge_bleu(b_output_lines,  raw_tgt, rouge_type=rouge_type)

        if len(base_output_lines) != len(raw_tgt):
            rep_num = len(base_output_lines) // len(raw_tgt)
            sample_tgt = [t for t in raw_tgt for _ in range(rep_num)]
            base_rouge_list = self.get_squad_rouge_bleu(base_output_lines, sample_tgt, rouge_type=rouge_type)
        else:
            base_rouge_list = self.get_squad_rouge_bleu(base_output_lines,  raw_tgt, rouge_type=rouge_type)


        if self.sample_num != 0 or self.cand_num != 1:
            sample_q = [t for t in querys for _ in range(self.cand_num*(self.sample_num+1))]
            sample_t = [t for t in raw_tgt for _ in range(self.cand_num*(self.sample_num+1))]
            sample_r = [t for t in raw_src for _ in range(self.cand_num*(self.sample_num+1))]
            reward = self.get_rouge_reward_tensor(sample_q, output_lines, sample_r, sample_t, rouge_list)
        else:
            reward = self.get_rouge_reward_tensor(querys, output_lines, raw_src, raw_tgt, rouge_list)
        if len(b_output_lines) != len(raw_tgt):
            sample_q = [t for t in querys for _ in range(len(b_output_lines) // len(querys))]
            sample_t = [t for t in raw_tgt for _ in range(len(b_output_lines) // len(raw_tgt))]
            sample_r = [t for t in raw_src for _ in range(len(b_output_lines) // len(raw_src))]
            b_reward = self.get_rouge_reward_tensor(sample_q, b_output_lines, sample_r, sample_t, b_rouge_list)
        else:
            b_reward = self.get_rouge_reward_tensor(querys, b_output_lines, raw_src, raw_tgt, b_rouge_list)

        if len(base_output_lines) != len(raw_tgt):
            sample_q = [t for t in querys for _ in range(len(base_output_lines) // len(querys))]
            sample_t = [t for t in raw_tgt for _ in range(len(base_output_lines) // len(raw_tgt))]
            sample_r = [t for t in raw_src for _ in range(len(base_output_lines) // len(raw_src))]
            base_reward = self.get_rouge_reward_tensor(sample_q, base_output_lines, sample_r, sample_t, base_rouge_list)
        else:
            base_reward = self.get_rouge_reward_tensor(querys, base_output_lines, raw_src, raw_tgt, base_rouge_list)


        if probs is not None:
            reward = torch.mul(reward, probs.squeeze(-1))
        if b_probs is not None:
            b_reward = torch.mul(b_reward, b_probs.squeeze(-1))
        return reward, b_reward, base_reward





    # @timer
    def forward(self, querys, raw_src, raw_tgt, traces, b_traces, log_probs, raw_tgt_tk, traces_tk, b_traces_tk, tokenizer, zero_traces, b_zero_traces, labels_traces, probs=None, b_probs=None, truth_log_probs=None, pre_gen_scores=None,
                predict_baseline=None, steps=None, not_normal_log_probs=None,
                model=None, input_ids=None, y_s=None, y_b=None, inputs_brio=None, refs=None, base_traces=None):
        #self.reward_count_cache = {}
        device = log_probs.device
        output_lines = traces
        b_output_lines = b_traces
        all_reward = []
        all_b_reward = []
        all_ori_b_reward = []
        all_base_reward = []
        # print(self.rewards)
        base_reward = None
        for i, reward_name in enumerate(self.rewards):
            if reward_name == 'rouge':
                reward, b_reward = self.get_rouge_reawrd(querys, raw_src, raw_tgt, traces, b_traces, log_probs, raw_tgt_tk, traces_tk, b_traces_tk, tokenizer, zero_traces, b_zero_traces, labels_traces, probs, b_probs)
            elif reward_name == 'squad_rouge_bleu':
                reward, b_reward, base_reward = self.get_squad_rouge_bleu_reawrd(querys, raw_src, raw_tgt, traces, b_traces,
                                                                   log_probs,
                                                                   raw_tgt_tk, traces_tk, b_traces_tk, tokenizer,
                                                                   zero_traces,
                                                                   b_zero_traces, labels_traces, probs, b_probs,
                                                                   refs=refs)


            reward = reward * self.cur_rewards_weight[i]
            ori_b_reward = b_reward
            b_reward = b_reward * self.cur_rewards_weight[i]
            all_reward.append(reward)
            all_b_reward.append(b_reward)
            all_ori_b_reward.append(ori_b_reward)
            if base_reward is not None:
                all_base_reward.append(base_reward)
        if pre_gen_scores is not None:
            all_reward = pre_gen_scores

        else:
            all_reward = torch.stack(all_reward, dim=1) #bs, rewards

        all_b_reward = torch.stack(all_b_reward, dim=1)
        all_ori_b_reward = torch.stack(all_ori_b_reward, dim=1)
        if len(all_base_reward) != 0:
            all_base_reward = torch.stack(all_base_reward, dim=1)
        all_b_reward_return = all_ori_b_reward.clone()

        all_reward = all_reward.sum(-1)
        all_b_reward = all_b_reward.sum(-1)

        if self.sample_num != 0 or self.cand_num != 1:
            all_reward = all_reward.reshape(-1, self.cand_num*(self.sample_num+1))
            if self.loss_type == 'cl':
                ind = torch.argsort(all_reward, dim=1, descending=True)
                log_probs = log_probs.sum(-1).reshape(-1, self.cand_num*(self.sample_num+1))
                log_probs = torch.gather(log_probs, 1, ind)
                rl_loss = RankingLoss(log_probs, margin=self.margin)
            else:
                log_probs = log_probs.sum(-1).reshape(-1, self.cand_num * (self.sample_num + 1))
                if 'not_normal' in self.loss_type:
                    not_normal_log_probs = not_normal_log_probs.sum(-1).reshape(-1, self.cand_num * (self.sample_num + 1))
                bs = log_probs.shape[0]
                all_reward_c = all_reward.reshape(bs, -1, 1).repeat(1, 1, self.cand_num * (self.sample_num + 1))
                all_reward_r = all_reward.reshape(bs, 1, -1).repeat(1, self.cand_num * (self.sample_num + 1), 1)
                all_reward_m = all_reward_c - all_reward_r  # bs, n, n
                all_reward_cmp = all_reward_c > all_reward_r
                log_probs_c = log_probs.detach().clone().reshape(bs, -1, 1).repeat(1, 1, self.cand_num * (
                            self.sample_num + 1))
                log_probs_r = log_probs.detach().clone().reshape(bs, 1, -1).repeat(1, self.cand_num * (
                            self.sample_num + 1), 1)
                if 'margin' in self.loss_type:
                    rank_index = torch.argsort(all_reward, dim=1, descending=True).cuda() * 0.001
                    all_index_c = rank_index.reshape(bs, -1, 1).repeat(1, 1, self.cand_num * (self.sample_num + 1))
                    all_index_r = rank_index.reshape(bs, 1, -1).repeat(1, self.cand_num * (self.sample_num + 1), 1)
                    all_index_m = all_index_c - all_index_r
                    log_probs_m = log_probs_c - log_probs_r + all_index_m > 0
                else:
                    log_probs_m = log_probs_c > log_probs_r
                log_probs_m = all_reward_cmp != log_probs_m
                log_probs = log_probs.reshape(bs, 1, -1)
                if 'order' in self.loss_type:
                    all_reward_m = all_reward_m * log_probs_m
                if 'order_dist' in self.loss_type:
                    all_reward_m = torch.ones_like(all_reward_m).cuda() * (all_reward_m > 0) - torch.ones_like(
                        all_reward_m).cuda() * (all_reward_m < 0)
                    all_reward_m = all_reward_m * log_probs_m
                if 'not_normal' in self.loss_type:
                    not_normal_log_probs = not_normal_log_probs.reshape(bs, 1, -1)
                    rl_loss = torch.bmm(not_normal_log_probs, all_reward_m).squeeze(2).sum(-1)  # bs, 1, n
                    rl_loss = -rl_loss.mean()
                else:
                    rl_loss = torch.bmm(log_probs, all_reward_m).squeeze(2).sum(-1)  # bs, 1, n
                    rl_loss = -rl_loss.mean()
        else:
            assert 1==0

        return rl_loss, all_b_reward_return, all_reward, all_base_reward


    def get_rouge_sum(self, output_lines, raw_tgt):
        r1s = []
        rls = []
        r2s = []
        # rlsums = []
        # print(self.rouge_type)
        rouge_metric = []
        if '1' in self.rouge_type:
            rouge_metric.append('rouge1')
        if '2' in self.rouge_type:
            rouge_metric.append('rouge2')
        if 'l' in self.rouge_type:
            rouge_metric.append('rougeL')
        scorer = scorer_dict[self.rouge_type]
        # scorer = rouge_scorer.RougeScorer(rouge_metric, use_stemmer=True)
        output_lines = [e.strip() for e in output_lines]
        raw_tgt =  [e.strip() for e in raw_tgt]
        for hyp, ref in zip(output_lines, raw_tgt):
            res = scorer.score(hyp, ref)
            if '1' in self.rouge_type:
                r1 = res['rouge1'].fmeasure
                r1s.append(r1)
            if '2' in self.rouge_type:
                r2 = res['rouge2'].fmeasure
                r2s.append(r2)
            # rlsum = res['rougeLsum'].fmeasure
            if 'l' in self.rouge_type:
                rl = res['rougeL'].fmeasure
                rls.append(rl)
            if 'sum' in self.rouge_type:
                rl = res['rougeLsum'].fmeasure
                rls.append(rl)
        if self.rouge_type == '1':
            return r1s
        elif self.rouge_type == '2':
            return r2s
        elif self.rouge_type == 'l':
            # print(len(rls))
            return rls
        elif self.rouge_type == '12':
            for i in range(len(r1s)):
                r1s[i] = (r1s[i] + r2s[i]) / 2
            return r1s
        elif self.rouge_type == '12l':
            for i in range(len(r1s)):
                r1s[i] = (r1s[i] + rls[i] + r2s[i]) / 3
            return r1s
        elif self.rouge_type == '12sum':
            for i in range(len(r1s)):
                r1s[i] = (r1s[i] + rls[i] + r2s[i]) / 3
            return r1s
        elif self.rouge_type == 'cnn':
            for i in range(len(r1s)):
                r1s[i] = (r1s[i] + rlsums[i] + r2s[i])
            return r1s
        elif self.rouge_type == '12f1':
            for i in range(len(r1s)):
                if r1s[i] == 0 and r2s[i] == 0:
                    r1s[i] = 0
                else:
                    r1s[i] = 2*r1s[i]*r2s[i]/(r1s[i]+r2s[i]+1e-8)
            return r1s

    def get_squad_rouge_bleu(self, output_lines, raw_tgt, rouge_type=None):
        metrics = []
        if rouge_type is not None:
            if '1' in rouge_type or '2' in rouge_type:
                metrics.append('rouge-n')
            if 'l' in rouge_type:
                metrics.append('rouge-l')
        else:
            if '1' in self.rouge_type or '2' in self.rouge_type:
                metrics.append('rouge-n')
            if 'l' in self.rouge_type:
                metrics.append('rouge-l')
        evaluator_m = rouge.Rouge(metrics=metrics, max_n=2,
                                  apply_avg=False)
        def eval_multi(gold_str_list, pre_str_list, rouge_type=None,trace_type=None):
            scores = evaluator_m.get_scores(pre_str_list, [it for it in gold_str_list])
            # print(gold_str_list[0])
            res = []
            if rouge_type is not None:
                if rouge_type == 'mean':
                    for fpr in scores['rouge-{}'.format('l')]:
                        res.append(fpr['f'][0] / 3)
                    for i, fpr in enumerate(scores['rouge-{}'.format('1')]):
                        res[i] += fpr['f'][0] / 3
                    for i, fpr in enumerate(scores['rouge-{}'.format('2')]):
                        res[i] += fpr['f'][0] / 3
                elif rouge_type == '12l':
                    for fpr in scores['rouge-{}'.format('l')]:
                        res.append(fpr['f'][0] / 3)
                    for i, fpr in enumerate(scores['rouge-{}'.format('1')]):
                        res[i] += fpr['f'][0] / 3
                    for i, fpr in enumerate(scores['rouge-{}'.format('2')]):
                        res[i] += fpr['f'][0] / 3
                elif rouge_type == '12':
                    for i, fpr in enumerate(scores['rouge-{}'.format('1')]):
                        res.append(fpr['f'][0] / 2)
                    for i, fpr in enumerate(scores['rouge-{}'.format('2')]):
                        res[i] += fpr['f'][0] / 2
                else:
                    for fpr in scores['rouge-{}'.format(rouge_type)]:
                        res.append(fpr['f'][0])
            elif self.rouge_type == '12':
                for fpr in scores['rouge-{}'.format('1')]:
                    res.append(fpr['f'][0] / 2)
                for i, fpr in enumerate(scores['rouge-{}'.format('2')]):
                    res[i] += fpr['f'][0] / 2
            elif self.rouge_type != 'mean':
                for fpr in scores['rouge-{}'.format(self.rouge_type)]:
                    res.append(fpr['f'][0])
            elif self.rouge_type == 'mean':
                for fpr in scores['rouge-{}'.format('l')]:
                    res.append(fpr['f'][0] / 3)
                for i, fpr in enumerate(scores['rouge-{}'.format('1')]):
                    res[i] += fpr['f'][0] / 3
                for i, fpr in enumerate(scores['rouge-{}'.format('2')]):
                    res[i] += fpr['f'][0] / 3
            return res
        res = eval_multi(raw_tgt, output_lines, rouge_type=rouge_type)

        # res = [0 for _ in range(len(output_lines))]
        to_eval_hyp_dict = {i: [output_lines[i].strip().lower().encode()] for i in range(len(output_lines))}
        # print(raw_tgt[0])
        to_eval_ref_dict = {i: [e.strip().lower().encode() for e in raw_tgt[i]] for i in
                            range(len(raw_tgt))}
        # _, rouge_scores = rouge_squad.compute_score(to_eval_ref_dict, to_eval_hyp_dict)
        _, bleu_scores = bleu_squad.compute_score(to_eval_ref_dict, to_eval_hyp_dict)
        bleu4 = bleu_scores[3]
        # rougel = rouge_scores.tolist()
        for i in range(len(res)):
            res[i] = res[i] / 2 + bleu4[i] / 2

        return res

    def get_rouge(self, output_lines, raw_tgt, rouge_type=None):
        if self.reward_type == 'rouges':
            if len(output_lines) <= 256:
                res2 = self.get_rouge_sum(output_lines, raw_tgt)
                return res2
            # # print(res2)



            # print(len(output_lines))
            else:
                # start = time.time()
                tasks_size = 32
                task_num = (len(output_lines) + tasks_size - 1) // tasks_size
                # tasks_size = (len(output_lines) + task_num - 1) // task_num
                # print('CPU核的数量：', cpu_count())
                with Pool(processes=4) as pool:
                    # zip_output_tgt = zip(output_lines, raw_tgt, [i for i in range(len(output_lines))])
                    zip_output_tgt = [(output_lines[i], raw_tgt[i], i) for i in range(len(output_lines))]
                    tasks = [[zip_output_tgt[task_id * tasks_size:(task_id + 1) * tasks_size], task_id] for task_id in
                             range(task_num)]
                    # zip_output_tgt = output_lines
                    if self.rouge_type == '12':
                        for _ in pool.imap_unordered(get_rouge_sum_12, tasks):
                            pass
                    elif self.rouge_type == '12l':
                        for _ in pool.imap_unordered(get_rouge_sum, tasks):
                            pass
                    elif self.rouge_type == '12sum':
                        for _ in pool.imap_unordered(get_rouge_sum_12sum, tasks):
                            pass
                res = [all_res_dict[i] for i in range(len(output_lines))]

                return res


        else:
            metrics = []
            if rouge_type is not None:
                if '1' in rouge_type or '2' in rouge_type:
                    metrics.append('rouge-n')
                if 'l' in rouge_type:
                    metrics.append('rouge-l')
            else:
                if '1' in self.rouge_type or '2' in self.rouge_type:
                    metrics.append('rouge-n')
                if 'l' in self.rouge_type:
                    metrics.append('rouge-l')
            evaluator_m = rouge.Rouge(metrics=metrics, max_n=2,
                                      apply_avg=False)
            def eval_multi(gold_str_list, pre_str_list, rouge_type=None,trace_type=None):
                scores = evaluator_m.get_scores(pre_str_list, [[it] for it in gold_str_list])
                res = []
                if rouge_type is not None:
                    if rouge_type == 'mean':
                        for fpr in scores['rouge-{}'.format('l')]:
                            res.append(fpr['f'][0] / 3)
                        for i, fpr in enumerate(scores['rouge-{}'.format('1')]):
                            res[i] += fpr['f'][0] / 3
                        for i, fpr in enumerate(scores['rouge-{}'.format('2')]):
                            res[i] += fpr['f'][0] / 3
                    elif rouge_type == '12l':
                        for fpr in scores['rouge-{}'.format('l')]:
                            res.append(fpr['f'][0] / 3)
                        for i, fpr in enumerate(scores['rouge-{}'.format('1')]):
                            res[i] += fpr['f'][0] / 3
                        for i, fpr in enumerate(scores['rouge-{}'.format('2')]):
                            res[i] += fpr['f'][0] / 3
                    elif rouge_type == '12':
                        for i, fpr in enumerate(scores['rouge-{}'.format('1')]):
                            res.append(fpr['f'][0] / 2)
                        for i, fpr in enumerate(scores['rouge-{}'.format('2')]):
                            res[i] += fpr['f'][0] / 2
                    else:
                        for fpr in scores['rouge-{}'.format(rouge_type)]:
                            res.append(fpr['f'][0])
                elif self.rouge_type == '12':
                    for fpr in scores['rouge-{}'.format('1')]:
                        res.append(fpr['f'][0] / 2)
                    for i, fpr in enumerate(scores['rouge-{}'.format('2')]):
                        res[i] += fpr['f'][0] / 2
                elif self.rouge_type != 'mean':
                    for fpr in scores['rouge-{}'.format(self.rouge_type)]:
                        res.append(fpr['f'][0])
                elif self.rouge_type == 'mean':
                    for fpr in scores['rouge-{}'.format('l')]:
                        res.append(fpr['f'][0] / 3)
                    for i, fpr in enumerate(scores['rouge-{}'.format('1')]):
                        res[i] += fpr['f'][0] / 3
                    for i, fpr in enumerate(scores['rouge-{}'.format('2')]):
                        res[i] += fpr['f'][0] / 3
                return res

            res = eval_multi(raw_tgt, output_lines, rouge_type=rouge_type)

            return res
    def get_squad_rouge_bleu(self, output_lines, raw_tgt, rouge_type=None):
        res = [0 for _ in range(len(output_lines))]
        to_eval_hyp_dict = {i:[output_lines[i].strip().lower().encode()]  for i in range(len(output_lines))}
        # print(raw_tgt[0])
        to_eval_ref_dict = {i: [e.strip().lower().encode() for e in raw_tgt[i].split('<#REF#>')] for i in range(len(raw_tgt))}
        _, rouge_scores = rouge_squad.compute_score(to_eval_ref_dict, to_eval_hyp_dict)
        _, bleu_scores = bleu_squad.compute_score(to_eval_ref_dict, to_eval_hyp_dict)
        # _, meteor_scores = meteor_squad.compute_score(to_eval_ref_dict, to_eval_hyp_dict)
        bleu4 = bleu_scores[3]
        rougel = rouge_scores.tolist()
        # meteor_scores = meteor_scores.tolist()
        for i in range(len(res)):
            res[i] = rougel[i] / 3 + bleu4[i] / 3 * 2# + meteor_scores[i] / 3
        return res
    def get_rouge_reward_tensor(self, querys, output_lines, raw_src, raw_tgt, rouge_list, punish=True):
        reward = torch.tensor(rouge_list, dtype=torch.float).cuda()
        if punish and self.punish_non_click_reward:
            pc_lines, reward_weight = self.pc_preprocessor(querys, output_lines, raw_src, raw_tgt)
            if self.impose_way == "add":
                reward = reward - (1 - torch.tensor(reward_weight).to(reward.device))
            else:
                reward = reward * torch.tensor(reward_weight).to(reward.device)
        return reward
