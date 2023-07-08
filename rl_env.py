import re
import random
import numpy as np
from configparser import ConfigParser

import torch
from torch import nn
#import rouge
import time

from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED, as_completed, ALL_COMPLETED
from lmqg.automatic_evaluation_tool import text_normalization
from rouge_score import rouge_scorer

from threading import Thread
import torch.nn.functional as F
from nltk.translate.meteor_score import meteor_score
from sklearn.utils.extmath import softmax
from multiprocessing import Pool
import nltk
from access_dict_by_dot import AccessDictByDot
import rouge
from lmqg.automatic_evaluation_tool.bleu.bleu import Bleu as bleu_squad
bleu_squad = bleu_squad()
from lmqg.automatic_evaluation_tool.rouge import Rouge as rouge_squad
rouge_squad = rouge_squad()
scorer_pool = [rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True) for _ in range(100)]
scorer_dict = {'12l': rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)}
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from multiprocessing import Array
all_res_dict = Array('d', [float(x) for x in range(32*64)])
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
    for i, sc in enumerate(r1s):
        all_res_dict[local_idx[i]] = sc




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

class RL_env(nn.Module):
    def __init__(self,
                 tokenizer,

                 metric_type='none',
                 distinct_normalize=False,
                 rewards='pc2',
                 rouge_type='12l',
                 sample_num=0,
                 local_rank=-1,
                 cand_num=1,
                 loss_type=None,
                 margin=0.001,
                 ):
        super().__init__()

        self.metric_type = metric_type
        self.rouge_type = rouge_type
        self.sample_num =sample_num
        self.cand_num = cand_num
        self.loss_type = loss_type
        self.rewards = rewards.split(',')
        self.margin = margin
        self.tokenizer = tokenizer


    def preprocess_lines(self, output_line):
        return output_line.\
            replace('<pad>', ' ').\
            replace('<s>', ' ').\
            replace('</s>', ' ').\
            replace('.', ' <SEP> ').\
            replace(',', ' <SEP2> ').\
            lower().\
            strip()

    def get_rouge_reawrd(self,
                         querys,
                         raw_src,
                         raw_tgt,
                         traces,
                         b_traces,
                         rouge_type=None,
                         ):
        output_lines = traces
        b_output_lines = b_traces
        raw_tgt = [self.preprocess_lines(output_line) for output_line in raw_tgt]
        output_lines = [self.preprocess_lines(output_line) for output_line in output_lines]
        b_output_lines = [self.preprocess_lines(output_line) for output_line in b_output_lines]

        if self.sample_num != 0 or self.cand_num != 1:
            sample_tgt = [t for t in raw_tgt for _ in range(self.cand_num*(self.sample_num+1))]
            rouge_list = self.get_rouge(output_lines, sample_tgt, rouge_type=rouge_type)
        else:
            rouge_list = self.get_rouge(output_lines, raw_tgt, rouge_type=rouge_type)

        if len(b_output_lines) != len(raw_tgt):
            rep_num = len(b_output_lines) // len(raw_tgt)
            sample_tgt = [t for t in raw_tgt for _ in range(rep_num)]
            b_rouge_list = self.get_rouge(b_output_lines, sample_tgt, rouge_type=rouge_type)
        else:
            b_rouge_list = self.get_rouge(b_output_lines,  raw_tgt, rouge_type=rouge_type)

        reward = self.get_reward_tensor(rouge_list)
        b_reward = self.get_reward_tensor(b_rouge_list)
        return reward, b_reward

    def get_squad_rouge_bleu_reawrd(self, querys, raw_src, raw_tgt, traces, b_traces,
                                    rouge_type=None, refs=None):
        output_lines = traces
        b_output_lines = b_traces
        raw_tgt = refs
        output_lines = [self.preprocess_lines(output_line) for output_line in output_lines]
        b_output_lines = [self.preprocess_lines(output_line) for output_line in b_output_lines]
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
        reward = self.get_reward_tensor(rouge_list)
        b_reward = self.get_reward_tensor(b_rouge_list)
        return reward, b_reward

    def get_str(self, y, ori_line_split=None):
        pre_output_ids = y.tolist()
        output_ids = []
        for i in range(len(pre_output_ids)):
            output_id = []
            tot = 0
            for j in range(len(pre_output_ids[i])):
                if pre_output_ids[i][j] == -100:
                    break
                output_id.append(pre_output_ids[i][j])
                tot += 1
            output_ids.append(output_id)
        traces = [self.tokenizer.decode(output_ids[i], skip_special_tokens=False, clean_up_tokenization_spaces=False)
                  for i in range(len(output_ids))]
        return traces

    def get_label_str(self, y):
        pre_output_ids = y.tolist()
        output_ids = []
        for i in range(len(pre_output_ids)):
            output_id = []
            for j in range(0, len(pre_output_ids[i])):
                if pre_output_ids[i][j] == -100:
                    break
                output_id.append(pre_output_ids[i][j])
            output_ids.append(output_id)

        traces = [self.tokenizer.decode(output_ids[i], skip_special_tokens=False, clean_up_tokenization_spaces=False)
                  for i in range(len(output_ids))]
        return traces

    def get_tks(self, y):
        pre_output_ids = y.tolist()
        output_ids = []

        for i in range(len(pre_output_ids)):
            output_id = []
            for j in range(len(pre_output_ids[i])):
                if pre_output_ids[i][j] == -100:
                    break
                output_id.append(pre_output_ids[i][j])
            output_ids.append(output_id)
        tks = [
            self.tokenizer.convert_ids_to_tokens(output_ids[i]) for i in
            range(len(output_ids))]
        return tks

    def rl_step(self,
                y_b,
                y_s,
                max_ids,
                masked_ids,
                input_ids,
                labels,
                log_probs,
                querys,
                non_zero_sum_tensor=None,
                not_normal_log_probs=None,
                raw_src=None,
                refs=None,):

        target_mask = ~labels.data.eq(-100)
        b_traces = self.get_str(y_b)
        traces = self.get_str(y_s)
        max_ids_tk = max_ids[:, :-1].tolist()
        all_gen_b_traces_tk = [
            self.tokenizer.convert_ids_to_tokens(max_ids_tk[i]) for i in
            range(len(max_ids_tk))]
        masked_ids_tk = masked_ids[:, :-1].tolist()
        all_gen_traces_tk = [
            self.tokenizer.convert_ids_to_tokens(masked_ids_tk[i]) for i in
            range(len(masked_ids_tk))]
        all_gen_b_traces_tk = [
            self.tokenizer.convert_ids_to_tokens(y_b[i]) for i in
            range(len(y_b))]
        all_gen_traces_tk = [
            self.tokenizer.convert_ids_to_tokens(y_s[i]) for i in
            range(len(y_s))]
        raw_tgt = self.get_label_str(labels)
        raw_tgt_tk = self.get_tks(labels)
        rl_loss, b_reward_dict, all_reward = self.forward(querys, raw_src, raw_tgt, traces, b_traces,
                                                         log_probs,
                                                         not_normal_log_probs=not_normal_log_probs,
                                                         refs=refs)
        return rl_loss, b_reward_dict,  all_gen_traces_tk, all_gen_b_traces_tk, all_reward

    def forward(self, querys, raw_src, raw_tgt, traces, b_traces, log_probs,
                 not_normal_log_probs=None,
                 refs=None):
        device = log_probs.device
        output_lines = traces
        b_output_lines = b_traces
        all_reward = []
        all_b_reward = []
        all_base_reward = []
        base_reward = None
        for i, reward_name in enumerate(self.rewards):
            if reward_name == 'rouge':
                reward, b_reward = self.get_rouge_reawrd(querys, raw_src, raw_tgt, traces, b_traces)
            elif reward_name == 'squad_rouge_bleu':
                reward, b_reward = self.get_squad_rouge_bleu_reawrd(querys, raw_src, raw_tgt, traces, b_traces,refs=refs)
            else:
                raise NotImplementedError("the reward_name not implemented yet.")
            all_reward.append(reward)
            all_b_reward.append(b_reward)

        all_reward = torch.stack(all_reward, dim=1) #bs, rewards
        all_b_reward = torch.stack(all_b_reward, dim=1)
        if len(all_base_reward) != 0:
            all_base_reward = torch.stack(all_base_reward, dim=1)
        all_b_reward_return = all_b_reward.clone()
        all_reward = all_reward.sum(-1)
        all_b_reward = all_b_reward.sum(-1)
        if self.sample_num != 0 or self.cand_num != 1:
            all_reward = all_reward.reshape(-1, self.cand_num*(self.sample_num+1))
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
                # default rl loss
                rl_loss = torch.bmm(log_probs, all_reward_m).squeeze(2).sum(-1)  # bs, 1, n
                rl_loss = -rl_loss.mean()
        else:
            raise NotImplementedError("the sample_num or cand_num not implemented yet.")
        return rl_loss, all_b_reward_return, all_reward

    def get_rouge_sum(self, output_lines, raw_tgt):
        r1s = []
        rls = []
        r2s = []
        rouge_metric = []
        if '1' in self.rouge_type:
            rouge_metric.append('rouge1')
        if '2' in self.rouge_type:
            rouge_metric.append('rouge2')
        if 'l' in self.rouge_type:
            rouge_metric.append('rougeL')
        scorer = scorer_dict[self.rouge_type]
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
            if 'l' in self.rouge_type:
                rl = res['rougeL'].fmeasure
                rls.append(rl)
        if self.rouge_type == '12l':
            for i in range(len(r1s)):
                r1s[i] = (r1s[i] + rls[i] + r2s[i]) / 3
            return r1s
        else:
            raise NotImplementedError("the rouge_type not implemented yet.")

    def get_rouge(self, output_lines, raw_tgt, rouge_type=None):

        if len(output_lines) <= 256:
            res2 = self.get_rouge_sum(output_lines, raw_tgt)
            return res2
        else:
            tasks_size = 32
            task_num = (len(output_lines) + tasks_size - 1) // tasks_size
            with Pool(processes=4) as pool:
                zip_output_tgt = [(output_lines[i], raw_tgt[i], i) for i in range(len(output_lines))]
                tasks = [[zip_output_tgt[task_id * tasks_size:(task_id + 1) * tasks_size], task_id] for task_id in
                         range(task_num)]
                if self.rouge_type == '12l':
                    for _ in pool.imap_unordered(get_rouge_sum, tasks):
                        pass
                else:
                    raise NotImplementedError("the rouge_type not implemented yet.")
            res = [all_res_dict[i] for i in range(len(output_lines))]
            return res


    def get_squad_rouge_bleu(self, output_lines, raw_tgt, rouge_type=None): # only for squad
        res = [0 for _ in range(len(output_lines))]
        to_eval_hyp_dict = {i:[output_lines[i].strip().lower().encode()]  for i in range(len(output_lines))}
        to_eval_ref_dict = {i: [e.strip().lower().encode() for e in raw_tgt[i].split('<#REF#>')] for i in range(len(raw_tgt))}
        _, rouge_scores = rouge_squad.compute_score(to_eval_ref_dict, to_eval_hyp_dict)
        _, bleu_scores = bleu_squad.compute_score(to_eval_ref_dict, to_eval_hyp_dict)
        bleu4 = bleu_scores[3]
        rougel = rouge_scores.tolist()
        for i in range(len(res)):
            res[i] = rougel[i] / 3 + bleu4[i] / 3 * 2
        return res
    def get_reward_tensor(self, rouge_list):
        reward = torch.tensor(rouge_list, dtype=torch.float).cuda()
        return reward

    def add_padding_(self, raw_txt, pad_id, max_len=256):
        txts_ids = [self.tokenizer.encode(txt) for txt in raw_txt]
        for t in txts_ids:
            assert len(t) != 0
        padding_txts_ids = []
        batch_max_seq_len = max([len(txt) for txt in txts_ids])
        batch_max_seq_len = min(batch_max_seq_len, max_len)
        for txt_ids in txts_ids:
            padding_txts_ids.append(
                txt_ids[:batch_max_seq_len] + [pad_id] * (batch_max_seq_len - len(txt_ids[:batch_max_seq_len])))
        return torch.tensor(padding_txts_ids, dtype=torch.long).cuda()
