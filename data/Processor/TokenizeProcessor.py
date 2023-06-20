from numpy.core.records import array
from . import register_processor
from transformers import AutoTokenizer
import numpy
from .BasicProcessor import BaseProcessor
import random
from transformers import logging
logger = logging.get_logger(__name__)
from typing import List 
import torch 
import numpy as np 
from trainer.Tools.task_utils import TaskInfo

random.seed(37)

@register_processor("selectorGen_tokenize")
class SELTokenize(BaseProcessor):
    def __init__(self,idx,out_name,model,cfg=None,task_cfg=None,**kwargs):
        super().__init__(cfg, model, **kwargs)
        self.idx = idx
        self.max_length = getattr(cfg,"tok_max_length",512) if cfg else 512
        self.max_target_length = getattr(cfg,"max_length", 128) if cfg else 128
        self.sen_sep = getattr(cfg,"sen_sep", ' #Z# ') if cfg else ' #Z# '
        self.out_key = ["input_ids","attention_mask","labels"]
        self.padding_values = [0,0,-100]
        if cfg.local_rank <= 0:
            self.fn.save_pretrained(cfg.output_dir)
    def process(self,columns):
        if len(self.idx) == 1:
            sens = columns[self.idx[0]].split(self.sen_sep)
            res = self.fn(sens,max_length=self.max_length,truncation=True)
        else:
            sens = columns[self.idx[0]].split(self.sen_sep)
            tgt = columns[self.idx[1]]

            res = self.fn(sens,max_length=self.max_length,truncation=True)
            with self.fn.as_target_tokenizer():
                labels = self.fn(tgt,return_tensors='np',max_length=self.max_target_length,truncation=True)
                res["labels"] = numpy.squeeze(labels["input_ids"],0)
        res["sentence_num"] = [len(res["input_ids"])]
        return res

@register_processor("cls_tokenize")
class CLSTokenize(BaseProcessor):
    def __init__(self,idx,out_name,model,cfg=None,task_cfg=None,**kwargs):
        super().__init__(cfg, model, **kwargs)
        self.idx = idx
        self.max_length = getattr(cfg,"tok_max_length",512) if cfg else 512
        self.out_key = ["input_ids","token_type_ids","attention_mask"]
        self.padding_values = [0,0,0]
        self.sen_sep = getattr(cfg,"sen_sep", ' #Z# ') if cfg else ' #Z# '
        if cfg.local_rank <= 0:
            self.fn.save_pretrained(cfg.output_dir)
    def process(self,columns):
        encode = []
        if len(self.idx) == 1:
            if self.sen_sep in columns[self.idx[0]]:
                encode = columns[self.idx[0]].split(self.sen_sep)[:2]
            else:
                encode = [columns[self.idx[0]]]
        else:
            encode = [columns[self.idx[0]],columns[self.idx[1]]]
        if len(encode) == 1:
            res = self.fn.encode_plus(encode[0],max_length=self.max_length,truncation=True,return_attention_mask=True)
        else:
            res = self.fn.encode_plus(text=encode[0],text_pair=encode[1],max_length=self.max_length,truncation=True,return_attention_mask=True)
        if "token_type_ids" not in res:
            res["token_type_ids"] = [0] * len(res["input_ids"])
        return res


@register_processor("twinbert_tokenize")
class S2STokenize(BaseProcessor):
    def __init__(self,idx,out_name,model,cfg=None,task_cfg=None,**kwargs):
        super().__init__(cfg, model, **kwargs)
        self.idx = idx
        self.max_length = getattr(cfg,"tok_max_length",512) if cfg else 512
        self.max_target_length = getattr(cfg,"max_length", 128) if cfg else 128
        if len(self.idx) == 1:
            self.out_key = ["q_input_ids","q_attention_mask"]
            self.padding_values = [0,0]
        else:
            self.out_key = ["q_input_ids","q_attention_mask","d_input_ids","d_attention_mask"]
            self.padding_values = [0,0,0,0]
        if cfg.local_rank <= 0:
            self.fn.save_pretrained(cfg.output_dir)

    def process(self,columns):
        def add_prefix_to_key(d, prefix):
            out_dict = {prefix + k: d[k] for k in d}
            return out_dict

        if len(self.idx) == 1:
            try:
                res = self.fn(columns[self.idx[0]],return_tensors='np',max_length=self.max_length,truncation=True)
                res = add_prefix_to_key(res, 'q_')
            except:
                logger.warning("Data Error")
                res = self.fn('',return_tensors='np',max_length=self.max_length,truncation=True)
        else:
            try:
                res = self.fn(columns[self.idx[0]],return_tensors='np',max_length=self.max_length,truncation=True)
                res = add_prefix_to_key(res, "q_")
                d_res = self.fn(columns[self.idx[1]],return_tensors='np',max_length=self.max_target_length,truncation=True)
                res.update(add_prefix_to_key(d_res, 'd_'))
            except:
                res = self.fn('',return_tensors='np',max_length=self.max_length,truncation=True)
                res = add_prefix_to_key(res, "q_")
                d_res = self.fn('',return_tensors='np',max_length=self.max_target_length,truncation=True)
                res.update(add_prefix_to_key(d_res, 'd_'))
                logger.warning("Data Error" + str(columns))
                
        return dict([(k,numpy.squeeze(v,0)) for k,v in res.items()])


@register_processor("s2s_tokenize")
class S2STokenize(BaseProcessor):
    def __init__(self,idx,out_name,model,cfg=None,task_cfg=None,**kwargs):
        super().__init__(cfg, model, **kwargs)
        self.idx = idx
        self.max_length = getattr(cfg,"tok_max_length",512) if cfg else 512
        self.max_target_length = getattr(cfg,"max_length", 128) if cfg else 128

        if len(self.idx) == 1:
            self.out_key = ["input_ids","attention_mask"]
            self.padding_values = [0,0]
        else:
            self.out_key = ["input_ids","attention_mask","labels"]
            self.padding_values = [0,0,-100]
        if cfg.local_rank <= 0:
            self.fn.save_pretrained(cfg.output_dir)
    def process(self,columns):
        if len(self.idx) == 1:
            try:
                res = self.fn(columns[self.idx[0]],return_tensors='np',max_length=self.max_length,truncation=True)
            except:
                logger.warning("Data Error")
                res = self.fn('',return_tensors='np',max_length=self.max_length,truncation=True)
        else:
            try:
            #if True:

                res = self.fn(columns[self.idx[0]],return_tensors='np',max_length=self.max_length,truncation=True)
                with self.fn.as_target_tokenizer():
                    labels = self.fn(columns[self.idx[1]],return_tensors='np',max_length=self.max_target_length,truncation=True)
                res["labels"] = labels["input_ids"]
            except:
                res = self.fn('',return_tensors='np',max_length=self.max_length,truncation=True)
                with self.fn.as_target_tokenizer():
                    labels = self.fn('',return_tensors='np',max_length=self.max_target_length,truncation=True)
                res["labels"] = labels["input_ids"]
                logger.warning("Data Error" + str(columns))
                
        return dict([(k,numpy.squeeze(v,0)) for k,v in res.items()])

# Multi-task optimization 
@register_processor("MTO_s2s_tokenize")
class MTO_S2STokenize(BaseProcessor):
    def __init__(self,idx,out_name,model,cfg=None,task_cfg=None,**kwargs):
        super().__init__(cfg, model, **kwargs)
        self.idx = idx
        self.max_length = getattr(cfg,"tok_max_length",512) if cfg else 512
        self.max_target_length = getattr(cfg,"max_length", 128) if cfg else 128

        if len(self.idx) == 1:
            self.out_key = ["input_ids","attention_mask"]
            self.padding_values = [0,0]
        else:
            self.out_key = ["input_ids","attention_mask","labels"]
            self.padding_values = [0,0,-100]
        if cfg.local_rank <= 0:
            self.fn.save_pretrained(cfg.output_dir)

        default_tasks = "EN,FR,DE,ES,SV,IT,NL"
        multi_tasks_from_config = getattr(cfg, "multi_tasks", default_tasks) if cfg else default_tasks
        self.tasks = multi_tasks_from_config.split(",")
        self.task_info = TaskInfo(self.tasks)

    def process(self,columns):
        """  
        columns is one line on the fly 
        """
        if len(self.idx) == 1:
            try:
                res = self.fn(columns[self.idx[0]],return_tensors='np',max_length=self.max_length,truncation=True)
            except:
                logger.warning("Data Error")
                res = self.fn('',return_tensors='np',max_length=self.max_length,truncation=True)
        else:
            try:
                res = self.fn(columns[self.idx[0]],return_tensors='np',max_length=self.max_length,truncation=True)
                with self.fn.as_target_tokenizer():
                    labels = self.fn(columns[self.idx[1]],return_tensors='np',max_length=self.max_target_length,truncation=True)
                task_label = self.task_info.convert_task_to_id(columns[self.idx[2]], return_tensor='')
                res["labels"] = labels["input_ids"]
                res["task_label"] = task_label
                
            except Exception as error :
                print(error)
                res = self.fn('',return_tensors='np',max_length=self.max_length,truncation=True)
                with self.fn.as_target_tokenizer():
                    labels = self.fn('',return_tensors='np',max_length=self.max_target_length,truncation=True)
                task_label = None
                res["labels"] = labels["input_ids"]
                res["task_label"] = task_label
                logger.warning("Data Error" + str(columns))
                
        return dict([(k,numpy.squeeze(v,0)) if isinstance(v, np.ndarray) else (k, v) for k,v in res.items()])

# Seq Ranking 
@register_processor("rank_tokenize")
class SeqRankTokenize(BaseProcessor) : 
    def __init__(self,idx,out_name,model,cfg=None,task_cfg=None,**kwargs):
        super().__init__(cfg, model, **kwargs)
        self.idx = idx
        self.max_length = getattr(cfg,"tok_max_length",512) if cfg else 512
        self.max_target_length = getattr(cfg,"max_length", 128) if cfg else 128

        self.task_cfg = task_cfg
        self.rank_num = self.task_cfg.rank_num
        self.boost_theta = self.task_cfg.boost_theta

        if self.task_cfg.rare_labels is not None: 
            self.rare_labels = self.task_cfg.rare_labels.split(",")

        elif self.task_cfg.rare_labels is None and self.task_cfg.rare_labels_threshold > 0: 
            from .constants import GuidedAdsGenConstant
            id_categories_to_map = GuidedAdsGenConstant.categories_map
            categories_to_id_map = {value: key for key, value in id_categories_to_map.items()}
            rare_labels_threshold = self.task_cfg.rare_labels_threshold
            self.rare_labels = [int(categories_to_id_map[cate]) for cate, emp_prob in GuidedAdsGenConstant.empirical_probs.items() if emp_prob <= rare_labels_threshold * 100]

        if len(self.idx) == 1:
            self.out_key = ["input_ids","attention_mask"]
            self.padding_values = [0,0]
        else:
            extra_out_key = []
            extra_padding_value = []
            self.out_key = ["input_ids","attention_mask","labels"]
            self.padding_values = [0,0,-100]
            
            if self.task_cfg.binary_rank: 
                extra_out_key += ["bin_labels"]
                extra_padding_value += [-100]
            self.out_key += extra_out_key
            self.padding_values += extra_padding_value
            
        if cfg.local_rank <= 0:
            self.fn.save_pretrained(cfg.output_dir)

    def cvt_probs_to_rankings(self, probs: str) : 
        """
        convert multi-asset prob sequences to rankings for one landing page 
        """
        prob_seqs = [prob_seq.replace('[', '').replace(']', '') for prob_seq in probs.split(",")]
        prob_mtx = np.array([[float(prob) for prob in probs.split()] for probs in prob_seqs])
        boost_prob_mtx = self.boost_prob(prob_mtx, self.boost_theta)
        ranking_score = self.compute_label_prob(boost_prob_mtx)
        sorted_indices = np.argsort(-ranking_score, axis=-1)
        raw_ranking_level = np.expand_dims(np.arange(self.rank_num)[::-1], 0) 
        _ranking_level = np.zeros((1, self.rank_num)) 
        np.put_along_axis(_ranking_level, sorted_indices, raw_ranking_level, -1)

        bin_rank_level = self.cvt_probs_to_bin_rankings(prob_mtx)

        return _ranking_level, bin_rank_level

    def cvt_probs_to_bin_rankings(self, probs: np.ndarray) : 
        bin_rank_probs = probs 
        bin_rank_probs = np.where(bin_rank_probs >= 0.5, 1., 0.)
        bin_rank_level = np.max(bin_rank_probs, keepdims=True, axis=0)

        return bin_rank_level

    def compute_label_prob(self, prob_mtx: np.ndarray) -> np.ndarray :
        confid_prob_mtx = 1 - prob_mtx * (prob_mtx > 0.5)
        label_prob = np.prod(confid_prob_mtx, axis=0, keepdims=True)

        return 1 - label_prob 

    def boost_prob(self, prob_max: np.ndarray, boost_theta: float) : 
        """
        boost probabilities for categories to balance the mass bias for high-empirical-prob categories
        for rare labels, we both 1) boost the probabilities to 1-\theta if with confidence
        and 2) boost the probabilities to second-to-last probabilities if without confidence
        """
        boost_seq = np.array([[1 if label in self.rare_labels else 0 for label in range(self.rank_num)]])
        _prob_max = prob_max 
        min_sort_prob_indices = np.argsort(_prob_max, axis=-1)[:, [0]]
        min2_sort_prob_indices = np.argsort(_prob_max, axis=-1)[:, [1]]
        min_mask = np.zeros_like(_prob_max)
        np.put_along_axis(min_mask, min_sort_prob_indices, 1, axis=-1)
        _prob_max[((_prob_max > 0.5) * boost_seq).astype(bool)] = 1 - boost_theta
        min2_sort_probs = np.take_along_axis(_prob_max, min2_sort_prob_indices, axis=-1)
        _prob_max = np.where((min_mask * boost_seq * (_prob_max <= 0.5)), min2_sort_probs, _prob_max)

        return _prob_max

    def process(self, columns) : 
        """
        tokenize source sequence and convert rank probs into rank labels 
        """
        if len(self.idx) == 1:
            try:
                res = self.fn(columns[self.idx[0]],return_tensors='np',max_length=self.max_length,truncation=True)
            except:
                logger.warning("Data Error")
                res = self.fn('',return_tensors='np',max_length=self.max_length,truncation=True)
        else:
            try:
                res = self.fn(columns[self.idx[0]],return_tensors='np',max_length=self.max_length,truncation=True)
                rank_labels, bin_rank_labels = self.cvt_probs_to_rankings(columns[self.idx[1]])
                res["labels"] = rank_labels 
                if self.task_cfg.binary_rank : res["bin_labels"] = bin_rank_labels
                
            except Exception as error :
                print(error)
                res = self.fn('',return_tensors='np',max_length=self.max_length,truncation=True)
                rank_labels, bin_rank_labels = self.cvt_probs_to_rankings(columns[self.idx[1]])
                res["labels"] = rank_labels 
                if self.task_cfg.binary_rank : res["bin_labels"] = bin_rank_labels
                logger.warning("Data Error" + str(columns))
    
        return dict([(k,numpy.squeeze(v,0)) for k,v in res.items()])

    
@register_processor("s2s_denoising")
class S2SDenosing(BaseProcessor):
    def __init__(self,idx,out_name,model,cfg=None,task_cfg=None,**kwargs):
        super().__init__(cfg, model, **kwargs)
        self.idx = idx
        self.max_length = getattr(cfg,"tok_max_length",512) if cfg else 512
        self.max_target_length = getattr(cfg,"max_length", 128) if cfg else 128
        self.noise_density = getattr(cfg,"noise_density", 0.15) if cfg else 0.15
        if len(self.idx) == 1:
            self.out_key = ["input_ids","attention_mask","labels"]
            self.padding_valuds = [0,0,-100]
        else:
            raise ValueError("only expecting one field for self-supervised denosing task")
        if cfg.local_rank <= 0:
            self.fn.save_pretrained(cfg.output_dir)

    def process(self,columns):
        text = columns[self.idx[0]]
        words = text.split()
        num_words = len(words)
        masked_word_indices = sorted(
            random.sample(range(num_words), int(num_words * self.noise_density)))
        src_words = []
        tgt_words = []
        extra_id = 0
        prev_masked_word_idx = None
        for i, w in enumerate(words):
            if i in masked_word_indices:
                if i - 1 == prev_masked_word_idx: # in the same span
                    tgt_words.append(w)
                else:
                    tgt_words.append(f"extra_id_{extra_id} {w}")
                    src_words.append(f"extra_id_{extra_id}")
                    extra_id += 1
                prev_masked_word_idx = i
            else:
                src_words.append(w)
        
        src = " ".join(src_words)
        tgt = " ".join(tgt_words)

        try:
            res = self.fn.prepare_seq2seq_batch(src, tgt_texts=tgt, return_tensors='np',
                                                max_length=self.max_length, max_target_length = self.max_target_length)
        except:
            print("Data Error")
            return None
        return dict([(k,numpy.squeeze(v,0)) for k,v in res.items()])
