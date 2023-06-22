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






