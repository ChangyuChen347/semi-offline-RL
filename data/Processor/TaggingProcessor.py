from . import register_processor
from transformers import AutoTokenizer
import numpy

from .BasicProcessor import BaseProcessor
from data.helper.formatter import Formatter

@register_processor("format_tagger")
class FormatTagger(BaseProcessor):
    def __init__(self,idx,out_name,model,cfg=None):
        self.idx = idx
        self.formatter = Formatter(model,cfg)
        self.out_key = ["input_ids","attention_mask","labels"]
    def process(self,columns):
        res = self.formatter.tagging(columns[self.idx[0]])
        return res
