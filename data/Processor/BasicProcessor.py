from . import register_processor
from transformers import AutoTokenizer
import numpy
class BaseProcessor:
    def __init__(self, cfg, model, **kwargs):
        self.out_key = []
        self.padding_values = []
        tokenizer = kwargs.pop("tokenizer", None)
        self.fn = tokenizer if tokenizer else AutoTokenizer.from_pretrained(model, cache_dir=cfg.cache_dir)
    def property(self):
        return {
                "values":dict([key,value] for key,value in zip(self.out_key,self.padding_values)) if self.padding_values else {}
                }
@register_processor("basic")
class SelectColumn(BaseProcessor):
    def __init__(self,idx,out_name,model=None,cfg = None,task_cfg=None):
        self.idx = idx
        self.out_key = [out_name]
        self.out_name = out_name
        self.padding_values = None

    def process(self,columns):

        return {self.out_name:columns[self.idx]}

@register_processor("convert2int")
class ConvertColumn(BaseProcessor):
    def __init__(self,idx,out_name,model=None,cfg = None,task_cfg=None):
        self.idx = idx
        self.out_key = [out_name]
        self.out_name = out_name
        self.padding_values=[None]
    def process(self,columns):
        return {self.out_name:int(columns[self.idx])}


@register_processor("label_mapping")
class LabelMapping(BaseProcessor):
    def __init__(self,idx,out_name,model=None,cfg=None,task_cfg=None):
        self.idx = idx[0]
        self.out_key = [out_name]
        self.out_name = out_name
        self.mapping = dict()
        self.padding_values = [None]
        for s in cfg.label_mapping.split(","):
            k,v = s.split(":")
            self.mapping[k] = int(v) if "." not in v else float(v)
    def process(self,columns):
        return {self.out_name:self.mapping[columns[self.idx]]}


@register_processor("convert2float")
class ConvertColumn(BaseProcessor):
    def __init__(self,idx,out_name,model=None,cfg = None,task_cfg=None):
        self.idx = idx[0]
        self.out_key = [out_name]
        self.out_padding = []
        self.out_name = out_name
        self.padding_values = [None]

    def process(self,columns):
        return {self.out_name:float(columns[self.idx])}


@register_processor("convert2floatlist")
class ConvertColumn(BaseProcessor):
    def __init__(self,idx,out_name,model=None,cfg = None,task_cfg=None):
        self.idx = idx[0]
        self.out_key = [out_name]
        self.out_padding = [-1]
        self.out_name = out_name
        self.padding_values = [None]

    def process(self,columns):
        """
        assuming a list of float in the format: 0.5,0.2,0.3
        """
        col = columns[self.idx]
        col = col.replace(",", " ")
        vals = [float(x) for x in col.split()]
        return {self.out_name:vals}

@register_processor("convert2multilabel")
class Sparse2Dense(BaseProcessor):
    def __init__(self,idx,out_name,model=None,cfg=None,task_cfg=None):
        self.idx = idx[0]
        self.out_key = [out_name]
        self.out_padding = [-1]
        self.out_name = out_name
        self.padding_values = [None]
        self.num_labels = len(cfg.label_mapping.split(','))
        self.label_mapping = {}
        for s in cfg.label_mapping.split(","):
            k,v = s.split(":")
            self.label_mapping[k] = int(v) if "." not in v else float(v)

    def process(self,columns):
        col = columns[self.idx]
        col = col.split(',')
        vals = [0.0 for i in range(0,self.num_labels)]
        for c in col:
            vals[self.label_mapping[c]] = 1.0
        return {self.out_name:vals}



