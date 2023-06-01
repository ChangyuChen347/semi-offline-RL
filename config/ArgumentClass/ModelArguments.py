from dataclasses import dataclass,field
from torch.utils import data
from transformers import (

    AutoModelForSequenceClassification,
    AutoModelForSeq2SeqLM,
    AutoModelForPreTraining,
)

from . import register_argumentclass

# Monkeypatch code to add twinbert into huggingface mapping
# so that AutoTokenizer and AutoConfig can load twinbert
# from models.twinbert import TwinBertConfig
from transformers.models.auto import configuration_auto
# configuration_auto.CONFIG_MAPPING["twinbert"] = TwinBertConfig
from transformers.models.auto import tokenization_auto
from transformers.models.bert.tokenization_bert import BertTokenizer
# tokenization_auto.TOKENIZER_MAPPING[TwinBertConfig] = (BertTokenizer, BertTokenizer)



class BaseArguments:
    def process(self):
        #return {"smile":"^_^"}
        return {}




@register_argumentclass("model_seq2seq")
@dataclass
class Seq2SeqArguments(BaseArguments):
#Generation
    auto_model = AutoModelForSeq2SeqLM
    max_length: int = field(default=20,metadata={"help":"Longest sequence to be generated"})
    min_length: int = field(default=5, metadata={"help":"Shortest sequence to be generated"})
    num_beams:  int = field(default=1, metadata={"help":"beam size"})
    num_beam_groups:  int = field(default=1, metadata={"help":"diverse beam"})
    repetition_penalty: float = field(default=1.0,metadata={"help":"factor to supress repetition"})
    length_penalty:     float = field(default=1.0,metadata={"help":"factor to penalize short result"})
    num_return_sequences: int = field(default=1,metadata={"help":"number of return sequences"})
    no_repeat_ngram_size: int = field(default=0,metadata={"help":"set no repeat ngram"})
    early_stopping: bool = field(default=True,metadata={"help":"early stop for decoding"})
    diversity_penalty: float=field(default=0,metadata={"help":"diverse beam"})
    rl_weight: float = field(default=0.8,metadata={"help":"weight for rl"})
    mask_input: bool = field(default=True)
    inv_out_mask: bool = field(default=False)
    mask_rate: float = field(default=0.5)
    out_mask_rate: float = field(default=0.5)
    io_not_same_mask: bool = field(default=False)
    use_cont_mask_id: bool = field(default=False)
    lm_head2: bool = field(default=False)
    do_parallel_test_model: bool = field(default=False)
    increase_id: str = field(default='fix_decrease')
    decoding_method: str=field(default='seq')
    tail_unmask_num: int=field(default=0)
    truth_log_probs: bool=field(default=False)
    use_all_probs: bool=field(default=False)
    tokenizer_name: str=field(default='google/mt5-small')
    cand_pos_remove_sp_tk: bool=field(default=True)
    span_mask: bool=field(default=True)
    sample_num: int=field(default=0)
    sample_method: str=field(default='loop')
    non_seq_cmp_with_max: bool = field(default=True)
    normalize: bool=field(default=False)
    normalize_penalty: int=field(default=2)
    keep_prob: float=field(default=0)
    random_prob: float=field(default=0)
    not_mask_stop: bool=field(default=False)
    sample_topk: int=field(default=-1)
    do_rl: bool = field(default=True)
    use_max_baseline: bool=field(default=False)
    use_logit: bool=field(default=False)
    prob_w: bool=field(default=False)
    ppo: bool=field(default=False)
    nmask_comma: bool = field(default=True)
    nmask_next_comma: bool = field(default=True)
    fix_mask: bool=field(default=True)
    v2_0401: bool=field(default=False)

