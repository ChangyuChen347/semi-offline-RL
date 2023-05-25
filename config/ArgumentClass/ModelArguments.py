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

@register_argumentclass("model_seq2label")
@dataclass
class Seq2LabelArguments(BaseArguments):
    auto_model = AutoModelForSequenceClassification
    num_labels: int = field(default=1,metadata={"help":"num class for multiclass classfication"})
    num_tasks:  int = field(default=1,metadata={"help":"num task for multitask classification"})
    default_task: int = field(default=0,metadata={"help":"default task id"})
    cls_mode: str = field(default="multi_task",metadata={"help":"multi_task or multi_class"})
    loss_rce_beta: float = field(default=0.0, metadata= {"help":"hyper weight for ce"})
    problem_type: str = field(default=None, metadata= {"help": "None (default)|regression|single_label_classification|multi_label_classification; default will try to infer problem type from labels"})

@register_argumentclass("model_seq2label_unilm")
@dataclass
class UniLMSeq2Label(BaseArguments):
    rel_pos_type: int = field(default=0,metadata={"help":"The rel postion type, 0 for disable"})
    rel_pos_bins: int = field(default=0,metadata={"help":"The config for rel_pos_type"})
    max_rel_pos: int = field(default=0,metadata={"help":"Max Relative position"})
    num_labels: int = field(default=1,metadata={"help":"num class for multiclass classfication"})
    fast_qkv: bool = field(default=False,metadata={"help":"Don't know"})
    classifier_dropout: float = field(default=0.1,metadata={"help":"dropout for last layer"})

@register_argumentclass("model_singletwinbert_classificationx")
@dataclass
class SingleTwinBertClassificationXArguments(BaseArguments):
    num_labels: int = field(default=2,metadata={"help":"num class for multiclass classfication"})
    embed_dim: int = field(default=32,metadata={"help":"output embedding dimension size"})
    temperature: float = field(default=1.0,metadata={"help":"temperature in softmax"})
    freeze: bool = field(default=True,metadata={"help":"freeze encoder weight"})
    num_layers: int = field(default=None,metadata={"help":"Num of layers to use, default None will use all layers"})
    tanh_activation: bool = field(default=False,metadata={"help":"apply tanh activation for embedding"})

@register_argumentclass("model_twinbert_classificationx")
@dataclass
class TwinBertClassificationXArguments(BaseArguments):
    temperature: float = field(default=1.0,metadata={"help":"temperature in softmax"})
    num_layers: int = field(default=None,metadata={"help":"Num of layers to use, default None will use all layers"})
    downscale_pooler: bool = field(default=True,metadata={"help":"Downscale BERT hidden representation to smaller dim before classification"})
    post_xing_layers: int = field(default=0,metadata={"help":"if > 0, number of hidden layers after crossing layer; defaul=0 no hidden layer"})
    post_xing_dim: int = field(default=192,metadata={"help":"dimension of hiddem layer"})
    load_xing_params: bool = field(default=True,metadata={"help":"try to load parameters of crossing layer from ckpt"})





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
@register_argumentclass("model_seq2tag")
@dataclass
class Seq2TagArguments(BaseArguments):
    last_layer: str = field(default="fwd",metadata={"help":"Last layer type, support fwd, transformer"})
    num_labels: int = field(default=2,metadata={"help":"num tags"})

@register_argumentclass("model_pretrain")
@dataclass
class PretrainArguments(BaseArguments):
    mlm_prob: float = field(default=0.15,metadata={"help":"Mask ratio"})

@register_argumentclass("model_guided_seq2seq")
@dataclass
class GuidedSeq2SeqArguments(BaseArguments):
    latent_class: int = field(default=4,metadata={"help":"Set Latent Class"})
    selected_class: int = field(default=2,metadata={"help":"Set predict Latent Class count"})
    guided_mode: str = field(default="enc_dec",metadata={"help":"Added latent signal to enc, enc_dec or dec"})
    kl_loss_weight: float = field(default="0.0",metadata={"help":"Weight of KL loss"})
    cls_loss_weight: float = field(default="0.0",metadata={"help":"Weight of CLS loss"})
    em: str = field(default="hard",metadata={"help":"hard for hard-em, soft for soft-em"})
    max_length: int = field(default=20,metadata={"help":"Longest sequence to be generated"})
    min_length: int = field(default=5, metadata={"help":"Shortest sequence to be generated"})
    num_beams:  int = field(default=1, metadata={"help":"beam size"})
    repetition_penalty: float = field(default=1.0,metadata={"help":"factor to supress repetition"})
    length_penalty:     float = field(default=1.0,metadata={"help":"factor to penalize short result"})
    num_return_sequences: int = field(default=1,metadata={"help":"number of return sequences"})

@register_argumentclass("model_selector_seq2seq")
@dataclass
class SelectorSeq2SeqArguments(BaseArguments):
    sample_pos: int = field(default=1, metadata={"help": "The sample position. 0: before enc; 1: after enc"})
    sample_num: int = field(default=2,metadata={"help":"The number of sentence smapled"})
    sample_loss_weight: float = field(default="0.0",metadata={"help":"Weight of loss of sample classifier"})
    latent_num: int = field(default=1,metadata={"help":"The number of latent variables"})
    latent_loss_weight: float = field(default="0.0",metadata={"help":"Weight of loss of latent classifier"})
    latent_enc_fusion: str = field(default="concat",metadata={"help":"The way to fusion the inputs emb and labels emb"})
    max_length: int = field(default=20,metadata={"help":"Longest sequence to be generated"})
    min_length: int = field(default=1, metadata={"help":"Shortest sequence to be generated"})
    num_beams:  int = field(default=1, metadata={"help":"beam size"})
    repetition_penalty: float = field(default=1.0,metadata={"help":"factor to supress repetition"})
    length_penalty:     float = field(default=1.0,metadata={"help":"factor to penalize short result"})
    num_return_sequences: int = field(default=1,metadata={"help":"number of return sequences"})
    no_repeat_ngram_size: int = field(default=0,metadata={"help":"set no repeat ngram"})
    early_stopping: bool = field(default=True,metadata={"help":"early stop for decoding"})

@register_argumentclass("model_multi_task_opt_seq2seq")
@dataclass
class MultiTaskOptArguments(Seq2SeqArguments):
    use_history_weight: bool = field(default=False, metadata={"help": "if task_weight is based on previous history or only current one"})
    rank_alpha: float = field(default=0.2, metadata={"help": "hyperparameter for ranking sampler instances"})
    task_scoring_method: str = field(default="batch_task_dist", metadata={"help": "either contant, batch_task_dist or task_loss"})
    dist_alpha: float = field(default=1, metadata={"help": "adjust task distribution weight"})
    loss_alpha: float = field(default=0., metadata={"help": "adjust loss contribution to sampling weight"})

