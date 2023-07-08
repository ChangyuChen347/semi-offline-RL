from dataclasses import dataclass,field
from . import register_argumentclass
from transformers import TrainingArguments

@register_argumentclass("base")
@dataclass
class BaseArguments:
    scene: str = field(default="realtime_gen",metadata={"help":"Specific Scene"})

@register_argumentclass("train")
@dataclass
class TrainArguments(TrainingArguments):
#preprocess_header: processor:col0:col1:col2:...:keyname
    model: str = field(default="local",metadata={"help":"Either local or model identifier from huggingface.co/models"})
    task: str = field(default="",metadata={"help":"Model Task"})
    train_dir: str = field(default="../Data/train.txt",metadata={"help":"Training data path"})
    train_header: str = field(default="",metadata={"help":"Training data header"})
    train_model_header: str = field(default="",metadata={"help":"Training preprocess header"})
    eval_dir: str = field(default="../Data/eval.txt",metadata={"help":"validation data path"})
    eval_header: str = field(default="",metadata={"help":"Eval data header"})
    eval_model_header: str = field(default="",metadata={"help":"Eval preprocess header"})
    predict_header: str = field(default="",metadata={"help":"Predict data header"})
    predict_model_header: str = field(default="",metadata={"help":"Predict preprocess header"})
    result_header: str = field(default="",metadata={"help":"Result output header"})
    result_file: str = field(default="predict.txt",metadata={"help":"Result file name"})


    previous_dir: str = field(default="./cache", metadata={"help": "previous model path"})
    output_dir: str = field(default="./model_output", metadata={"help": "Output data path"})
    cache_dir: str = field(default="./cache", metadata={"help": "Path to cache auto models"})
    logging_dir: str = field(default="./log_output", metadata={"help": "Path to save logging"})
    from_tf: bool = field(default=False, metadata={"help": "load model from tf weight"})
    datareader_streaming: bool = field(default=False, metadata={"help": "whether to load data in streaming way"})
    dataloader_num_workers: int = field(default=3, metadata={"help": "reader workers"})
    outputter_num_workers: int = field(default=None,
                                       metadata={"help": "outputter workers, default (None) to dataloader_num_workers"})
    extract_chunk: int = field(default=10, metadata={"help": "param for pool.imap, chunks processes for one time"})
    shuffle_num_batch: int = field(default=50, metadata={"help": "N batch for shuffle buffer"})
    num_train_epochs: int = field(default=1, metadata={"help": "N epochs to train"})
    logging_steps: int = field(default=200, metadata={"help": "default step to logging (print, azureml, tensorboard)"})
    save_steps: int = field(default=5000, metadata={"help": "default step to save ckpt, should be same as eval_steps"})
    save_total_limit: int = field(default=2, metadata={"help": "save total limit"})
    eval_steps: int = field(default=5000, metadata={"help": "evaluation every N steps"})
    evaluation_strategy: str = field(default="steps", metadata={"help": "evaluation strategy: no/steps/epoch"})
    load_best_model_at_end: bool = field(default=False, metadata={"help": "load best model at the end for save"})
    greater_is_better: bool = field(default=True, metadata={"help": "help to judge best model"})
    fp16: bool = field(default=False, metadata={"help": "whether to enable mix-precision"})
    remove_unused_columns: bool = field(default=False, metadata={
        "help": "Remove columns not required by the model when using an nlp.Dataset."})
    disable_tqdm: bool = field(default=True, metadata={"help": "Disable tqdm, print log instead"})
    # Train Setting
    trainer: str = field(default="common", metadata={"help": "user-defined trainer to select"})
    eval_metrics: str = field(default="acc", metadata={"help": "Metrics to eval model with delimiter ,"})
    evaluate_during_training: bool = field(default=True, metadata={"help": "Wheter to enable evaluation during training"})
    to_cache: str = field(default="", metadata={"help": "To hotfix fields pop issue in huggingface"})
    output_type: str = field(default="score", metadata={"help": "Set outputter type"})
    eager_predict: bool = field(default=False, metadata={"help": "Eager prediction for debugging"})
    dense_gradients: bool = field(default=False, metadata={"help": "Sync dense gradient for efficiency"})
    compress_gradients: bool = field(default=False, metadata={"help": "Sync gradient in fp16"})

    label_names: str = field(default=None, metadata={"help": "label names for label ids"})
    # Preprocessor
    # label_mapping: str = field(default="1:1,0:0", metadata={"help": "labelname:label,..."})
    tok_max_length: int = field(default=1024, metadata={"help": "max length for tokenizer"})
    max_length: int = field(default=256, metadata={"help": "max length for decoder"})
    # format_level: str = field(default="wordpiece", metadata={"help": "format level, term/wordpiece"})
    special_tokens: str = field(default="", metadata={"help": "Set special token for tokenizer, split by ,"})

#Ouputter

    rl_weight: float = field(default=0.8, metadata={"help": "weight for rl"})

    exp_name: str = field(default="exp",metadata={"help":"name for model saving path"})
    print_every: int = field(default=1000, metadata={"help":"print log every real steps"})
    save_every: int = field(default=1000, metadata={"help":"save model every real steps"})
    batch_mean: bool = field(default=True, metadata={"help": "`need to be developed` normalize loss by batch"})

    warmup_steps: int=field(default=-1)

    rewards: str=field(default='rouge', metadata={"help": "rewards: rouge/bleu/... a sing reward or a list separated by `,` "})

    rouge_type: str=field(default='l', metadata={"help": "1/2/l/12/12l/..."})
    smooth: float=field(default=0)
    seq_decode_model: str=field(default='bart', metadata={"help": "bart/t5/... for pad_token_id/eos_token_id and scoring mode"})
    recover: str=field(default="", metadata={"help": "recovering checkpoint"})
    length_normalize_4_rl: bool=field(default=False, metadata={"help": "length normalization for loss"})
    training_length_penalty: float=field(default=1)
    groundtruth_input: bool=field(default=False, metadata={"help": "use Ground truth for training"})
    cand_num: int=field(default=1)
    kd_inputs_best: bool=field(default=False, metadata={"help": "choose best KD sentence for training"})
    kd_inputs_worst: bool = field(default=False, metadata={"help": "choose worst KD sentence for training"})
    weight_decay: float = field(default=0.0, metadata={"help": "Weight decay for AdamW if we apply some."})
    loss_type: str = field(default='default',
                       metadata={"help": "default is the default average-baseline loss"})
    margin: float = field(default=0.1, metadata={"help": "margin of BRIO loss"})
    do_rl: bool = field(default=True)