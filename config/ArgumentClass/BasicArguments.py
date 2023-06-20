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

    previous_dir: str = field(default="./cache",metadata={"help":"previous model path"})
    output_dir: str = field(default="./temp",metadata={"help":"Output data path"})
    cache_dir: str = field(default="./cache",metadata={"help":"Path to cache auto models"})
    logging_dir: str = field(default="./temp",metadata={"help":"Path to save logging"})
    from_tf: bool = field(default=False,metadata={"help":"load model from tf weight"})
    datareader_streaming: bool = field(default=False,metadata={"help":"whether to load data in streaming way"})
    dataloader_num_workers: int = field(default=3,metadata={"help":"reader workers"})
    outputter_num_workers: int = field(default=None,metadata={"help":"outputter workers, default (None) to dataloader_num_workers"})
    extract_chunk: int = field(default=10,metadata={"help":"param for pool.imap, chunks processes for one time"})
    shuffle_num_batch: int = field(default=50,metadata={"help":"N batch for shuffle buffer"})
    num_train_epochs: int= field(default=1,metadata={"help":"N epochs to train"})
    logging_steps: int= field(default=200,metadata={"help":"default step to logging (print, azureml, tensorboard)"})
    save_steps: int=field(default=5000,metadata={"help":"default step to save ckpt, should be same as eval_steps"})
    save_total_limit: int=field(default=2,metadata={"help":"save total limit"})
    eval_steps: int=field(default=5000,metadata={"help":"evaluation every N steps"})
    evaluation_strategy: str=field(default="steps",metadata={"help":"evaluation strategy: no/steps/epoch"})
    load_best_model_at_end: bool=field(default=True,metadata={"help":"load best model at the end for save"})
    greater_is_better: bool=field(default=True,metadata={"help":"help to judge best model"})
    #save_last: bool=field(default=False,metadata={"help":"only save the last model"})
    #task: str = field(default="classification",metadata={"help":"Model Task"})
    #model: str = field(default="bert-base-uncased",metadata={"help":"Either local or model identifier from huggingface.co/models"})
    fp16: bool=field(default=False, metadata={"help":"whether to enable mix-precision"})    
    remove_unused_columns: bool = field(default=False, metadata={"help": "Remove columns not required by the model when using an nlp.Dataset."})
    disable_tqdm: bool = field(default=True, metadata={"help": "Disable tqdm, print log instead"})
#Train Setting
    trainer: str=field(default="common", metadata={"help":"user-defined trainer to select"})
    eval_metrics: str = field(default="acc",metadata={"help":"Metrics to eval model with delimiter ,"})
    evaluate_during_training: bool = field(default=True,metadata={"help":"Wheter to enable evaluation during training"})
    to_cache: str = field(default="",metadata={"help":"To hotfix fields pop issue in huggingface"})
    output_type: str = field(default="score",metadata={"help":"Set outputter type"})
    eager_predict: bool = field(default=False,metadata={"help":"Eager prediction for debugging"})
    dense_gradients: bool = field(default=False,metadata={"help":"Sync dense gradient for efficiency"})
    compress_gradients: bool = field(default=False,metadata={"help":"Sync gradient in fp16"})
    multi_tasks : str = field(default="DE,FR,EN,ES,SV,IT,NL", metadata={"help":"Predefined multi-tasks"})
    label_names: str = field(default=None, metadata={"help": "label names for label ids"})
#Preprocessor
    label_mapping: str = field(default="1:1,0:0",metadata={"help":"labelname:label,..."})
    tok_max_length: int = field(default=1024,metadata={"help":"max length for tokenizer"})
    max_length: int  = field(default=256,metadata={"help":"max length for decoder"})
    format_level: str = field(default="wordpiece",metadata={"help":"format level, term/wordpiece"})
    special_tokens: str = field(default="",metadata={"help":"Set special token for tokenizer, split by ,"})
    sen_sep: str = field(default=" #Z# ",metadata={"help":"The seperator of sentences in a landing page"})
#Ouputter
    output_mapping: str = field(default="0:0,1:1,2:2,3:3,4:4,5:5,6:6,7:7,8:8,9:9,10:10,11:11,12:12")
    rl_weight: float = field(default=0.8, metadata={"help": "weight for rl"})
    rl_mask_prob: float = field(default=0.5, metadata={"help": "not use"})
    exp_name: str = field(default="exp",metadata={"help":"name for model saving path"})
    pt: bool = field(default=True,metadata={"help":"Save model to local/azure False/True"})
    continue_train: bool=field(default=False, metadata={"help":"Reload checkpoints of the same `exp_name` if exists "})
    print_every: int = field(default=1000, metadata={"help":"print log every real steps"})
    save_every: int = field(default=1000, metadata={"help":"save model every real steps"})
    reward_type: str = field(default='pc', metadata={"help": "`need to be developed` pc/pc2, rouge/rouges "})
    loss_type: str = field(default='pc', metadata={"help": "pc is the default average-baseline loss, max is the greedy-baseline loss"})
    batch_mean: bool = field(default=True, metadata={"help": "`need to be developed` normalize loss by batch"})
    seq_num_mean: bool = field(default=True, metadata={"help": "`need to be developed` normalize loss by seq_num"})
    fix_word_embedding: bool = field(default=False)
    label_smoothing: float = field(default=0.1, metadata={"help": "label_smoothing"})
    num_warmup_steps: int =field(default=1000, metadata={"help": "not use, the option of GAN"})
    warmup_steps: int=field(default=-1)
    post_replace: bool = field(default=False, metadata={"help": "`need to be developed`"})

    distinct_normalize: bool = field(default=False, metadata={"help": "not use"})
    rewards: str=field(default='pc2', metadata={"help": "rewards: rouge/bleu/pc/... a sing reward or a list separated by `,` "})


    clean: bool=field(default=False, metadata={"help": "not use"})
    rouge_type: str=field(default='l', metadata={"help": "1/2/l/12/12l/..."})
    kd_inputs: bool=field(default=False, metadata={"help": "use KD as the input"})
    real_time_gen: bool=field(default=False, metadata={"help": "not use"})
    use_eval: bool=field(default=False, metadata={"help": "not use"})
    compute_rep: bool=field(default=False)
    window_size: int=field(default=300)
    punish_weight: float=field(default=1)
    rep_punish_weight: float=field(default=1)
    default_lambda: str=field(default=0.1)
    freq_learning_rate: float=field(default=0.5)
    rep_learning_rate: float=field(default=0.003)
    rl_weight_lr: float=field(default=0.05)
    max_lambda_cut: float=field(default=1)
    gama: float=field(default=0.999)
    eval_non_seq: bool=field(default=False, metadata={"help": "not use"})
    update_rl_weight: bool=field(default=False)
    all_steps: int=field(default=50000, metadata={"help": "rl_weight scheduler all_steps"})
    st_mask_rate: float=field(default=0,  metadata={"help": "rl_weight scheduler init mask_rate"})
    max_mask_rate: float=field(default=0.5,  metadata={"help": "rl_weight scheduler max mask_rate"})
    update_with_dif: bool=field(default=False)
    lambda_discount: float=field(default=0.9999)
    recover_path: str=field(default="")
    init_dynamic_rl_weight_lambda: float=field(default=0)
    smooth: float=field(default=0)
    seq_decode: bool=field(default=False, metadata={"help": "not use"})
    seq_decode_do_sample: bool=field(default=False, metadata={"help": "sample when decoding"})
    seq_decode_model: str=field(default='bart', metadata={"help": "bart/t5/... for pad_token_id/eos_token_id and scoring mode"})
    recover: str=field(default="", metadata={"help": "recovering checkpoint"})
    naive_seq_baseline: bool=field(default=False, metadata={"help": "option for seq baselines"})
    naive_seq_baseline_mask: bool=field(default=False, metadata={"help": "adding masked ce loss for seq baselines"})
    naive_seq_baseline_mask_weight: float=field(default=1)
    count_normal_ce: bool=field(default=False, metadata={"help": "adding ce loss for seq baselines"})
    mask_spe_tk: bool=field(default=False)
    naive_seq_baseline_sample_num: int=field(default=1, metadata={"help": "sample_num for seq baselines"})
    length_normalize_4_rl: bool=field(default=False, metadata={"help": "length normalization for loss"})
    use_loss: str=field(default='pg', metadata={"help": "different losses: selfcritic/avg/..."})
    training_num_beams: int=field(default=1)
    training_num_beams_groups: int=field(default=1)
    training_early_stopping: bool=field(default=False)
    training_repetition_penalty: float=field(default=1)
    training_top_p: float=field(default=1)
    training_length_penalty: float=field(default=1)
    decode_training_length_penalty: float=field(default=1)
    training_min_length: int=field(default=56)
    training_max_length: int=field(default=142)
    training_no_repeat_ngram_size: int=field(default=3)
    use_seq_gen_res: bool=field(default=True,)
    use_pre_gen_res: bool=field(default=False, metadata={"help": "use pre-processed sentences for seq baselines"})
    not_replace_kd: bool=field(default=False, metadata={"help": "use Ground truth for training"})
    replace_kd_prob: float=field(default=1)
    margin: float=field(default=0.1, metadata={"help": "BRIO"})
    scale: float=field(default=1, metadata={"help": "BRIO"})
    cand_num: int=field(default=1, metadata={"help": "sample_num of BRIO"})
    exclude_eos: bool=field(default=False)
    new_cand_mask_y_s: bool=field(default=False)
    new_cand_mask: bool=field(default=False, metadata={"help": "mask special token"})
    kd_inputs_best: bool=field(default=False, metadata={"help": "choose best KD sentence for training"})
    kd_inputs_worst: bool = field(default=False, metadata={"help": "choose worst KD sentence for training"})
    weight_decay: float = field(default=0.0, metadata={"help": "Weight decay for AdamW if we apply some."})


