import os
from dataclasses import dataclass
import numpy as np
from numpy.lib.function_base import average, median
import torch
from torch.distributed.distributed_c10d import init_process_group
from transformers import Trainer
# from .trainer_test import Trainer
from data.data_reader import CustomizeCollator
from transformers.trainer_utils import speed_metrics
import trainer.Metrics as Metrics
import trainer.Outputter as Outputter
import time
from transformers import logging
from transformers.trainer_callback import PrinterCallback,TrainerCallback, TrainerState
from config.decorator import replace
from torch.utils.data.dataloader import DataLoader
import copy
from torch.utils.data.dataset import Dataset, IterableDataset
from transformers.file_utils import is_datasets_available
from transformers.trainer_pt_utils import IterableDatasetShard, LabelSmoother
from transformers.trainer_utils import TrainOutput
import math
import nltk
from unirl import RankingLoss
import torch.nn.functional as F


from transformers.trainer_utils import (
    PREFIX_CHECKPOINT_DIR,
    BestRun,
    EvalLoopOutput,

    HPSearchBackend,
    HubStrategy,
    IntervalStrategy,
    PredictionOutput,
    ShardedDDPOption,
    TrainerMemoryTracker,
    TrainOutput,
    default_compute_objective,
    default_hp_space,
    denumpify_detensorize,
    get_last_checkpoint,
    number_of_arguments,
    set_seed,
    speed_metrics,
)
from transformers.trainer_pt_utils import (
    DistributedLengthGroupedSampler,
    DistributedSamplerWithLoop,
    DistributedTensorGatherer,
    IterableDatasetShard,
    LabelSmoother,
    LengthGroupedSampler,
    SequentialDistributedSampler,
    ShardSampler,
    distributed_broadcast_scalars,
    distributed_concat,
    find_batch_size,
    get_parameter_names,
    nested_concat,
    nested_detach,
    nested_numpify,
    nested_truncate,
    nested_xla_mesh_reduce,
    reissue_pt_warnings,
)
from data.tokenizer_utils import *
from torch import nn
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
import math 
import torch.distributed as dist
# import matplotlib.pyplot as plt
from unirl import PolicyGradientForUniLM

from transformers import AdamW, get_linear_schedule_with_warmup


from s2s_ft.tokenization_unilm import UnilmTokenizer
from trainer.Trainer import register_trainer
from collections import deque
import collections
from transformers.debug_utils import DebugOption, DebugUnderflowOverflow
from torch.utils.data.distributed import DistributedSampler
from transformers.file_utils import is_torch_tpu_available,  is_sagemaker_mp_enabled

if is_datasets_available():
    import datasets
logger = logging.get_logger(__name__)
logger.setLevel('INFO')

from transformers.integrations import AzureMLCallback

from typing import NamedTuple
class EvalPrediction(NamedTuple):
    """
    Evaluation output (always contains labels), to be used to compute metrics.

    Parameters:
        predictions (:obj:`np.ndarray`): Predictions of the model.
        label_ids (:obj:`np.ndarray`): Targets to be matched.
    """

    predictions: Union[np.ndarray, Tuple[np.ndarray]]
    label_ids: np.ndarray
    q2_ids: Optional[List[str]]
    raw_src: Optional[List[str]]
    raw_src_origin: Optional[List[str]]

def timer(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        res = func(*args, **kwargs)
        # print('共耗时约 {:.2f} 秒'.format(time.time() - start))
        return res

    return wrapper


@register_trainer("rl")
class Trainer(Trainer):
    def __init__(self, model, args, model_args, task_args, train_dataset, eval_dataset, auto_tokenizer, pred_dataset):
        data_collator = CustomizeCollator(train_dataset,eval_dataset,pred_dataset)
        self.pred_dataset = pred_dataset.get_dataset() if pred_dataset else None
        #auto_tokenizer = prepare_tokenizer(model_args._name_or_path, args.cache_dir, special_tokens=args.special_tokens)
        # resize embedding, will do nothing if `old_num_tokens==new_num_tokens`
        model.resize_token_embeddings(len(auto_tokenizer))

        self.args = args
        tokenizer_name = 'bert-base-uncased'
        unused2token = {}
        never_split = unused2token
        tokenizer = UnilmTokenizer.from_pretrained(
            tokenizer_name,
            do_lower_case=True, do_basic_tokenize=True,
            cache_dir=None, never_split=never_split)
        stop_path = "data/stop_words.txt"
        rl_config = self.args.rl_config  # 'local.more3.unirl.ini'
        special_token_ids = []
        print(self.args.reward_type)
        self.rl_agent = PolicyGradientForUniLM.from_config(
            rl_config, tokenizer,  80, 'multiply', unused2token,
            special_token_ids, self.args.reward_type, distinct_normalize=self.args.distinct_normalize,
            rewards=self.args.rewards, rewards_weight=self.args.rewards_weight,
            rouge_type=self.args.rouge_type, sample_num=model.config.sample_num, cand_num=self.args.cand_num,
            local_rank=self.args.local_rank, loss_type=self.args.loss_type, margin=self.args.margin)

        if args.do_train:
            if args.eval_metrics == "eval_loss":
                metrics_calculator = None
                args.metric_for_best_model = "eval_loss"
            else:
                #metrics_calculator = MetricsCalculator(args.eval_metrics, model_args._name_or_path, args, task_args) if args.eval_metrics else None
                model_dict = {}
                if 'fact' in self.args.rewards:
                    model_dict['fact'] = self.rl_agent.fact_model
                metrics_calculator = MetricsCalculator(args.eval_metrics, model_args._name_or_path, args, task_args,
                                                       auto_tokenizer, model_dict) if args.eval_metrics else None

                args.metric_for_best_model = MetricsCalculator.cvt_mtc(args.eval_metrics.split(",")[0], False)
        else:
            metrics_calculator = None
        if args.do_predict:
            print(args.result_header) #query
            self.result_header = args.result_header.split(
                ",") if "," in args.result_header else eval_dataset.feature_extractor.model_header
            print(self.result_header) #query:doc:
            self.outputter = getattr(Outputter, args.output_type)(args, model_args._name_or_path,
                                                                  tokenizer=auto_tokenizer)
            print(args.output_type) #generation
            print(self.outputter)
            #assert 1==0

        if 'azure_ml' in args.report_to:
            args.report_to.remove('azure_ml')

        # adjust labels for both loss and metrics computation 
        # in case there are multiple label components 
        default_label_names = ["labels"]
        args.label_names = args.label_names.split(",") if args.label_names else default_label_names

        if self.args.recover != "":
            model.load_state_dict(torch.load(args.recover))


        if 'brio' in self.args.rewards:
            self.rl_agent.brio_model = copy.deepcopy(model)
            self.rl_agent.brio_model.cuda()
            # self.rl_agent.brio_model.load_state_dict(torch.load('brio_model'))

        super().__init__(
            model = model,
            args = args,
            train_dataset = train_dataset.get_dataset() if train_dataset else None,
            eval_dataset = eval_dataset.get_dataset() if eval_dataset else None,
            data_collator = data_collator,
            compute_metrics = metrics_calculator
            )

        if self.args.smooth > 0:
            self.smoother = LabelSmoother(epsilon=self.args.smooth)
        else:
            self.smoother = None

        self.train_start_time = time.time()
        self.model.tokenizer = auto_tokenizer
        az_logger = AmlLogger()
        if az_logger.active:
            self.add_callback(AmlLogger)


        self.lambda_for_vocab = None
        self.lambda_for_vocab_lr = 1e-7
        self.log_probs_constraint = 0



        self.tokens_buf = {'gen': deque(), 'base_model': deque(), 'base': deque()}
        self.tokens_dict = {'gen': {}, 'base_model': {}, 'base': {}}
        self.his_reward_dict = {}
        self.his_all_reward = []
        self.his_max_reward = []
        self.his_min_reward = []
        self.his_mean_reward = []
        self.his_greedy_reward = []
        self.his_base_reward = []
        self.his_time = []
        self.his_condition_reward = []
        self.his_win_num = []
        self.cur_rewards_lambda = {}
        self.min_rewards_weight_sep = {}
        self.his_reward_dict['gen'] = {}
        self.his_reward_dict['base_model'] = {}
        for i, name in enumerate(self.args.rewards.split(',')):
            self.his_reward_dict['gen'][name] = []
            self.his_reward_dict['base_model'][name] = []
            self.cur_rewards_lambda[i] = 0.001
        for i, w in enumerate(self.args.min_rewards_weight_sep.split(',')):
            self.min_rewards_weight_sep[i] = float(w)

        self.his_dif_cont_rate = []
        self.his_recall = []
        self.tokens_cnt = 0
        self.line_cnt = 0
        if self.args.freq_th_type == 'ratio':
            assert self.args.th < 1
        else:
            assert self.args.th > 1

        self.tk2lambda = {}
        self.beat_base_his = []
        self.rep_tk2lambda = {}
        self.window_size = self.args.window_size
        #self.suf = self.args.sufs.split(',')
        self.suf_id = 0
        self.dynamic_rl_weight_lambda = self.args.init_dynamic_rl_weight_lambda
        self.dynamic_rl_weight = 0

    def get_str(self, y, tokenizer, process=True):
        np.random.seed(self.state.global_step)
        if type(y).__name__ != 'list':
            pre_output_ids = y.tolist()
        else:
            pre_output_ids = y
        output_ids = []
        for i in range(len(pre_output_ids)):
            output_id = []
            for j in range(0, len(pre_output_ids[i])):
                if pre_output_ids[i][j] == -100:
                    break
                output_id.append(pre_output_ids[i][j])
            output_ids.append(output_id)
        traces = [tokenizer.decode(output_ids[i], skip_special_tokens=False, clean_up_tokenization_spaces=False)
                  for i in range(len(output_ids))]
        if not process:
            return traces
        def process(x):
            x = x.replace('</s>', ' ')
            x = x.replace('<s>', ' ')
            x = x.replace('<pad>', ' ')
            return '\n'.join(nltk.sent_tokenize(x))
            # return ' '.join(x.strip().split())
        if process:
            traces = [process(e) for e in traces]
        return traces

    def get_tks(self, y, tokenizer, process=True):
        # print(y.shape)
        np.random.seed(self.state.global_step)
        if type(y).__name__ != 'list':
            pre_output_ids = y.tolist()
        else:
            pre_output_ids = y
        output_ids = []
        for i in range(len(pre_output_ids)):
            output_id = []
            for j in range(0, len(pre_output_ids[i])):
                if pre_output_ids[i][j] == -100:
                    break
                output_id.append(pre_output_ids[i][j])
            output_ids.append(output_id)

        tks = [
            tokenizer.convert_ids_to_tokens(output_ids[i]) for i in
            range(len(output_ids))]
        tks = [(output_ids[i][_id], tks[i][_id]) for i in range(len(output_ids)) for _id in range(len(output_ids[i]))]
        last_p = None
        for p in tks:
            if "▁," in p[1]:
                print(p)
            if "▁." in p[1]:
                print(p)
            last_p = p


        return tks
    def get_ads_str(self, y, tokenizer, process=True):
        #print(y.shape)
        np.random.seed(self.state.global_step)
        if type(y).__name__ != 'list':
            pre_output_ids = y.tolist()
        else:
            pre_output_ids = y
        output_ids = []
        for i in range(len(pre_output_ids)):
            output_id = []
            for j in range(0, len(pre_output_ids[i])):
                if pre_output_ids[i][j] == -100:
                    break
                output_id.append(pre_output_ids[i][j])
            output_ids.append(output_id)
        traces = [tokenizer.decode(output_ids[i], skip_special_tokens=False, clean_up_tokenization_spaces=False)
                  for i in range(len(output_ids))]
        if not process:
            return traces
        def process(x):
            x = x.replace('</s>', ' ')
            x = x.replace('<s>', ' ')
            x = x.replace('<pad>', ' ')
            return x
            # return ' '.join(x.strip().split())
        if process:
            traces = [process(e) for e in traces]
        return traces

    def rl_ids_2_str(self, y_b, y_s, max_ids, masked_ids, input_ids,
                     labels, non_zero_sum_tensor, log_probs, querys,
                         tokenizer, y_zero_b, y_zero_s, y_zero_labels,
                     base_y_b=None,truth_log_probs=None, pre_gen_scores=None, predict_baseline=None,
                     not_normal_log_probs=None, raw_src=None, inputs_brio=None, refs=None, base_traces=None):

        np.random.seed(self.state.global_step)
        target_mask = ~labels.data.eq(-100)

        def get_str(y, ori_line_split=None, split_tk=None):
            pre_output_ids = y.tolist()

            output_ids = []
            for i in range(len(pre_output_ids)):
                output_id = []
                if split_tk is not None:
                    ori_line_split_ = ori_line_split[i][1:]
                    print(ori_line_split_)
                    print(pre_output_ids[i])
                    print('-----')
                tot = 0
                for j in range(len(pre_output_ids[i])):#range(0, min(len(pre_output_ids[i]), cur_non_zero_sum[i])):
                    if pre_output_ids[i][j] == -100:
                        break
                    if split_tk is not None:
                        if tot < len(ori_line_split_) and ori_line_split_[tot] == split_tk:
                            output_id.append(split_tk)
                            tot += 1

                    output_id.append(pre_output_ids[i][j])
                    tot += 1
                output_ids.append(output_id)
            #print(output_ids)
            traces = [tokenizer.decode(output_ids[i], skip_special_tokens=False, clean_up_tokenization_spaces=False)
                        for i in range(len(output_ids))]
            return traces

        def get_str_base(y):
            pre_output_ids = y.tolist()
            output_ids = []
            for i in range(len(pre_output_ids)):
                output_id = []
                for j in range(0, len(pre_output_ids[i])):
                    if pre_output_ids[i][j] == -100:
                        break
                    output_id.append(pre_output_ids[i][j])
                output_ids.append(output_id)
            #print(output_ids)
            traces = [tokenizer.decode(output_ids[i], skip_special_tokens=False, clean_up_tokenization_spaces=False)
                        for i in range(len(output_ids))]
            return traces
        def get_label_str(y):
            pre_output_ids = y.tolist()
            output_ids = []
            for i in range(len(pre_output_ids)):
                output_id = []
                for j in range(0, len(pre_output_ids[i])):
                    if pre_output_ids[i][j] == -100:
                        break
                    output_id.append(pre_output_ids[i][j])
                output_ids.append(output_id)
            #print(output_ids)
            traces = [tokenizer.decode(output_ids[i], skip_special_tokens=False, clean_up_tokenization_spaces=False)
                        for i in range(len(output_ids))]
            return traces
        def get_tks(y):
            pre_output_ids = y.tolist()
            output_ids = []

            for i in range(len(pre_output_ids)):
                output_id = []
                for j in range(len(pre_output_ids[i])):#range(0, min(len(pre_output_ids[i]), cur_non_zero_sum[i])):
                    if pre_output_ids[i][j] == -100:
                        break
                    output_id.append(pre_output_ids[i][j])
                output_ids.append(output_id)
            tks = [
                tokenizer.convert_ids_to_tokens(output_ids[i]) for i in
                range(len(output_ids))]
            return tks
        base_traces = get_str_base(base_traces)
        b_traces = get_str(y_b)
        traces = get_str(y_s)
        if base_y_b is not None:
            base_b_traces = get_str(base_y_b)
        # 我可以打印每个token接受的平均的reward，eos接受的
        b_zeros_tks = get_tks(y_zero_b)
        b_zero_traces = get_str(y_zero_b)
        # print(b_zero_traces)
        b_zero_traces = [t.replace('<pad>', ' ') for t in b_zero_traces]
        b_zero_traces = [' '.join(t.split()) for t in b_zero_traces]
        #print(b_zero_traces)
        zeros_tks = get_tks(y_zero_s)
        # print(zeros_tks)
        zero_traces = get_str(y_zero_s)
        zero_traces = [t.replace('<pad>', ' ') for t in zero_traces]
        zero_traces = [' '.join(t.split()) for t in zero_traces]
        labels_zero_tks = get_tks(y_zero_labels)
        def get_clean_label(clean=False):
            if not clean:
                labels_traces = get_str(y_zero_labels)
                return labels_traces
            labels_traces_tks = get_tks(y_zero_labels)
            labels_traces_truth =get_tks(labels)
            for i, t in enumerate(labels_traces_tks):
                assert len(labels_traces_tks[i]) == len(labels_traces_truth[i])
                expo_set = set()
                for j, tk in enumerate(labels_traces_tks[i]):
                    if tk == '<pad>' and tk != labels_traces_truth[i][j]:
                        expo_set.add(labels_traces_truth[i][j])
                for j, tk in enumerate(labels_traces_tks[i]):
                    if tk in expo_set:
                        labels_traces_tks[i][j] = '<pad>'
            labels_traces = [tokenizer.convert_tokens_to_string(traces_pos) for traces_pos in labels_traces_tks]
            return labels_traces
        labels_traces = get_clean_label(self.args.clean)
        labels_traces = [t.replace('<pad>', ' ') for t in labels_traces]
        labels_traces = [' '.join(t.split()) for t in labels_traces]

        max_ids_tk = max_ids[:, :-1].tolist()
        all_gen_b_traces_tk = [
            tokenizer.convert_ids_to_tokens(max_ids_tk[i]) for i in
            range(len(max_ids_tk))]

        masked_ids_tk = masked_ids[:, :-1].tolist()
        all_gen_traces_tk = [
            tokenizer.convert_ids_to_tokens(masked_ids_tk[i]) for i in
            range(len(masked_ids_tk))]
        all_gen_b_traces_tk = [
            tokenizer.convert_ids_to_tokens(y_b[i]) for i in
            range(len(y_b))]
        all_gen_traces_tk = [
            tokenizer.convert_ids_to_tokens(y_s[i]) for i in
            range(len(y_s))]

        raw_tgt = get_label_str(labels)

        raw_tgt_tk = get_tks(labels)

        rl_loss, b_reward_dict, all_reward, other_reward = self.rl_agent(querys, raw_src, raw_tgt, traces, b_traces,
                                                          log_probs, raw_tgt_tk, all_gen_traces_tk, all_gen_b_traces_tk,
                                                          tokenizer, zero_traces, b_zero_traces, labels_traces, truth_log_probs=truth_log_probs, pre_gen_scores=pre_gen_scores, predict_baseline=predict_baseline,
                                                               steps=self.state.global_step, not_normal_log_probs=not_normal_log_probs, input_ids=input_ids,
                                                               y_s=y_s, y_b=y_b, inputs_brio=inputs_brio, refs=refs, base_traces=base_traces)
        return rl_loss, b_reward_dict, b_zeros_tks, zeros_tks, labels_zero_tks, all_gen_traces_tk, all_gen_b_traces_tk, all_reward, other_reward



    def train(
        self,
        resume_from_checkpoint: Optional[Union[str, bool]] = None,
        trial: Union["optuna.Trial", Dict[str, Any]] = None,
        ignore_keys_for_eval: Optional[List[str]] = None,
        **kwargs,
    ):
        """
        Main training entry point.

        Args:
            resume_from_checkpoint (:obj:`str` or :obj:`bool`, `optional`):
                If a :obj:`str`, local path to a saved checkpoint as saved by a previous instance of
                :class:`~transformers.Trainer`. If a :obj:`bool` and equals `True`, load the last checkpoint in
                `args.output_dir` as saved by a previous instance of :class:`~transformers.Trainer`. If present,
                training will resume from the model/optimizer/scheduler states loaded here.
            trial (:obj:`optuna.Trial` or :obj:`Dict[str, Any]`, `optional`):
                The trial run or the hyperparameter dictionary for hyperparameter search.
            ignore_keys_for_eval (:obj:`List[str]`, `optional`)
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions for evaluation during the training.
            kwargs:
                Additional keyword arguments used to hide deprecated arguments
        """

        # from transformers import BartTokenizer, BartForConditionalGeneration
        # self.model.tmp_model = BartForConditionalGeneration.from_pretrained('lmqg/bart-large-squad')
        # x = [[0, 50265, 12674, 1755, 1437, 50265, 617, 4939, 69, 3501, 756, 6, 8996, 25, 15629, 3250, 381, 16597, 957, 11, 5, 2266, 4388, 4003, 18137, 6, 23906, 10023, 4, 2]]
        # q = self.model.tmp_model.generate(torch.tensor(x))
        # print(q)
        # assert 1==0

        resume_from_checkpoint = None if not resume_from_checkpoint else resume_from_checkpoint

        # memory metrics - must set up as early as possible
        self._memory_tracker.start()

        args = self.args

        self.is_in_train = True

        # do_train is not a reliable argument, as it might not be set and .train() still called, so
        # the following is a workaround:
        if args.fp16_full_eval and not args.do_train:
            self._move_model_to_device(self.model, args.device)

        if "model_path" in kwargs:
            resume_from_checkpoint = kwargs.pop("model_path")
            warnings.warn(
                "`model_path` is deprecated and will be removed in a future version. Use `resume_from_checkpoint` "
                "instead.",
                FutureWarning,
            )
        if len(kwargs) > 0:
            raise TypeError(f"train() received got unexpected keyword arguments: {', '.join(list(kwargs.keys()))}.")
        # This might change the seed so needs to run first.
        self._hp_search_setup(trial)

        # Model re-init
        model_reloaded = False
        if self.model_init is not None:
            # Seed must be set before instantiating the model when using model_init.
            set_seed(args.seed)
            self.model = self.call_model_init(trial)
            model_reloaded = True
            # Reinitializes optimizer and scheduler
            self.optimizer, self.lr_scheduler = None, None
        # Load potential model checkpoint
        if isinstance(resume_from_checkpoint, bool) and resume_from_checkpoint:
            resume_from_checkpoint = get_last_checkpoint(args.output_dir)
            if resume_from_checkpoint is None:
                raise ValueError(f"No valid checkpoint found in output directory ({args.output_dir})")

        if resume_from_checkpoint is not None:
            if not os.path.isfile(os.path.join(resume_from_checkpoint, WEIGHTS_NAME)):
                raise ValueError(f"Can't find a valid checkpoint at {resume_from_checkpoint}")

            logger.info(f"Loading model from {resume_from_checkpoint}).")

            if os.path.isfile(os.path.join(resume_from_checkpoint, CONFIG_NAME)):
                config = PretrainedConfig.from_json_file(os.path.join(resume_from_checkpoint, CONFIG_NAME))
                checkpoint_version = config.transformers_version
                if checkpoint_version is not None and checkpoint_version != __version__:
                    logger.warn(
                        f"You are resuming training from a checkpoint trained with {checkpoint_version} of "
                        f"Transformers but your current version is {__version__}. This is not recommended and could "
                        "yield to errors or unwanted behaviors."
                    )

            if args.deepspeed:
                # will be resumed in deepspeed_init
                pass
            else:
                # We load the model state dict on the CPU to avoid an OOM error.
                state_dict = torch.load(os.path.join(resume_from_checkpoint, WEIGHTS_NAME), map_location="cpu")
                # If the model is on the GPU, it still works!
                self._load_state_dict_in_model(state_dict)

                # release memory
                del state_dict

        # If model was re-initialized, put it on the right device and update self.model_wrapped
        if model_reloaded:
            if self.place_model_on_device:
                self._move_model_to_device(self.model, args.device)
            self.model_wrapped = self.model

        # Keeping track whether we can can len() on the dataset or not
        train_dataset_is_sized = isinstance(self.train_dataset, collections.abc.Sized)

        # Data loader and number of training steps
        print('Data loader and number of training steps')
        train_dataloader = self.get_train_dataloader()

        print('Setting up training control variables:')
        # Setting up training control variables:
        # number of training epochs: num_train_epochs
        # number of training steps per epoch: num_update_steps_per_epoch
        # total number of training steps to execute: max_steps
        total_train_batch_size = args.train_batch_size * args.gradient_accumulation_steps * args.world_size
        if train_dataset_is_sized:
            num_update_steps_per_epoch = len(train_dataloader) // args.gradient_accumulation_steps
            num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
            if args.max_steps > 0:
                max_steps = args.max_steps
                num_train_epochs = args.max_steps // num_update_steps_per_epoch + int(
                    args.max_steps % num_update_steps_per_epoch > 0
                )
                # May be slightly incorrect if the last batch in the training datalaoder has a smaller size but it's
                # the best we can do.
                num_train_samples = args.max_steps * total_train_batch_size
            else:
                max_steps = math.ceil(args.num_train_epochs * num_update_steps_per_epoch)
                num_train_epochs = math.ceil(args.num_train_epochs)
                num_train_samples = len(self.train_dataset) * args.num_train_epochs
        else:
            # see __init__. max_steps is set when the dataset has no __len__
            max_steps = args.max_steps
            # Setting a very large number of epochs so we go as many times as necessary over the iterator.
            num_train_epochs = sys.maxsize
            num_update_steps_per_epoch = max_steps
            num_train_samples = args.max_steps * total_train_batch_size

        if DebugOption.UNDERFLOW_OVERFLOW in self.args.debug:
            if self.args.n_gpu > 1:
                # nn.DataParallel(model) replicates the model, creating new variables and module
                # references registered here no longer work on other gpus, breaking the module
                raise ValueError(
                    "Currently --debug underflow_overflow is not supported under DP. Please use DDP (torch.distributed.launch)."
                )
            else:
                debug_overflow = DebugUnderflowOverflow(self.model)  # noqa

        delay_optimizer_creation = self.sharded_ddp is not None and self.sharded_ddp != ShardedDDPOption.SIMPLE
        if args.deepspeed:
            deepspeed_engine, optimizer, lr_scheduler = deepspeed_init(
                self, num_training_steps=max_steps, resume_from_checkpoint=resume_from_checkpoint
            )
            self.model = deepspeed_engine.module
            self.model_wrapped = deepspeed_engine
            self.deepspeed = deepspeed_engine
            self.optimizer = optimizer
            self.lr_scheduler = lr_scheduler
        elif not delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        self.state = TrainerState()
        self.state.is_hyper_param_search = trial is not None

        # Activate gradient checkpointing if needed
        print('Activate gradient checkpointing if needed')
        if args.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        model = self._wrap_model(self.model_wrapped)

        #model.resize_token_embeddings(model.config.vocab_size + 1)
        print('for the rest of this function `model` is the outside model, whether it was wrapped or not')
        # for the rest of this function `model` is the outside model, whether it was wrapped or not
        if model is not self.model:
            self.model_wrapped = model

        if delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        # Check if saved optimizer or scheduler states exist
        self._load_optimizer_and_scheduler(resume_from_checkpoint)

        # important: at this point:
        # self.model         is the Transformers Model
        # self.model_wrapped is DDP(Transformers Model), Deepspeed(Transformers Model), etc.

        # Train!
        num_examples = (
            self.num_examples(train_dataloader) if train_dataset_is_sized else total_train_batch_size * args.max_steps
        )
        print('Train')
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {num_examples}")
        logger.info(f"  Num Epochs = {num_train_epochs}")
        logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {max_steps}")

        self.state.epoch = 0
        start_time = time.time()
        epochs_trained = 0
        steps_trained_in_current_epoch = 0
        steps_trained_progress_bar = None

        # Check if continuing training from a checkpoint
        if resume_from_checkpoint is not None and os.path.isfile(
            os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME)
        ):
            self.state = TrainerState.load_from_json(os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME))
            epochs_trained = self.state.global_step // num_update_steps_per_epoch
            if not args.ignore_data_skip:
                steps_trained_in_current_epoch = self.state.global_step % (num_update_steps_per_epoch)
                steps_trained_in_current_epoch *= args.gradient_accumulation_steps
            else:
                steps_trained_in_current_epoch = 0

            logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            logger.info(f"  Continuing training from epoch {epochs_trained}")
            logger.info(f"  Continuing training from global step {self.state.global_step}")
            if not args.ignore_data_skip:
                logger.info(
                    f"  Will skip the first {epochs_trained} epochs then the first {steps_trained_in_current_epoch} "
                    "batches in the first epoch. If this takes a lot of time, you can add the `--ignore_data_skip` "
                    "flag to your launch command, but you will resume the training on data already seen by your model."
                )
                if self.is_local_process_zero() and not args.disable_tqdm:
                    steps_trained_progress_bar = tqdm(total=steps_trained_in_current_epoch)
                    steps_trained_progress_bar.set_description("Skipping the first batches")

        # Update the references
        self.callback_handler.model = self.model
        self.callback_handler.optimizer = self.optimizer


        self.callback_handler.lr_scheduler = self.lr_scheduler
        self.callback_handler.train_dataloader = train_dataloader
        self.state.trial_name = self.hp_name(trial) if self.hp_name is not None else None
        if trial is not None:
            assignments = trial.assignments if self.hp_search_backend == HPSearchBackend.SIGOPT else trial
            self.state.trial_params = hp_params(assignments)
        else:
            self.state.trial_params = None
        # This should be the same if the state has been saved but in case the training arguments changed, it's safer
        # to set this after the load.
        self.state.max_steps = max_steps
        self.state.num_train_epochs = num_train_epochs
        self.state.is_local_process_zero = self.is_local_process_zero()
        self.state.is_world_process_zero = self.is_world_process_zero()

        # tr_loss is a tensor to avoid synchronization of TPUs through .item()
        tr_loss = torch.tensor(0.0).to(args.device)
        # _total_loss_scalar is updated everytime .item() has to be called on tr_loss and stores the sum of all losses
        self._total_loss_scalar = 0.0
        self._globalstep_last_logged = self.state.global_step
        model.zero_grad()

        self.control = self.callback_handler.on_train_begin(args, self.state, self.control)

        # Skip the first epochs_trained epochs to get the random state of the dataloader at the right point.
        if not args.ignore_data_skip:
            for epoch in range(epochs_trained):
                # We just need to begin an iteration to create the randomization of the sampler.
                for _ in train_dataloader:
                    break
        all_b_rewards = []
        all_ce_loss = []
        all_file = []
        for epoch in range(epochs_trained, num_train_epochs):
            if isinstance(train_dataloader, DataLoader) and isinstance(train_dataloader.sampler, DistributedSampler):
                train_dataloader.sampler.set_epoch(epoch)
            elif isinstance(train_dataloader.dataset, IterableDatasetShard):
                train_dataloader.dataset.set_epoch(epoch)

            if is_torch_tpu_available():
                parallel_loader = pl.ParallelLoader(train_dataloader, [args.device]).per_device_loader(args.device)
                epoch_iterator = parallel_loader
            else:
                epoch_iterator = train_dataloader

            # Reset the past mems state at the beginning of each epoch if necessary.
            if args.past_index >= 0:
                self._past = None

            steps_in_epoch = (
                len(epoch_iterator) if train_dataset_is_sized else args.max_steps * args.gradient_accumulation_steps
            )
            self.control = self.callback_handler.on_epoch_begin(args, self.state, self.control)

            his_b_rewards = {name: [] for name in self.args.rewards.split(',')}
            epoch_b_rewards = {name: [] for name in self.args.rewards.split(',')}

            self.bi_training_steps_count = 0
            his_ce_loss = []
            his_loss = []
            self.distinct_his = []
            self.his_rl_loss = []
            self.his_value_loss = []
            self.his_value_acc = []
            self.his_kd_ce_loss = []
            self.his_kl_loss = []
            self.his_probs = []
            self.his_2_gram_loss = []
            self.his_3_gram_loss = []
            self.his_4_gram_loss = []
            self.his_2_gram_acc = []
            self.his_eos_probs = []
            self.his_says_probs = []
            self.his_comma_probs = []

            self.his_eos_probs_kd = []
            self.his_says_probs_kd = []
            self.his_comma_probs_kd = []
            self.his_3_gram_acc = []
            self.his_4_gram_acc = []
            self.his_norm = []
            self.his_kd_ce_eos_loss = []
            self.reward_dist_dict = {}
            his_d_rl_weight = []
            his_d_rl_weight_l = []
            epoch_ce_loss = []
            for step, inputs in enumerate(epoch_iterator):
                rep_num = 1
                ori_inputs = copy.deepcopy(inputs)
                for rep_idx in range(rep_num):
                    if steps_trained_in_current_epoch > 0:
                        steps_trained_in_current_epoch -= 1
                        if steps_trained_progress_bar is not None:
                            steps_trained_progress_bar.update(1)
                        if steps_trained_in_current_epoch == 0:
                            self._load_rng_state(resume_from_checkpoint)
                        continue
                    elif steps_trained_progress_bar is not None:
                        steps_trained_progress_bar.close()
                        steps_trained_progress_bar = None

                    if step % args.gradient_accumulation_steps == 0:
                        self.control = self.callback_handler.on_step_begin(args, self.state, self.control)

                    if (
                        ((step + 1) % args.gradient_accumulation_steps != 0)
                        and args.local_rank != -1
                        and args._no_sync_in_gradient_accumulation
                    ):
                        # Avoid unnecessary DDP synchronization since there will be no backward pass on this example.
                        with model.no_sync():
                            tr_loss_step, ce_loss, b_reward = self.training_step(model, inputs)
                    else:

                        tr_loss_step, ce_loss, b_reward = self.training_step(model, inputs)

                        for name, value in b_reward.items():
                            if name == 'ori_loss':
                                value = value.view(-1).clone().detach().cpu().numpy()
                            if name not in his_b_rewards:
                                his_b_rewards[name] = []
                            his_b_rewards[name].append(value)
                            if name not in epoch_b_rewards:
                                epoch_b_rewards[name] = []
                            epoch_b_rewards[name].append(value)

                        his_ce_loss.append(ce_loss.clone().detach().cpu().numpy())
                        his_loss.append(tr_loss_step.clone().detach().cpu().numpy())
                        epoch_ce_loss.append(ce_loss.clone().detach().cpu().numpy())
                        his_d_rl_weight.append(self.dynamic_rl_weight)
                        his_d_rl_weight_l.append(self.dynamic_rl_weight_lambda)
                    if (
                        args.logging_nan_inf_filter
                        and not is_torch_tpu_available()
                        and (torch.isnan(tr_loss_step) or torch.isinf(tr_loss_step))
                    ):
                        # if loss is nan or inf simply add the average of previous logged losses
                        tr_loss += tr_loss / (1 + self.state.global_step - self._globalstep_last_logged)
                    else:
                        tr_loss += tr_loss_step

                    self.current_flos += float(self.floating_point_ops(inputs))

                    # Optimizer step for deepspeed must be called on every step regardless of the value of gradient_accumulation_steps
                    if self.deepspeed:
                        self.deepspeed.step()
                    print_every = args.print_every
                    save_every = args.save_every
                    if args.pt:
                        output_dir = '/mnt/shared_data/zr_output/'
                    else:
                        output_dir = 'zr_output/'

                    if (step + 1) % save_every == 0:
                        if not os.path.exists(output_dir + args.exp_name):
                            os.makedirs(output_dir + args.exp_name)
                        save_path = output_dir + args.exp_name + '/_t5_model.{}_{}'.format(self.state.global_step, step)
                        print('save to ' + save_path, 'epoch ', self.state.epoch)
                        if isinstance(model, torch.nn.DataParallel) or isinstance(model,
                                                                                  torch.nn.parallel.DistributedDataParallel):
                            torch.save(model.module.state_dict(), save_path)
                        else:
                            torch.save(model.state_dict(), save_path)

                    if (step + 1) % print_every == 0:
                        for name in his_b_rewards.keys():
                            his_b_rewards[name] = his_b_rewards[name][-print_every:]
                       # his_b_rewards = his_b_rewards[-print_every:]
                        his_ce_loss = his_ce_loss[-print_every:]
                        self.his_rl_loss = self.his_rl_loss[-print_every:]
                        self.distinct_his = self.distinct_his[-print_every:]
                        self.his_value_loss = self.his_value_loss[-print_every:]
                        self.his_value_acc = self.his_value_acc[-print_every:]
                        self.his_kd_ce_loss = self.his_kd_ce_loss[-print_every:]
                        self.his_kl_loss = self.his_kl_loss[-print_every:]
                        his_loss = his_loss[-print_every:]
                        his_d_rl_weight = his_d_rl_weight[-print_every:]
                        his_d_rl_weight_l = his_d_rl_weight_l[-print_every:]
                        for name, v in his_b_rewards.items():
                            print('At step {}, his_b_rewards {} = {}'.format(step, name, np.mean(v)))
                        print('At step {}, his_ce_loss {}'.format(step, np.mean(his_ce_loss)))
                        print('At step {}, his_rl_loss {}'.format(step, np.mean(self.his_rl_loss)))
                        print('At step {}, his_kl_loss {}'.format(step, np.mean(self.his_kl_loss)))
                        print('At step {}, his_kd_ce_loss {}'.format(step, np.mean(self.his_kd_ce_loss)))
                        print('At step {}, his_loss {}'.format(step, np.mean(his_loss)))
                        print('At step {}, distinct_his {}'.format(step, np.mean(self.distinct_his)))
                        if self.args.use_dynamic_rl_weight:
                            print('At step {}, rl_weight {}'.format(step, np.mean(his_d_rl_weight)))
                            print('At step {}, rl_weight_l {}'.format(step, np.mean(his_d_rl_weight_l)))
                        print('At step {}, his_value_loss {}'.format(step, np.mean(self.his_value_loss)))
                        print('At step {}, his_value_acc {}'.format(step, np.mean(self.his_value_acc)))
                        print('At step {}, his_max_reward {}'.format(step, np.mean(self.his_max_reward)))
                        print('At step {}, his_min_reward {}'.format(step, np.mean(self.his_min_reward)))
                        print('At step {}, his_mean_reward {}'.format(step, np.mean(self.his_mean_reward)))
                        print('At step {}, his_greedy_reward {}'.format(step, np.mean(self.his_greedy_reward)))
                        print('At step {}, his_base_reward {}'.format(step, np.mean(self.his_base_reward)))
                        print('At step {}, his_win_num {}'.format(step, np.mean(self.his_win_num)))
                        # assert 1==0



                        to_print_base_tokens_dict = sorted(self.tokens_dict['base_model'].items(), key=lambda item:-item[1])

                        to_print_tokens_dict = sorted(self.tokens_dict['gen'].items(), key=lambda item: -item[1])
                        all_file.append(to_print_base_tokens_dict[:10])
                        all_file.append(to_print_tokens_dict[:10])


                        print('mask_rate:', model.mask_rate)
                        print('his_dif_cont_rate', np.mean(self.his_dif_cont_rate))
                        self.his_dif_cont_rate = self.his_dif_cont_rate[-print_every:]
                        to_print_dif = sorted(self.rep_tk2lambda.items(), key=lambda item: -item[1])
                        # print('to_print_dif', [(t[0], t[1] / self.tokens_cnt) for t in to_print_dif[:10]])
                        print('rep', [(t[0], t[1]) for t in to_print_dif[:20]])
                        cur_rewards_weight = self.rl_agent.get_rewards_weight()
                        for i, name in enumerate(self.args.rewards.split(',')):
                            print(name, self.cur_rewards_lambda[i], cur_rewards_weight[i])
                            if name not in self.his_reward_dict['gen']:
                                self.his_reward_dict['gen'][name] = []
                            mean_his_reward = np.mean(self.his_reward_dict['gen'][name])
                            if name not in self.his_reward_dict['base_model']:
                                self.his_reward_dict['base_model'][name] = []
                            mean_his_reward_base = np.mean(self.his_reward_dict['base_model'][name])
                            print('name: {}, base: {} cur: {} dif abs: {} dif rel: {}'.format(name, mean_his_reward_base, mean_his_reward, mean_his_reward_base-mean_his_reward, (mean_his_reward_base-mean_his_reward) / mean_his_reward_base))
                        self.beat_base_his = self.beat_base_his[-print_every:]
                        print('beat_base', np.mean(self.beat_base_his))
                        self.his_recall = self.his_recall[-print_every:]
                        print('his_nar_recall', np.mean(self.his_recall))
                        self.his_probs = self.his_probs[-print_every:]
                        print('his_probs', np.mean(self.his_probs))
                        self.his_2_gram_acc = self.his_2_gram_acc[-print_every:]
                        self.his_3_gram_acc = self.his_3_gram_acc[-print_every:]
                        self.his_4_gram_acc = self.his_4_gram_acc[-print_every:]
                        self.his_2_gram_loss = self.his_2_gram_loss[-print_every:]
                        self.his_3_gram_loss = self.his_3_gram_loss[-print_every:]
                        self.his_4_gram_loss = self.his_4_gram_loss[-print_every:]
                        self.his_eos_probs = self.his_eos_probs[-print_every:]
                        self.his_comma_probs = self.his_comma_probs[-print_every:]
                        self.his_says_probs = self.his_says_probs[-print_every:]

                        self.his_eos_probs_kd = self.his_eos_probs_kd[-print_every:]
                        self.his_comma_probs_kd = self.his_comma_probs_kd[-print_every:]
                        self.his_says_probs_kd = self.his_says_probs_kd[-print_every:]
                        self.his_norm = self.his_norm[-print_every:]
                        self.his_kd_ce_eos_loss = self.his_kd_ce_eos_loss[-print_every:]
                        to_print_count = sorted(self.reward_dist_dict.items(), key=lambda item: -len(item[1]))
                        to_print_count = to_print_count[:100]
                        to_print_count = sorted(to_print_count, key=lambda item: -np.mean(item[1]))
                        to_print_dist = sorted(self.reward_dist_dict.items(), key=lambda item: -np.mean(item[1]))
                        if self.args.seq_decode_model == 'bart':
                            to_check = [4, 479, 2156, 6, 1, 2,1437, 22, 60, 128, 12, 480, 111, 22209, 49519, 12905, 0, 50141]
                        else:
                            to_check = []
                        reward_dist_dict = [(tk, np.mean(self.reward_dist_dict[tk])) if tk in self.reward_dist_dict else (tk, 0) for tk in to_check]

                        try:
                            print('to_print_dist',
                                  [(self.tmp_tokenizer.convert_ids_to_tokens([t[0]]), np.mean(t[1])) for t in
                                   to_print_dist[:20] if t[0] != -1 ] )
                        except (OverflowError , UnicodeEncodeError) as e:
                            print([t[0] for t in
                                   to_print_dist[:20]])
                            print(e)
                        try:

                            print('to_print_dist',
                                  [(self.tmp_tokenizer.convert_ids_to_tokens([t[0]]), np.mean(t[1])) for t in
                                   to_print_dist[-20:] if t[0] != -1 ])
                        except (OverflowError , UnicodeEncodeError) as e:

                            print(e)
                        try:

                            to_print_dist = sorted(reward_dist_dict, key=lambda item: -item[1])
                            print('to_check_dist',
                                  [(self.tmp_tokenizer.convert_ids_to_tokens([t[0]]), np.mean(t[1])) for t in
                                   to_print_dist[:] if t[0] != -1 ])

                        except (OverflowError , UnicodeEncodeError) as e:
                            # print(to_print_count[-20:])
                            print(e)
                        try:
                            print('to_print_count', [(self.tmp_tokenizer.convert_ids_to_tokens([t[0]]), np.mean(t[1])) for t in
                                                    to_print_count[:20] if t[0] != -1 ])

                        except (OverflowError , UnicodeEncodeError) as e:
                            print('to_print_count',
                                  [t[0]for t in
                                   to_print_count[:20]])
                            print(e)
                        try:
                            print('to_print_count', [(self.tmp_tokenizer.convert_ids_to_tokens([t[0]]), np.mean(t[1])) for t in
                                                     to_print_count[-20:] if t[0] != -1 ])
                        except (OverflowError , UnicodeEncodeError) as e:
                            print(e)
                        print('eos_probs: {}'.format(np.mean(self.his_eos_probs)))
                        print('says_probs: {}'.format(np.mean(self.his_says_probs)))
                        print('comma_probs: {}'.format(np.mean(self.his_comma_probs)))
                        print('eos_probs_kd: {}'.format(np.mean(self.his_eos_probs_kd)))
                        print('says_probs_kd: {}'.format(np.mean(self.his_says_probs_kd)))
                        print('comma_probs_kd: {}'.format(np.mean(self.his_comma_probs_kd)))
                        print('eos_loss: {}'.format(np.mean(self.his_kd_ce_eos_loss)))
                        print('his_norm: {}'.format(np.mean(self.his_norm)))
                        print('gram_acc: 2:{}, 3:{}, 4:{}'.format(np.mean(self.his_2_gram_acc), np.mean(self.his_3_gram_acc),
                              np.mean(self.his_4_gram_acc)))

                        print('gram_loss: 2:{}, 3:{}, 4:{}'.format(np.mean(self.his_2_gram_loss), np.mean(self.his_3_gram_loss),
                              np.mean(self.his_4_gram_loss)))

                    if not self.args.do_parallel_test and ((step + 1) % args.gradient_accumulation_steps == 0 or (
                        # last step in epoch but step is always smaller than gradient_accumulation_steps
                        steps_in_epoch <= args.gradient_accumulation_steps
                        and (step + 1) == steps_in_epoch
                    )):
                        # Gradient clipping
                        if args.max_grad_norm is not None and args.max_grad_norm > 0 and not self.deepspeed:
                            # deepspeed does its own clipping

                            if self.use_amp:
                                # AMP: gradients need unscaling
                                self.scaler.unscale_(self.optimizer)

                            if hasattr(self.optimizer, "clip_grad_norm"):
                                # Some optimizers (like the sharded optimizer) have a specific way to do gradient clipping
                                self.optimizer.clip_grad_norm(args.max_grad_norm)
                            elif hasattr(model, "clip_grad_norm_"):
                                # Some models (like FullyShardedDDP) have a specific way to do gradient clipping
                                model.clip_grad_norm_(args.max_grad_norm)
                            else:
                                # Revert to normal clipping otherwise, handling Apex or full precision
                                nn.utils.clip_grad_norm_(
                                    amp.master_params(self.optimizer) if self.use_apex else model.parameters(),
                                    args.max_grad_norm,
                                )

                        # Optimizer step
                        optimizer_was_run = True
                        if self.deepspeed:
                            pass  # called outside the loop
                        elif is_torch_tpu_available():
                            xm.optimizer_step(self.optimizer)
                        elif self.use_amp:
                            scale_before = self.scaler.get_scale()
                            self.scaler.step(self.optimizer)
                            self.scaler.update()
                            scale_after = self.scaler.get_scale()
                            optimizer_was_run = scale_before <= scale_after
                        else:
                            #print(self.lr_scheduler.get_last_lr()[0])
                            self.optimizer.step()
                        if optimizer_was_run and not self.deepspeed:
                            self.lr_scheduler.step()

                        model.zero_grad()
                        self.state.global_step += 1
                        self.state.epoch = epoch + (step + 1) / steps_in_epoch

                        self.control = self.callback_handler.on_step_end(args, self.state, self.control)
                        self.model.decoding_method = 'non_seq'

                        self._maybe_log_save_evaluate(tr_loss, model, trial, epoch, ignore_keys_for_eval)

                    else:
                        self.control = self.callback_handler.on_substep_end(args, self.state, self.control)

                    if self.control.should_epoch_stop or self.control.should_training_stop:
                        break

            self.control = self.callback_handler.on_epoch_end(args, self.state, self.control)
            # self.model.decoding_method = 'seq'
            # self._maybe_log_save_evaluate(tr_loss, model, trial, epoch, ignore_keys_for_eval)

            self._maybe_log_save_evaluate(tr_loss, model, trial, epoch, ignore_keys_for_eval)
            #print('epoch_b_rewards {}'.format(np.mean(epoch_b_rewards)))
            for name, v in epoch_b_rewards.items():
                print('At step {}, his_b_rewards {} = {}'.format(step, name, np.mean(v)))
            print('epoch_ce_loss {}'.format(np.mean(epoch_ce_loss)))
            # for name, v in epoch_b_rewards.items():
            #     all_b_rewards[name].extend(v)
            #all_b_rewards.extend(epoch_b_rewards)
            all_ce_loss.extend(epoch_ce_loss)
            #print('all_b_rewards {}'.format(np.mean(all_b_rewards)))
            print('all_ce_loss {}'.format(np.mean(all_ce_loss)))
            if DebugOption.TPU_METRICS_DEBUG in self.args.debug:
                if is_torch_tpu_available():
                    # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
                    xm.master_print(met.metrics_report())
                else:
                    logger.warning(
                        "You enabled PyTorch/XLA debug metrics but you don't have a TPU "
                        "configured. Check your training configuration if this is unexpected."
                    )
            if self.control.should_training_stop:
                break
            if self.args.do_parallel_test:
                print('At step {}, his_b_rewards {}'.format(step, np.mean(his_b_rewards)))
                print('At step {}, his_ce_loss {}'.format(step, np.mean(his_ce_loss)))
        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of training
            delattr(self, "_past")
        if self.args.do_parallel_test:
            import pickle as pkl
            #pkl.dump(do_parallel_pkl, open('{}_do_parallel_pkl'.format(self.args.exp_name), 'wb'))
        logger.info("\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")
        if args.load_best_model_at_end and self.state.best_model_checkpoint is not None:
            # Wait for everyone to get here so we are sur the model has been saved by process 0.
            if is_torch_tpu_available():
                xm.rendezvous("load_best_model_at_end")
            elif args.local_rank != -1:
                dist.barrier()

            logger.info(
                f"Loading best model from {self.state.best_model_checkpoint} (score: {self.state.best_metric})."
            )

            best_model_path = os.path.join(self.state.best_model_checkpoint, WEIGHTS_NAME)
            if os.path.exists(best_model_path):
                # We load the model state dict on the CPU to avoid an OOM error.
                state_dict = torch.load(best_model_path, map_location="cpu")
                # If the model is on the GPU, it still works!
                self._load_state_dict_in_model(state_dict)
            else:
                logger.warn(
                    f"Could not locate the best model at {best_model_path}, if you are running a distributed training "
                    "on multiple nodes, you should activate `--save_on_each_node`."
                )

            if self.deepspeed:
                self.deepspeed.load_checkpoint(
                    self.state.best_model_checkpoint, load_optimizer_states=False, load_lr_scheduler_states=False
                )


        self._total_loss_scalar += tr_loss.item()
        train_loss = self._total_loss_scalar / self.state.global_step

        metrics = speed_metrics("train", start_time, num_samples=num_train_samples, num_steps=self.state.max_steps)
        self.store_flos()
        metrics["total_flos"] = self.state.total_flos
        metrics["train_loss"] = train_loss

        self.is_in_train = False

        self._memory_tracker.stop_and_update_metrics(metrics)

        self.log(metrics)

        self.control = self.callback_handler.on_train_end(args, self.state, self.control)

        return TrainOutput(self.state.global_step, train_loss, metrics)


    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.

        Subclass and override to inject custom behavior.

        Args:
            model (:obj:`nn.Module`):
                The model to train.
            inputs (:obj:`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument :obj:`labels`. Check your model's documentation for all accepted arguments.

        Return:
            :obj:`torch.Tensor`: The tensor with training loss on this batch.
        """
        model.train()
        inputs = self._prepare_inputs(inputs)

        if is_sagemaker_mp_enabled():
            scaler = self.scaler if self.use_amp else None
            loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps, scaler=scaler)
            return loss_mb.reduce_mean().detach().to(self.args.device)

        if self.use_amp:
            with autocast():
                if self.args.do_parallel_test:
                    model.eval()
                    with torch.no_grad():
                        loss, outputs, rl_loss, b_reward = self.compute_rl_loss(model, inputs, return_outputs=True)
                else:
                    loss, outputs, rl_loss, b_reward = self.compute_rl_loss(model, inputs, return_outputs=True)

        else:
            if self.args.do_parallel_test:
                model.eval()
                with torch.no_grad():
                    loss, outputs, rl_loss, b_reward = self.compute_rl_loss(model, inputs, return_outputs=True)
            else:

                loss, outputs, rl_loss, b_reward = self.compute_rl_loss(model, inputs, return_outputs=True)

        if 'error' in b_reward:
            model.zero_grad()
            return loss.detach() * self.args.gradient_accumulation_steps, torch.tensor(0).cuda(), {}

        if self.args.update_mask_rate:
            model.mask_rate = self.args.st_mask_rate + (self.args.max_mask_rate-self.args.st_mask_rate) * float(self.state.global_step) / self.args.all_steps
            if self.args.update_rl_weight:
                self.args.rl_weight = 0.5 *  float(self.state.global_step) / self.args.all_steps
            model.mask_rate = min(self.args.max_mask_rate, model.mask_rate)


        ce_loss = loss.clone()
        ce_losses = loss



        if self.args.use_dynamic_rl_weight:
            loss = (1-self.dynamic_rl_weight) * -rl_loss + self.dynamic_rl_weight * ce_losses
        else:
            # print(self.args.rl_weight,  self.loss_weight)
            if self.args.rl_weight < 1:
                if self.args.naive_seq_baseline or self.args.add_rl_loss:

                    loss = self.args.rl_weight * rl_loss + ce_losses
                    self.his_rl_loss.append(rl_loss.detach().clone().cpu().numpy())
                else:
                    loss = self.args.rl_weight * -rl_loss + (1 - self.args.rl_weight) * ce_losses
                    self.his_rl_loss.append(-rl_loss.detach().clone().cpu().numpy())
            else:

                loss = self.args.rl_weight * rl_loss + ce_losses
                # print(loss)
                # loss = self.args.loss_weight * loss
                self.his_rl_loss.append(rl_loss.detach().clone().cpu().numpy())

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        if self.args.gradient_accumulation_steps > 1 and not self.deepspeed:
            # deepspeed handles loss scaling by gradient_accumulation_steps in its `backward`
            loss = loss / self.args.gradient_accumulation_steps
            # print(his_loss * accu)
        st = time.time()
        if not self.args.do_parallel_test:
            if self.use_amp:
                self.scaler.scale(loss).backward()
            elif self.use_apex:
                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            elif self.deepspeed:
                # loss gets scaled under gradient_accumulation_steps in deepspeed
                loss = self.deepspeed.backward(loss)
            else:
                loss.backward()
        ed = time.time()-st
        self.his_time.append(ed)


        return loss.detach() * self.args.gradient_accumulation_steps, ce_loss.detach(), b_reward



    @timer
    def compute_rl_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """

        if self.smoother is not None and "labels" in inputs:
            #labels = inputs.pop("labels")
            labels = inputs['labels']
        else:
            labels = None

        if isinstance(model, torch.nn.DataParallel) or isinstance(model, torch.nn.parallel.DistributedDataParallel):
            tmp_tokenizer = model.module.tokenizer

        else:
            tmp_tokenizer = model.tokenizer
        self.tmp_tokenizer = tmp_tokenizer
        def add_padding_(raw_txt, pad_id):
            txts_ids = [tmp_tokenizer.encode(txt) for txt in raw_txt]
            for t in txts_ids:
                assert len(t) != 0
            padding_txts_ids = []
            batch_max_seq_len = max([len(txt) for txt in txts_ids])
            batch_max_seq_len = min(batch_max_seq_len, 142)
            for txt_ids in txts_ids:
                padding_txts_ids.append(
                    txt_ids[:batch_max_seq_len] + [pad_id] * (batch_max_seq_len - len(txt_ids[:batch_max_seq_len])))
            return torch.tensor(padding_txts_ids, dtype=torch.long).cuda()

        beams = 16
        seq_num = 16

        if self.args.seq_decode_model == 't5':
            eos_token_id = 1
            pad_token_id = 0
        else:
            if self.args.seq_decode_model == 'bart':
                eos_token_id = 2  # bart pad = 1,
                pad_token_id = 1  # bart pad = 1,
            elif self.args.seq_decode_model == 'pegasus':
                eos_token_id = 1
                pad_token_id = 0

        inputs['attention_mask'] = ~inputs['input_ids'].eq(pad_token_id)

        if self.args.seq_decode_model == 'bart':
            inputs['labels'] = inputs['labels'][:, 1:]

        ori_inputs = copy.deepcopy(inputs)
        q2 = inputs['q2']
        if '<#QUERY#>' in q2[0]:
            q2 = [e.split('<#QUERY#>')[0] for e in q2]

        mask_labels = None
        masked_pos_shift = None
        masked_pos_non_shift = None
        base_y_b = None

        if self.args.use_normal_ce:
            inputs_ce = copy.deepcopy(inputs)

        inputs['not_seq_decode'] = True

        if self.args.seq_decode_model == 't5':
            eos_token_id = 1
            pad_token_id = 0
        elif self.args.seq_decode_model == 'pegasus':
            eos_token_id = 1
            pad_token_id = 0
        elif self.args.seq_decode_model == 'bart':
            eos_token_id = 2
            pad_token_id = 1

        model.train()
        if self.args.use_eval:
            model.eval()

        def compute_acc_and_loss(max_ids, labels_, mp, log_probs_all):
            tot_cont = 0
            dif_cont = 0
            # acc max_ids -> labels
            # loss sum of log_probs
            masked_probs = log_probs_all.gather(2, max_ids.unsqueeze(2)).squeeze()
            eos_probs = log_probs_all.gather(2, torch.ones_like(
                max_ids.unsqueeze(2)).long().cuda() * eos_token_id).squeeze()
            says_probs = log_probs_all.gather(2, torch.ones_like(
                max_ids.unsqueeze(2)).long().cuda() * 161).squeeze()
            comma_probs = log_probs_all.gather(2, torch.ones_like(
                max_ids.unsqueeze(2)).long().cuda() * 4).squeeze()
            self.his_eos_probs.append(eos_probs.mean().clone().detach().cpu().numpy())
            self.his_says_probs.append(says_probs.mean().clone().detach().cpu().numpy())
            self.his_comma_probs.append(comma_probs.mean().clone().detach().cpu().numpy())
            pred_acc = max_ids == labels_
            # print(pred_acc)
            batch_p_numpy = mp.clone().detach().cpu().numpy()
            batch_2_gram_pos = []
            batch_3_gram_pos = []
            batch_4_gram_pos = []
            batch_n_gram_pos = []
            # print(batch_p_numpy)
            labels_np = labels_.cpu().clone().numpy()
            for k, p_n in enumerate(batch_p_numpy):
                cont_mask_num = 0
                _2_gram_pos = [0] * len(labels_np[k])
                _3_gram_pos = [0] * len(labels_np[k])
                _4_gram_pos = [0] * len(labels_np[k])
                _n_gram_pos = [0] * len(labels_np[k])
                for i in range(0, len(p_n)):
                    if p_n[i] == 0:
                        break
                    if i > 0 and p_n[i] == p_n[i - 1] + 1:
                        cont_mask_num += 1
                    elif i == 0: # 0 or not cont from last pos
                        cont_mask_num = 1
                    else:
                        cont_mask_num = 1
                    # print(cont_mask_num, p_n[i])
                    if labels_np[k][p_n[i]+1] != -100:
                        if cont_mask_num >= 1:
                            _n_gram_pos[p_n[i] + 1] = 1
                        if cont_mask_num == 1:
                            _2_gram_pos[p_n[i]+1] = 1
                        if cont_mask_num == 2:
                            _3_gram_pos[p_n[i]+1] = 1
                        if cont_mask_num == 3:
                            _4_gram_pos[p_n[i]+1] = 1
                batch_2_gram_pos.append(_2_gram_pos)
                batch_3_gram_pos.append(_3_gram_pos)
                batch_4_gram_pos.append(_4_gram_pos)
                batch_n_gram_pos.append(_n_gram_pos)
            # print(batch_2_gram_pos)
            # print(batch_3_gram_pos)
            batch_2_gram_pos = torch.tensor(batch_2_gram_pos).long().cuda()
            batch_3_gram_pos = torch.tensor(batch_3_gram_pos).long().cuda()
            batch_4_gram_pos = torch.tensor(batch_4_gram_pos).long().cuda()
            batch_n_gram_pos = torch.tensor(batch_n_gram_pos).long().cuda()
            _2_gram_loss = -(masked_probs * batch_2_gram_pos).sum() / (batch_2_gram_pos.sum())
            _3_gram_loss = -(masked_probs * batch_3_gram_pos).sum()/(batch_3_gram_pos.sum())
            _4_gram_loss = -(masked_probs * batch_4_gram_pos).sum()/(batch_4_gram_pos.sum())

            _2_gram_acc = (pred_acc * batch_2_gram_pos).sum() / (batch_2_gram_pos.sum())
            _3_gram_acc = (pred_acc * batch_3_gram_pos).sum() / (batch_3_gram_pos.sum())
            _4_gram_acc = (pred_acc * batch_4_gram_pos).sum() / (batch_4_gram_pos.sum())
            if batch_2_gram_pos.sum() != 0:
                self.his_2_gram_acc.append(_2_gram_acc.cpu())
                self.his_2_gram_loss.append(_2_gram_loss.cpu())
            if batch_3_gram_pos.sum() != 0:
                self.his_3_gram_acc.append(_3_gram_acc.cpu())
                self.his_3_gram_loss.append(_3_gram_loss.cpu())
            if batch_4_gram_pos.sum() != 0:
                self.his_4_gram_acc.append(_4_gram_acc.cpu())
                self.his_4_gram_loss.append(_4_gram_loss.cpu())
            return batch_2_gram_pos, batch_3_gram_pos, batch_4_gram_pos, batch_n_gram_pos


        def compute_dif(y_b, mp):
            tot_cont = 0
            dif_cont = 0
            p_numpy = mp.clone().detach().cpu().numpy()
            pre_output_ids = y_b.tolist()
            for k, p_n in enumerate(p_numpy):
                for i in range(len(p_n) - 1, -1, -1):
                    if i > 0 and p_n[i] == p_n[i - 1] + 1:
                        tot_cont += 1

                        if pre_output_ids[k][p_n[i]] != pre_output_ids[k][p_n[i - 1]]:
                            dif_cont += 1

            if tot_cont != 0:
                self.his_dif_cont_rate.append(dif_cont / tot_cont)


        pre_gen_scores = None
        not_normal_log_probs = None
        if self.args.kd_inputs:
            if '<#REFS#>' in inputs['q2'][0]:
                refs = [e.split('<#REFS#>')[1] for e in inputs['q2']]
                inputs['q2'] = [e.split('<#REFS#>')[0] for e in inputs['q2']]
            else:
                refs = None
            model.train()
            def get_kd_inputs(inputs):
                import random
                if '<#QUERY#>' in inputs['q2'][0]:
                    tmp_q2 = [t.split('<#QUERY#>')[1] for t in inputs['q2']]
                else:
                    tmp_q2 =  inputs['q2']
                cands = [t for e in tmp_q2 for t in random.sample(e.split('<#SEP#>')[:], self.args.cand_num)]
                if '<#SCORE#>' not in inputs['q2'][0] and (self.args.kd_inputs_best or self.args.kd_inputs_worst):
                    hyp = [t for e in tmp_q2 for t in e.split('<#SEP#>')]
                    ref = self.get_ads_str(inputs["labels"], tmp_tokenizer)
                    seq_num = self.model.config.sample_num + 1
                    expand_ref = [e for e in ref for _ in range(seq_num)]
                    rouge_scores = self.rl_agent.get_rouge(hyp, expand_ref, '12')
                    batch_size = inputs["input_ids"].shape[0]
                    hyp = [(e, rouge_scores[i]) for i, e in enumerate(hyp)]
                    cands = []
                    for _ind in range(batch_size):
                        hyps = hyp[_ind*seq_num:(_ind+1)*seq_num]
                        hyps = sorted(hyps, key=lambda item:-item[1])
                        hyps = [e[0] +'<#SCORE#>'+ str(e[1]) for e in hyps]
                        hyps = '<#SEP#>'.join(hyps)
                        cands.append(hyps)
                    tmp_q2 = cands

                if self.args.cand_num == 1 and self.args.kd_inputs_best:
                    cands = [t for e in tmp_q2 for t in random.sample(e.split('<#SEP#>')[:1], self.args.cand_num)]
                if self.args.cand_num == 1 and self.args.kd_inputs_worst:
                    cands = [t for e in tmp_q2 for t in random.sample(e.split('<#SEP#>')[-1:], self.args.cand_num)]

                if self.args.use_pre_gen_scores:
                    pre_gen_scores = [float(t.split('<#SCORE#>')[1]) for t in cands]
                    pre_gen_scores = torch.tensor(pre_gen_scores, dtype=torch.float).cuda()
                    pre_gen_scores = pre_gen_scores.reshape(-1, 1)
                else:
                    pre_gen_scores = None

                cands = [t.split('<#SCORE#>')[0] for t in cands]

                cands = add_padding_(cands, pad_id=-100)

                if self.args.seq_decode_model == 'bart':
                    cands = cands[:, 1:]

                kd_seq_labels = cands
                kd_inputs = copy.deepcopy(inputs)
                if self.args.cand_num != 1:
                    bs = kd_inputs['input_ids'].shape[0]
                    kd_inputs['labels'] = kd_inputs['labels'].unsqueeze(2).repeat(1, self.args.cand_num,
                                                                                                  1).reshape(
                        bs * self.args.cand_num, -1)
                    kd_inputs['query'] = [e for e in kd_inputs['query'] for _ in range(self.args.cand_num)]
                    kd_inputs['q2'] = [e for e in kd_inputs['q2'] for _ in range(self.args.cand_num)]
                    kd_inputs['doc'] = [e for e in kd_inputs['doc'] for _ in range(self.args.cand_num)]

                if not self.args.not_replace_kd:
                    if random.random() < self.args.replace_kd_prob:
                        kd_inputs['labels'] = kd_seq_labels
                        second_kd_inputs_labels = kd_inputs['labels']
                    else:
                        second_kd_inputs_labels = kd_inputs['labels']
                else:
                    second_kd_inputs_labels = kd_inputs['labels']
                    if self.args.mask_gt:
                        mask_labels, masked_pos_shift, masked_pos_non_shift, decoder_input_ids = model_return_dict[
                                                                                                 15:19]
                        kd_inputs['mask_labels'] = mask_labels
                        kd_inputs['masked_pos_shift'] = masked_pos_shift
                        kd_inputs['masked_pos_non_shift'] = masked_pos_non_shift
                return kd_inputs, second_kd_inputs_labels, pre_gen_scores
            if self.args.seq_decode_model == 't5':
                model.scoring_mode()
            elif self.args.seq_decode_model == 'bart':
                model.model.scoring_mode()
            elif self.args.seq_decode_model == 'pegasus':
                model.model.scoring_mode()

            kd_inputs, second_kd_inputs_labels, pre_gen_scores = get_kd_inputs(inputs)

            def get_merge_gt_kd_inputs(inputs, kd_inputs):
                batch_size = inputs['labels'].shape[0]

                s_res = kd_inputs['labels'].reshape(batch_size, 1, -1)
                g_res = inputs['labels'].reshape(batch_size, 1, -1)
                max_len = max(s_res.shape[-1], g_res.shape[-1])
                if max_len != s_res.shape[-1]:

                    pads = -100 * torch.ones(batch_size, 1, max_len - s_res.shape[-1], dtype=torch.long).cuda()

                    s_res = torch.cat([s_res, pads], dim=-1)
                second_kd_inputs_labels = s_res
                if max_len != g_res.shape[-1]:
                    pads = -100 * torch.ones(batch_size, 1, max_len - g_res.shape[-1], dtype=torch.long).cuda()
                    g_res = torch.cat([g_res, pads],
                                      dim=-1)

                s_res = torch.cat([g_res, s_res], dim=1)
                s_res = s_res.reshape(batch_size*2, -1)
                kd_inputs['labels'] = s_res
                return kd_inputs, second_kd_inputs_labels

            kd_inputs, second_kd_inputs_labels = get_merge_gt_kd_inputs(inputs, kd_inputs)
            ori_kd_inputs = copy.deepcopy(kd_inputs)
            batch_size = second_kd_inputs_labels.shape[0]
            second_kd_inputs_labels = second_kd_inputs_labels.reshape(batch_size, -1)
            model.train()

            try:
                start = time.time()
                merge_model_return_dict = model(**kd_inputs)
                self.his_time.append(time.time()-start)
                # raise ValueError("")
                # print('forward2 共耗时约 {:.2f} 秒'.format(time.time() - start))
            except ValueError as e:
                print(e)
                print(kd_inputs)
                return (torch.tensor(0).cuda(), None, 0, {'error':1})



            if self.args.seq_decode_model == 't5':
                model.generation_mode()
            elif self.args.seq_decode_model == 'bart':
                model.model.generation_mode()
            elif self.args.seq_decode_model == 'pegasus':
                model.model.generation_mode()
            if self.args.seq_decode_model == 't5':
                model.scoring_mode()
            elif self.args.seq_decode_model == 'bart':
                model.model.scoring_mode()
            elif self.args.seq_decode_model == 'pegasus':
                model.model.scoring_mode()
            if len(merge_model_return_dict) == 2:
                model_return_dict, kd_model_return_dict = merge_model_return_dict

                outputs, y_b, y_s, max_ids, masked_ids, input_ids, labels_, non_zero_sum_tensor, log_probs, y_zero_b, y_zero_s, y_zero_labels, truth_log_probs, log_probs_all, lm_logits = model_return_dict[
                                                                                                                                                                                           :15]

            compute_acc_and_loss(
                max_ids, labels_, model_return_dict[17], log_probs_all)


            model.train()

            _, y_b, y_s, max_ids, masked_ids, input_ids, _, non_zero_sum_tensor, log_probs, y_zero_b, y_zero_s, y_zero_labels, truth_log_probs, log_probs_all, _ = kd_model_return_dict[
                                                                                :15]

            ori_log_probs = log_probs.clone().detach()
            ori_log_probs_all = log_probs_all.clone().detach()
            eos_probs = log_probs_all.gather(2, torch.ones_like(
                max_ids.unsqueeze(2)).long().cuda() * eos_token_id).squeeze()
            says_probs = log_probs_all.gather(2, torch.ones_like(
                max_ids.unsqueeze(2)).long().cuda() * 161).squeeze()
            comma_probs = log_probs_all.gather(2, torch.ones_like(
                max_ids.unsqueeze(2)).long().cuda() * 4).squeeze()

            self.his_eos_probs_kd.append(eos_probs.mean().clone().detach().cpu().numpy())
            self.his_says_probs_kd.append(says_probs.mean().clone().detach().cpu().numpy())
            self.his_comma_probs_kd.append(comma_probs.mean().clone().detach().cpu().numpy())

            predict_baseline = None



            if self.args.seq_decode_model == 't5':
                model.generation_mode()
            elif self.args.seq_decode_model == 'bart':
                model.model.generation_mode()
            elif self.args.seq_decode_model == 'pegasus':
                model.model.generation_mode()


            mask_labels, masked_pos_shift, masked_pos_non_shift, decoder_input_ids = kd_model_return_dict[15:19]
            if self.args.mask_gt:
                kd_inputs['mask_labels'] = mask_labels
                kd_inputs['masked_pos_shift'] = masked_pos_shift
                kd_inputs['masked_pos_non_shift'] = masked_pos_non_shift

            # assert 1==0
            compute_dif(y_b, masked_pos_non_shift)


            if self.args.eos_replace:
                second_labels = second_kd_inputs_labels  # seq gen res
                bs = second_labels.shape[0]
                second_labels = secondc_labels.unsqueeze(1).expand(-1, model.config.sample_num + 1, -1)
                second_labels = second_labels.reshape(bs * (model.config.sample_num + 1), -1)
                traces = y_s
                b_traces = y_b
                eos_mask = torch.zeros_like(traces, dtype=torch.bool)
                eos_mask |= (second_labels == eos_token_id)  # other to <\s>
                eos_mask |= (traces == eos_token_id)
                eos_mask |= (b_traces == 1)  # 1 if gen <\s>   <\s> to other
                y_s = traces * (~eos_mask) + second_labels * eos_mask
                y_b = b_traces * (~eos_mask) + second_labels * eos_mask
                log_probs = log_probs * (~eos_mask).float()

        if self.args.mask_gt:
            mask_labels, masked_pos_shift, masked_pos_non_shift, decoder_input_ids = model_return_dict[15:19]
            inputs['mask_labels'] = mask_labels
            inputs['masked_pos_shift'] = masked_pos_shift
            inputs['masked_pos_non_shift'] = masked_pos_non_shift

        length_penalty = self.args.training_length_penalty

        length_penalty = self.args.training_length_penalty

        inputs_brio = copy.deepcopy(inputs)

        if (model.config.sample_num != 0 or self.args.cand_num !=1):
            cand_mask = ~log_probs.data.eq(0)
            _116_mask = True
            if _116_mask:
                if self.args.seq_decode_model == 'bart':

                    new_cand_mask = cand_mask & ~(
                            (y_s != 2) & (y_s != 0) & (
                            y_s != 1))
                    cand_mask = cand_mask & (y_s != 2) & (
                            y_s != 0) & (y_s != 1)

                    batch_size = labels_.shape[0]


                if self.args.seq_decode_model == 't5':
                    cand_mask = cand_mask & (y_s != 259) & (y_s != 260) & (y_s != 261) & (y_s != eos_token_id) & (
                                y_s != pad_token_id) & (y_s != 250100) & (y_s != 250101)
                if self.args.seq_decode_model == 't5' and self.model.config.tokenizer_name == 't5-small':
                    cand_mask = cand_mask & (y_s != 5) & (y_s != 6) & (y_s != eos_token_id) & (y_s != 2)
                if self.args.seq_decode_model == 'pegasus':
                    new_cand_mask = cand_mask & ~((y_s != eos_token_id) & (y_s != pad_token_id))
                    cand_mask = cand_mask & (y_s != eos_token_id) & (y_s != pad_token_id)

                # cand_mask = cand_mask & (y_s != 264) & (y_s != 250100) & (y_s != 250101)
                if self.args.exclude_eos:
                    cand_mask = cand_mask & (y_s != eos_token_id)

                if self.args.new_cand_mask:
                    cand_mask = cand_mask.reshape(batch_size, -1, cand_mask.shape[-1])
                    # print(cand_mask)
                    cand_num = cand_mask.shape[1]
                    cand_mask = cand_mask.sum(1)
                    cand_mask = cand_mask.data.eq(cand_num)
                    cand_mask = cand_mask.unsqueeze(1).repeat(1, cand_num, 1)
                    cand_mask = cand_mask.reshape(-1, cand_mask.shape[-1])

                if self.args.new_cand_mask_y_s:
                    new_cand_mask = new_cand_mask.reshape(batch_size, -1, cand_mask.shape[-1])
                    new_cand_mask = new_cand_mask.sum(1)
                    new_cand_mask = new_cand_mask > 0
                    new_cand_mask = new_cand_mask.unsqueeze(1).repeat(1, cand_num, 1)
                    new_cand_mask = new_cand_mask.reshape(-1, cand_mask.shape[-1])
                    y_s = y_s * ~new_cand_mask + second_kd_inputs_labels.unsqueeze(1).repeat(1, cand_num, 1).reshape(-1,
                                                                                                                     cand_mask.shape[
                                                                                                                         -1]) * new_cand_mask

                log_probs = log_probs * self.args.scale

                not_normal_log_probs = log_probs.clone()

                for_count_y_s = y_s * cand_mask + torch.ones_like(y_s).long().cuda() * -1 * ~cand_mask
                for_count_y_s = for_count_y_s.cpu().numpy()

                cand_mask = 1 / (((cand_mask.sum(-1)) ** length_penalty).unsqueeze(1))
                cand_mask = torch.where(torch.isinf(cand_mask), torch.full_like(cand_mask, 0), cand_mask)
                if self.args.length_normalize_4_rl:
                    log_probs = log_probs * cand_mask
                    log_probs = log_probs / (model.config.sample_num + 1) / self.args.cand_num


        start = time.time()

        if self.model.config.do_rl:

            rl_loss, b_reward_tensor, b_zero_tks, zero_tks, labels_zero_tks, all_gen_traces_tk, all_gen_b_traces_tk, all_reward, other_reward = self.rl_ids_2_str(y_b, y_s, max_ids, masked_ids, input_ids, labels_,
                                                  non_zero_sum_tensor, log_probs, q2, tmp_tokenizer, y_zero_b,
                                                  y_zero_s, y_zero_labels, base_y_b, truth_log_probs, pre_gen_scores, predict_baseline=predict_baseline, not_normal_log_probs=not_normal_log_probs, raw_src=inputs['query'], inputs_brio=inputs_brio, refs=refs,
                                        # base_traces=s_res)
                        base_traces=second_kd_inputs_labels)
            self.current_target = all_reward
            reward_dist_dict = {}

            all_reward = all_reward.reshape(batch_size, -1)
            all_reward = (all_reward - all_reward.mean(1).unsqueeze(1)).reshape(-1).cpu().numpy()

            for i, seq in enumerate(for_count_y_s):
                for tk in seq:
                    if tk not in reward_dist_dict:
                        reward_dist_dict[tk] = all_reward[i]
                    else:
                        reward_dist_dict[tk] += all_reward[i]
            for tk in reward_dist_dict.keys():
                if tk not in self.reward_dist_dict:
                    self.reward_dist_dict[tk] = [reward_dist_dict[tk]]
                else:
                    self.reward_dist_dict[tk].append(reward_dist_dict[tk])

            b_reward_dict = {}
            b_reward_tensor = b_reward_tensor.transpose(0, 1)
            for i, name in enumerate(self.args.rewards.split(',')):
                b_reward_dict[name] = b_reward_tensor[i].float().mean().detach().cpu().numpy()
                if name not in self.his_reward_dict['gen']:
                    self.his_reward_dict['gen'][name] = []
                self.his_reward_dict['gen'][name].append(b_reward_dict[name])
                self.his_reward_dict['gen'][name] = self.his_reward_dict['gen'][name][-2000:]

            # TODO: this needs to be fixed and made cleaner later.
            if self.args.past_index >= 0:
                self._past = outputs[self.args.past_index]
            if self.args.use_normal_ce:
                inputs_ce['normal_forward_no_mask'] = True
                model_return_dict = model(**inputs_ce)
                outputs = model_return_dict[0]

        if self.smoother is not None and labels_ is not None:
            loss = self.smoother(outputs, labels_)
        else:
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        return (loss, outputs, rl_loss, b_reward_dict) if return_outputs else loss


    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        """
        Prediction/evaluation loop, shared by :obj:`Trainer.evaluate()` and :obj:`Trainer.predict()`.

        Works both with or without labels.
        """
        prediction_loss_only = (
            prediction_loss_only if prediction_loss_only is not None else self.args.prediction_loss_only
        )

        # if eval is called w/o train init deepspeed here
        if self.args.deepspeed and not self.deepspeed:

            # XXX: eval doesn't have `resume_from_checkpoint` arg but we should be able to do eval
            # from the checkpoint eventually
            deepspeed_engine, _, _ = deepspeed_init(self, num_training_steps=0, resume_from_checkpoint=None)
            self.model = deepspeed_engine.module
            self.model_wrapped = deepspeed_engine
            self.deepspeed = deepspeed_engine
            # XXX: we don't need optim/sched for inference, but this needs to be sorted out, since
            # for example the Z3-optimizer is a must for zero3 to work even for inference - what we
            # don't need is the deepspeed basic optimizer which is self.optimizer.optimizer
            deepspeed_engine.optimizer.optimizer = None
            deepspeed_engine.lr_scheduler = None

        model = self._wrap_model(self.model, training=False)

        # if full fp16 is wanted on eval and this ``evaluation`` or ``predict`` isn't called while
        # ``train`` is running, halve it first and then put on device
        if not self.is_in_train and self.args.fp16_full_eval:
            model = model.half().to(self.args.device)

        batch_size = dataloader.batch_size

        logger.info(f"***** Running {description} *****")
        if isinstance(dataloader.dataset, collections.abc.Sized):
            logger.info(f"  Num examples = {self.num_examples(dataloader)}")
        else:
            logger.info("  Num examples: Unknown")
        logger.info(f"  Batch size = {batch_size}")

        model.eval()
        if self.args.recover_path != "":
            model.load_state_dict(torch.load(self.args.recover_path))
        self.callback_handler.eval_dataloader = dataloader
        # Do this before wrapping.
        eval_dataset = dataloader.dataset

        if is_torch_tpu_available():
            dataloader = pl.ParallelLoader(dataloader, [self.args.device]).per_device_loader(self.args.device)

        if self.args.past_index >= 0:
            self._past = None

        # Initialize containers
        # losses/preds/labels on GPU/TPU (accumulated for eval_accumulation_steps)
        losses_host = None
        preds_host = None
        labels_host = None
        # losses/preds/labels on CPU (final containers)
        all_losses = None
        all_preds = None
        all_labels = None
        all_q2s = []
        all_raw_src = []
        all_raw_src_origin = []
        # Will be useful when we have an iterable dataset so don't know its length.

        observed_num_examples = 0
        # Main evaluation loop
        for step, inputs in enumerate(dataloader):
            # Update the observed num examples
            observed_batch_size = find_batch_size(inputs)
            if observed_batch_size is not None:
                observed_num_examples += observed_batch_size
                # For batch samplers, batch_size is not known by the dataloader in advance.
                if batch_size is None:
                    batch_size = observed_batch_size
            q2s = inputs.get('q2')
            raw_src = [
                self.model.tokenizer.decode(inputs['input_ids'][i], skip_special_tokens=False, clean_up_tokenization_spaces=False) for i in
                range(len(inputs['input_ids']))]
            # print(raw_src)
            raw_src_origin = raw_src
            raw_src = [e.split()[0] for e in raw_src]
            # Prediction step
            #inputs.pop("labels")
            loss, logits, labels = self.prediction_step(model, inputs, prediction_loss_only, ignore_keys=ignore_keys)

            # Update containers on host
            if loss is not None:
                losses = self._nested_gather(loss.repeat(batch_size))
                losses_host = losses if losses_host is None else torch.cat((losses_host, losses), dim=0)
            if logits is not None:
                logits = self._pad_across_processes(logits)
                logits = self._nested_gather(logits)
                preds_host = logits if preds_host is None else nested_concat(preds_host, logits, padding_index=-100)
            if labels is not None:
                labels = self._pad_across_processes(labels)
                labels = self._nested_gather(labels)
                labels_host = labels if labels_host is None else nested_concat(labels_host, labels, padding_index=-100)
            all_q2s.extend(q2s)
            all_raw_src.extend(raw_src)
            all_raw_src_origin.extend(raw_src_origin)
            self.control = self.callback_handler.on_prediction_step(self.args, self.state, self.control)

            # Gather all tensors and put them back on the CPU if we have done enough accumulation steps.
            if self.args.eval_accumulation_steps is not None and (step + 1) % self.args.eval_accumulation_steps == 0:
                if losses_host is not None:
                    losses = nested_numpify(losses_host)
                    all_losses = losses if all_losses is None else np.concatenate((all_losses, losses), axis=0)
                if preds_host is not None:
                    logits = nested_numpify(preds_host)
                    all_preds = logits if all_preds is None else nested_concat(all_preds, logits, padding_index=-100)
                if labels_host is not None:
                    labels = nested_numpify(labels_host)
                    all_labels = (
                        labels if all_labels is None else nested_concat(all_labels, labels, padding_index=-100)
                    )

                losses_host, preds_host, labels_host = None, None, None

        if self.args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of the evaluation loop
            delattr(self, "_past")

        # Gather all remaining tensors and put them back on the CPU
        if losses_host is not None:
            losses = nested_numpify(losses_host)
            all_losses = losses if all_losses is None else np.concatenate((all_losses, losses), axis=0)
        if preds_host is not None:
            logits = nested_numpify(preds_host)
            all_preds = logits if all_preds is None else nested_concat(all_preds, logits, padding_index=-100)
        if labels_host is not None:
            labels = nested_numpify(labels_host)
            all_labels = labels if all_labels is None else nested_concat(all_labels, labels, padding_index=-100)

        # Number of samples
        if not isinstance(eval_dataset, IterableDataset):
            num_samples = len(eval_dataset)
        # The instance check is weird and does not actually check for the type, but whether the dataset has the right
        # methods. Therefore we need to make sure it also has the attribute.
        elif isinstance(eval_dataset, IterableDatasetShard) and hasattr(eval_dataset, "num_examples"):
            num_samples = eval_dataset.num_examples
        else:
            num_samples = observed_num_examples

        # Number of losses has been rounded to a multiple of batch_size and in a distributed training, the number of
        # samplers has been rounded to a multiple of batch_size, so we truncate.
        if all_losses is not None:
            all_losses = all_losses[:num_samples]
        if all_preds is not None:
            all_preds = nested_truncate(all_preds, num_samples)
        if all_labels is not None:
            all_labels = nested_truncate(all_labels, num_samples)

        # Metrics!
        if self.compute_metrics is not None and all_preds is not None and all_labels is not None:
            #metrics = self.compute_metrics(EvalPrediction(predictions=all_preds, label_ids=all_labels))

            metrics = self.compute_metrics(EvalPrediction(predictions=all_preds, label_ids=all_labels, q2_ids=all_q2s, raw_src=all_raw_src, raw_src_origin=all_raw_src_origin))
        else:
            metrics = {}

        # To be JSON-serializable, we need to remove numpy types or zero-d tensors
        metrics = denumpify_detensorize(metrics)

        if all_losses is not None:
            metrics[f"{metric_key_prefix}_loss"] = all_losses.mean().item()

        # Prefix all keys with metric_key_prefix + '_'
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

        return EvalLoopOutput(predictions=all_preds, label_ids=all_labels, metrics=metrics, num_samples=num_samples)

    def predict(self,description="Inference"):
        test_dataset = self.eval_dataset#self.eval_dataset.get_dataset()['train']
#        self._memory_tracker.start()
        dataloader = self.get_test_dataloader(test_dataset)
        start_time = time.time()
        model = self._wrap_model(self.model, training=False)
        if not self.is_in_train and self.args.fp16_full_eval:
            model = model.half().to(self.args.device)

        batch_size = dataloader.batch_size
        num_examples = self.num_examples(dataloader)
        logger.info(f"***** Running {description} *****")
        logger.info(f"  Num examples = {num_examples}")
        logger.info(f"  Batch size = {batch_size}")


        model.eval()
        with torch.profiler.profile(
            #activities=[torch.profiler.ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(
            wait=2,
            warmup=2,
            active=2,
            repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(self.args.logging_dir),
            with_stack=True
        ) as profiler:
            for step, inputs in enumerate(dataloader):

                batch = {}
                # keep batch features and drop the corresponding ones in inputs
                for fn in self.result_header:
                    if fn in inputs:
                        batch[fn] = inputs.pop(fn)

                loss, logits, labels = self.prediction_step(model, inputs,False)
                print(logits)

                self.outputter(batch,logits)

                if self.args.logging_steps > 0 and step % self.args.logging_steps == 0 and step > 0:
                    current_speed = speed_metrics("inference", start_time, step*self.args.per_device_eval_batch_size * self.args.world_size)
                    current_speed["job_progress"] = step*self.args.per_device_eval_batch_size * self.args.world_size / num_examples
                    self.log(current_speed)

        self.log(speed_metrics("inference", start_time, num_examples))
        self.outputter.close()


    def _maybe_log_save_evaluate(self, tr_loss, model, trial, epoch, ignore_keys_for_eval=None):
        if self.control.should_log:
            logs = {}
            tr_loss_scalar = tr_loss.item()
            # reset tr_loss to zero
            tr_loss -= tr_loss
            logs["loss"] = round(tr_loss_scalar / (self.state.global_step - self._globalstep_last_logged), 4)
            logs["learning_rate"] = self._get_learning_rate()
            logs.update(speed_metrics("train",self.train_start_time,self.args.per_device_train_batch_size * self.args.world_size * (self.state.global_step - self._globalstep_last_logged)))
            self.train_start_time = time.time()

            self._total_loss_scalar += tr_loss_scalar
            self._globalstep_last_logged = self.state.global_step
            self.store_flos()

            num_samples = self.args.per_device_train_batch_size * self.args.world_size * self.state.global_step
            logs["train_num_samples_consumed"] = num_samples
            logs["job_progress"] = self.state.global_step / self.state.max_steps
            self.log(logs)

        if self.control.should_evaluate:
            self.model.config.decoding_method = 'seq'
            metrics = self.evaluate()
            # metrics = self.evaluate(self.pred_dataset)
            num_samples = self.args.per_device_train_batch_size * self.args.world_size * self.state.global_step
            metrics["eval_num_samples_consumed"] = num_samples
            self._report_to_hp_search(trial, epoch, metrics)


class MetricsCalculator:
    def __init__(self, metrics, model_name, cfgs, task_cfgs, tokenizer, model_dict):
        self.metrs = {}
        for m in metrics.split(","):
            _m = self.cvt_mtc(m, True)
            #self.metrs[_m] = getattr(Metrics,_m)(model_name=model_name, cfg=cfgs, customize_cfg=task_cfgs)
            print(_m)
            self.metrs[_m] = getattr(Metrics, _m)(model_name=model_name, cfg=cfgs, customize_cfg=task_cfgs, tokenizer=tokenizer, model_dict=model_dict)

    def __call__(self,EvalPred):
        res = {}
        for m in self.metrs:
            res[m] = self.metrs[m](EvalPred)
        
        res = self.flat_mtrcs(res)
        return res 

    @classmethod
    def cvt_mtc(cls, mtr_name: str, is_root: bool) : 
        """
        convert user-defined metric of form "root_metric:node_metric"
        based on scenarios needs. Normally, root metric is used to init the 
        metric class while node metric is inherited from the root metric but 
        used in the evaluation with a fine-grained name.
        """
        _mtr = mtr_name 
        if ":" in _mtr : 
            if is_root : 
                _mtr = _mtr.split(":")[0]
            else : 
                _mtr = "_".join(_mtr.split(":"))

        return _mtr 
            


    def flat_mtrcs(self, res, prefix="") : 
        """
        Flatten recursively each metric if available and replace the metric in 
        res with a hierachical name. This is most useful for multiple sub-metrics 
        in one metric root category. An alternative is to split metric class 
        for different sub-metrics, which could also be efficient, depending on 
        the repeated implementation.

        Example: 
            replace `{"AUC" : {"overall": 1, "category": 2}}` to `{"AUC_overall" : 1, "AUC_category": 2}`
        
        """
        _res = dict()
        for res_key in res: 
            if isinstance(res[res_key], dict) : 
                node_level = str(prefix) + "_" + str(res_key) if prefix else res_key
                _mtr = res[res_key]
                _res.update(self.flat_mtrcs(_mtr, node_level))
            else : 
                if prefix : 
                    _res[str(prefix) + "_" + str(res_key)] = res[res_key]
                else : 
                    _res[res_key] = res[res_key]

        return _res

@replace(PrinterCallback)
class PrinterCallback(TrainerCallback):
    """
    A bare :class:`~transformers.TrainerCallback` that just prints the logs.
    """

    def on_log(self, args, state, control, logs=None, **kwargs):
        _ = logs.pop("total_flos", None)
        if state.is_local_process_zero:
            logger.info(logs)

class AmlLogger(AzureMLCallback):
    """Improve Huggingface Aml ingegration in two aspects
    1. disable Aml logger if not running inside Aml env
    2. choosing log, log_list, etc dynamically"""
    def __init__(self):
        self.active = False # when False, log function will do nothing
        try:
            from azureml.core import Run
            import azureml.core

            azureml_run = Run.get_context()
            self.azureml_run = azureml_run
            if not isinstance(self.azureml_run, azureml.core.run._OfflineRun):
                self.active = True 
        except:
            pass

        if not self.active:
            print("Cannot get context for AML: disable AmlLogger")

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not self.active:
            return

        if self.azureml_run and state.is_world_process_zero:
            for k, v in logs.items():
                self._emit(k, v)
    
    def _emit(self, name, value):
        """
        :param name: str, name of the metric to be logged
        :param value: scalar|list|tuple|dict with scalar or list values
        """
        if np.isscalar(value):
            self.azureml_run.log(name, value)
        elif isinstance(value, (list, tuple)):
            self.azureml_run.log_list(name, value)
        elif isinstance(value, dict):
            one_value = next(iter(value.values()))
            if np.isscalar(one_value):
                self.azureml_run.log_row(name, value)
            elif isinstance(value, (list, tuple)):
                self.azureml_run.log_table(name, value)
