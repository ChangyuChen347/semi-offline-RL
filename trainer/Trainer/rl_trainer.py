import os
from dataclasses import dataclass
import numpy as np
from numpy.lib.function_base import average, median
import torch
from torch.distributed.distributed_c10d import init_process_group
from transformers import Trainer
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
import torch.nn.functional as F
import random
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
from unirl import RL_env

from transformers import AdamW, get_linear_schedule_with_warmup


from s2s_ft.tokenization_unilm import UnilmTokenizer
from trainer.Trainer import register_trainer
from collections import deque
import collections
from transformers.debug_utils import DebugOption, DebugUnderflowOverflow
from torch.utils.data.distributed import DistributedSampler

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



class HisStatisticInfo:
    def __init__(self, args):
        self.args = args
        self.his_dif_cont_rate = []
        self.his_greedy_rewards = {name: [] for name in args.rewards.split(',')}
        self.his_ce_loss = []
        self.his_loss = []
        self.his_rl_loss = []
        self.reward_dist_dict = {} # token-wise reward
        self.his_reward_dict = {} # sentence reward

        self.his_2_gram_loss = []
        self.his_3_gram_loss = []
        self.his_4_gram_loss = []
        self.his_2_gram_acc = []
        self.his_3_gram_acc = []
        self.his_4_gram_acc = []

        self.print_every = args.print_every
    def update(self):
        for name in self.his_greedy_rewards.keys():
            self.his_greedy_rewards[name] = self.his_greedy_rewards[name][-print_every:]
        self.his_ce_loss = self.his_ce_loss[-print_every:]
        self.his_rl_loss = self.his_rl_loss[-print_every:]
        self.his_loss = self.his_loss[-print_every:]
        self.his_2_gram_acc = self.his_2_gram_acc[-print_every:]
        self.his_3_gram_acc = self.his_3_gram_acc[-print_every:]
        self.his_4_gram_acc = self.his_4_gram_acc[-print_every:]
        self.his_2_gram_loss = self.his_2_gram_loss[-print_every:]
        self.his_3_gram_loss = self.his_3_gram_loss[-print_every:]
        self.his_4_gram_loss = self.his_4_gram_loss[-print_every:]
        self.his_dif_cont_rate = self.his_dif_cont_rate[-print_every:]

    def print_info(self, tokenizer):
        for name, v in self.his_greedy_rewards.items():
            logger.info('At step {}, his_greedy_rewards {} = {}'.format(step, name, np.mean(v)))
        logger.info('At step {}, his_ce_loss {}'.format(step, np.mean(self.his_ce_loss)))
        logger.info('At step {}, his_rl_loss {}'.format(step, np.mean(self.his_rl_loss)))
        logger.info('At step {}, his_loss {}'.format(step, np.mean(self.his_loss)))
        logger.info('At step {}, his_dif_cont_rate'.format(step, np.mean(self.his_dif_cont_rate)))
        logger.info('At step {}, gram_acc: 2:{}, 3:{}, 4:{}'.format(step, np.mean(self.his_2_gram_acc),
                                                                    np.mean(self.his_3_gram_acc),
                                                                    np.mean(self.his_4_gram_acc)))
        for i, name in enumerate(self.args.rewards.split(',')):
            if name not in self.his_reward_dict:
                self.his_reward_dict[name] = []
            mean_his_reward = np.mean(self.his_reward_dict[name])
            logger.info('name: {}, reward: {}'.format(name, mean_his_reward))

        # Sorted by the times of appearance
        to_print_count = sorted(self.reward_dist_dict.items(), key=lambda item: -len(item[1]))
        to_print_count = to_print_count[:100]
        # Sorted by the values
        to_print_count = sorted(to_print_count, key=lambda item: -np.mean(item[1]))
        to_print_dist = sorted(self.reward_dist_dict.items(), key=lambda item: -np.mean(item[1]))
        try:
            # Print the tokens and their mean rewards for the top 20 non-"-1" tokens from to_print_dist
            logger.info('to_print_dist',
                        [(tokenizer.convert_ids_to_tokens([t[0]]), np.mean(t[1])) for t in
                         to_print_dist[:20] if t[0] != -1])
        except (OverflowError, UnicodeEncodeError) as e:
            logger.info('to_print_dist top 20',
                        [t[0] for t in
                         to_print_dist[:20]])
            logger.info(e)
        try:
            # Print the tokens and their mean rewards for the last 20 non-"-1" tokens from to_print_dist
            logger.info('to_print_dist',
                        [(tokenizer.convert_ids_to_tokens([t[0]]), np.mean(t[1])) for t in
                         to_print_dist[-20:] if t[0] != -1])
        except (OverflowError, UnicodeEncodeError) as e:
            logger.info('to_print_dist last 20',
                        [t[0] for t in
                         to_print_dist[-20:]])
            logger.info(e)
        try:
            # Print the tokens and their mean rewards for the top 20 non-"-1" tokens from to_print_count
            logger.info('to_print_count top 20',
                        [(tokenizer.convert_ids_to_tokens([t[0]]), np.mean(t[1])) for t in
                         to_print_count[:20] if t[0] != -1])
        except (OverflowError, UnicodeEncodeError) as e:
            logger.info('to_print_count top 20',
                        [t[0] for t in
                         to_print_count[:20]])
            logger.info(e)
        try:
            # Print the tokens and their mean rewards for the last 20 non-"-1" tokens from to_print_count
            logger.info('to_print_count last 20',
                        [(tokenizer.convert_ids_to_tokens([t[0]]), np.mean(t[1])) for t in
                         to_print_count[-20:] if t[0] != -1])
        except (OverflowError, UnicodeEncodeError) as e:
            logger.info('to_print_count last 20',
                        [t[0] for t in
                         to_print_count[-20:]])
            logger.info(e)

@register_trainer("rl")
class Trainer(Trainer):
    def __init__(self, model, args, model_args, task_args, train_dataset, eval_dataset, auto_tokenizer, pred_dataset):
        data_collator = CustomizeCollator(train_dataset,eval_dataset,pred_dataset)
        self.pred_dataset = pred_dataset.get_dataset() if pred_dataset else None
        # resize embedding, will do nothing if `old_num_tokens==new_num_tokens`
        model.resize_token_embeddings(len(auto_tokenizer))
        self.args = args

        self.tokenizer = auto_tokenizer

        self.rl_env = RL_env(
            self.tokenizer,
            reward_type=self.args.reward_type,
            rewards=self.args.rewards,
            rouge_type=self.args.rouge_type,
            sample_num=model.config.sample_num,
            cand_num=self.args.cand_num,
            local_rank=self.args.local_rank,
            loss_type=self.args.loss_type,
            margin=self.args.margin
        )
        if args.do_train:
            if args.eval_metrics == "eval_loss":
                metrics_calculator = None
                args.metric_for_best_model = "eval_loss"
            else:
                #metrics_calculator = MetricsCalculator(args.eval_metrics, model_args._name_or_path, args, task_args) if args.eval_metrics else None
                model_dict = {}
                if 'fact' in self.args.rewards:
                    model_dict['fact'] = self.rl_env.fact_model
                metrics_calculator = MetricsCalculator(args.eval_metrics, model_args._name_or_path, args, task_args,
                                                       auto_tokenizer, model_dict) if args.eval_metrics else None

                args.metric_for_best_model = MetricsCalculator.cvt_mtc(args.eval_metrics.split(",")[0], False)
        else:
            metrics_calculator = None
        if args.do_predict:

            self.result_header = args.result_header.split(
                ",") if "," in args.result_header else eval_dataset.feature_extractor.model_header

            self.outputter = getattr(Outputter, args.output_type)(args, model_args._name_or_path,
                                                                  tokenizer=auto_tokenizer)

        if 'azure_ml' in args.report_to:
            args.report_to.remove('azure_ml')

        # adjust labels for both loss and metrics computation 
        # in case there are multiple label components 
        default_label_names = ["labels"]
        args.label_names = args.label_names.split(",") if args.label_names else default_label_names

        if self.args.recover != "":
            model.load_state_dict(torch.load(args.recover))


        if 'brio' in self.args.rewards:
            self.rl_env.brio_model = copy.deepcopy(model)
            self.rl_env.brio_model.cuda()
            # self.rl_env.brio_model.load_state_dict(torch.load('brio_model'))

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

    def get_tks(self, y, tokenizer, process=True):

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



        return tks

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

        for epoch in range(epochs_trained, num_train_epochs):
            if isinstance(train_dataloader, DataLoader) and isinstance(train_dataloader.sampler, DistributedSampler):
                train_dataloader.sampler.set_epoch(epoch)
            elif isinstance(train_dataloader.dataset, IterableDatasetShard):
                train_dataloader.dataset.set_epoch(epoch)
            epoch_iterator = train_dataloader
            # Reset the past mems state at the beginning of each epoch if necessary.
            if args.past_index >= 0:
                self._past = None
            steps_in_epoch = (
                len(epoch_iterator) if train_dataset_is_sized else args.max_steps * args.gradient_accumulation_steps
            )
            self.control = self.callback_handler.on_epoch_begin(args, self.state, self.control)

            self.his_info = HisStatisticInfo(args)

            for step, inputs in enumerate(epoch_iterator):
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
                        tr_loss_step, ce_loss, greedy_reward = self.training_step(model, inputs)
                else:
                    tr_loss_step, ce_loss, greedy_reward = self.training_step(model, inputs)
                    for name, value in greedy_reward.items():
                        if name == 'ori_loss':
                            value = value.view(-1).clone().detach().cpu().numpy()
                        if name not in self.his_info.his_greedy_rewards:
                            self.his_info.his_greedy_rewards[name] = []
                        self.his_info.his_greedy_rewards[name].append(value)


                    self.his_info.his_ce_loss.append(ce_loss.clone().detach().cpu().numpy())
                    self.his_info.his_loss.append(tr_loss_step.clone().detach().cpu().numpy())


                if (
                    args.logging_nan_inf_filter
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
                output_dir = args.output_dir
                if (step + 1) % save_every == 0:
                    if not os.path.exists(output_dir + args.exp_name):
                        os.makedirs(output_dir + args.exp_name)
                    save_path = output_dir + args.exp_name + '/_model.{}_{}'.format(self.state.global_step, step)
                    logger.info('save to ' + save_path, 'epoch ', self.state.epoch)
                    if isinstance(model, torch.nn.DataParallel) or isinstance(model, torch.nn.parallel.DistributedDataParallel):
                        torch.save(model.module.state_dict(), save_path)
                    else:
                        torch.save(model.state_dict(), save_path)
                if (step + 1) % print_every == 0:
                    self.his_info.update()
                    self.his_info.print_info(tokenizer=self.rl_env.tokenizer)

                if ((step + 1) % args.gradient_accumulation_steps == 0 or (
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
                    elif self.use_amp:
                        scale_before = self.scaler.get_scale()
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        scale_after = self.scaler.get_scale()
                        optimizer_was_run = scale_before <= scale_after
                    else:

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

            self._maybe_log_save_evaluate(tr_loss, model, trial, epoch, ignore_keys_for_eval)

            if self.control.should_training_stop:
                break
        
        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of training
            delattr(self, "_past")
        
        logger.info("\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")
        if args.load_best_model_at_end and self.state.best_model_checkpoint is not None:
            # Wait for everyone to get here so we are sur the model has been saved by process 0.
            if args.local_rank != -1:
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
        if self.use_amp:
            with autocast():
                loss, outputs, rl_loss, greedy_reward = self.compute_rl_loss(model, inputs, return_outputs=True)
        else:
            loss, outputs, rl_loss, greedy_reward = self.compute_rl_loss(model, inputs, return_outputs=True)
        ce_loss = loss.clone()
        ce_losses = loss
        loss = self.args.rl_weight * rl_loss + ce_losses
        self.his_info.his_rl_loss.append(rl_loss.detach().clone().cpu().numpy())
        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training
        if self.args.gradient_accumulation_steps > 1 and not self.deepspeed:
            # deepspeed handles loss scaling by gradient_accumulation_steps in its `backward`
            loss = loss / self.args.gradient_accumulation_steps
        if self.use_amp:
            self.scaler.scale(loss).backward()
        elif self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        elif self.deepspeed:
            loss = self.deepspeed.backward(loss)
        else:
            loss.backward()
        return loss.detach() * self.args.gradient_accumulation_steps, ce_loss.detach(), greedy_reward

    def add_padding_(self, raw_txt, pad_id):
        txts_ids = [self.rl_env.tokenizer.encode(txt) for txt in raw_txt]
        for t in txts_ids:
            assert len(t) != 0
        padding_txts_ids = []
        batch_max_seq_len = max([len(txt) for txt in txts_ids])
        batch_max_seq_len = min(batch_max_seq_len, 142)
        for txt_ids in txts_ids:
            padding_txts_ids.append(
                txt_ids[:batch_max_seq_len] + [pad_id] * (batch_max_seq_len - len(txt_ids[:batch_max_seq_len])))
        return torch.tensor(padding_txts_ids, dtype=torch.long).cuda()

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


        '''
        Calculateing accuracy and loss while generating n-gram related positional information.
        '''
        def compute_acc_and_loss(max_ids, labels_, mp, log_probs_all):
            tot_cont = 0
            dif_cont = 0
            # acc max_ids -> labels
            # loss sum of log_probs
            masked_probs = log_probs_all.gather(2, max_ids.unsqueeze(2)).squeeze()
            pred_acc = max_ids == labels_
            batch_p_numpy = mp.clone().detach().cpu().numpy()
            batch_2_gram_pos = []
            batch_3_gram_pos = []
            batch_4_gram_pos = []
            batch_n_gram_pos = []
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
                self.his_info.his_2_gram_acc.append(_2_gram_acc.cpu())
                self.his_info.his_2_gram_loss.append(_2_gram_loss.cpu())
            if batch_3_gram_pos.sum() != 0:
                self.his_info.his_3_gram_acc.append(_3_gram_acc.cpu())
                self.his_info.his_3_gram_loss.append(_3_gram_loss.cpu())
            if batch_4_gram_pos.sum() != 0:
                self.his_info.his_4_gram_acc.append(_4_gram_acc.cpu())
                self.his_info.his_4_gram_loss.append(_4_gram_loss.cpu())
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
                self.his_info.his_dif_cont_rate.append(dif_cont / tot_cont)

        not_normal_log_probs = None
        # <#REFS#> is only for SQuAD
        if '<#REFS#>' in inputs['q2'][0]:
            # 'refs' may contain multi references in SQuAD
            refs = [e.split('<#REFS#>')[1] for e in inputs['q2']]
            inputs['q2'] = [e.split('<#REFS#>')[0] for e in inputs['q2']]
        else:
            refs = None

        '''
        choose the static data: data+/data-/groundtruth
        '''
        if '<#QUERY#>' in inputs['q2'][0]:
            tmp_q2 = [t.split('<#QUERY#>')[1] for t in inputs['q2']]
        else:
            tmp_q2 =  inputs['q2']
        assert self.args.cand_num == 1
        cands = [t for e in tmp_q2 for t in random.sample(e.split('<#SEP#>')[:], self.args.cand_num)]
        if self.args.cand_num == 1 and self.args.kd_inputs_best:
            cands = [t for e in tmp_q2 for t in random.sample(e.split('<#SEP#>')[:1], self.args.cand_num)]
        if self.args.cand_num == 1 and self.args.kd_inputs_worst:
            cands = [t for e in tmp_q2 for t in random.sample(e.split('<#SEP#>')[-1:], self.args.cand_num)]
        cands = [t.split('<#SCORE#>')[0] for t in cands]
        cands = self.add_padding_(cands, pad_id=-100)
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
        if self.args.not_replace_kd:
            second_kd_inputs_labels = kd_inputs['labels']
        else:
            kd_inputs['labels'] = kd_seq_labels
            second_kd_inputs_labels = kd_inputs['labels']


        if self.args.seq_decode_model == 't5':
            model.scoring_mode()
        elif self.args.seq_decode_model == 'bart':
            model.model.scoring_mode()
        elif self.args.seq_decode_model == 'pegasus':
            model.model.scoring_mode()

        # stack inputs for computing mle (g_res) and rl (s_res)
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

        batch_size = second_kd_inputs_labels.shape[0]
        second_kd_inputs_labels = second_kd_inputs_labels.reshape(batch_size, -1)
        model.train()

        try:
            start = time.time()
            merge_model_return_dict = model(**kd_inputs)
        except ValueError as e:
            print(e)
            return (torch.tensor(0).cuda(), None, 0, {'error':1})

        model_return_dict, kd_model_return_dict = merge_model_return_dict
        outputs, y_b, y_s, max_ids, masked_ids, \
            input_ids, labels_, non_zero_sum_tensor, \
            log_probs, y_zero_b, y_zero_s, y_zero_labels, \
            truth_log_probs, log_probs_all, lm_logits = model_return_dict[
                                                                                                                                                                                       :15]
        compute_acc_and_loss(
            max_ids, labels_, model_return_dict[17], log_probs_all)

        model.train()
        _, y_b, y_s, max_ids, masked_ids, input_ids, _, \
            non_zero_sum_tensor, log_probs, y_zero_b, \
            y_zero_s, y_zero_labels, \
            truth_log_probs, log_probs_all, _ = kd_model_return_dict[:15]

        mask_labels, masked_pos_shift, masked_pos_non_shift, decoder_input_ids = kd_model_return_dict[15:19]
        compute_dif(y_b, masked_pos_non_shift)

        length_penalty = self.args.training_length_penalty

        if (model.config.sample_num != 0 or self.args.cand_num !=1):
            cand_mask = ~log_probs.data.eq(0)
            '''
            mask the corresponding logprob if the sampled token is a special token
            '''
            if self.args.seq_decode_model == 'bart':
                new_cand_mask = cand_mask & ~(
                        (y_s != eos_token_id) & (y_s != 0) & (
                        y_s != pad_token_id))
                cand_mask = cand_mask & (y_s != eos_token_id) & (
                        y_s != 0) & (y_s != pad_token_id)
                batch_size = labels_.shape[0]
            elif self.args.seq_decode_model == 't5' and self.model.config.tokenizer_name == 't5-small':
                new_cand_mask = cand_mask & ~((y_s != eos_token_id) & (y_s != pad_token_id))
                cand_mask = cand_mask  & (y_s != eos_token_id) & (y_s != pad_token_id)
            elif self.args.seq_decode_model == 'pegasus':
                new_cand_mask = cand_mask & ~((y_s != eos_token_id) & (y_s != pad_token_id))
                cand_mask = cand_mask & (y_s != eos_token_id) & (y_s != pad_token_id)

            '''
            mask the same position for all samples if the sampled token is a special token
            '''
            cand_mask = cand_mask.reshape(batch_size, -1, cand_mask.shape[-1])
            cand_num = cand_mask.shape[1]
            cand_mask = cand_mask.sum(1)
            cand_mask = cand_mask.data.eq(cand_num)
            cand_mask = cand_mask.unsqueeze(1).repeat(1, cand_num, 1)
            cand_mask = cand_mask.reshape(-1, cand_mask.shape[-1])

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
        rl_loss = 0
        if self.model.config.do_rl:
            np.random.seed(self.state.global_step)
            rl_loss, \
                greedy_reward_tensor, \
                all_gen_traces_tk, \
                all_gen_b_traces_tk, \
                all_reward = \
                self.rl_env.rl_step(y_b,
                                    y_s,
                                    max_ids,
                                    masked_ids,
                                    input_ids,
                                    labels_,
                                    non_zero_sum_tensor,
                                    log_probs,
                                    q2,
                                    not_normal_log_probs=not_normal_log_probs,
                                    raw_src=inputs['query'],
                                    refs=refs,
                       )
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
                if tk not in self.his_info.reward_dist_dict:
                    self.his_info.reward_dist_dict[tk] = [reward_dist_dict[tk]]
                else:
                    self.his_info.reward_dist_dict[tk].append(reward_dist_dict[tk])

            greedy_reward_dict = {}
            greedy_reward_tensor = greedy_reward_tensor.transpose(0, 1)
            for i, name in enumerate(self.args.rewards.split(',')):
                greedy_reward_dict[name] = greedy_reward_tensor[i].float().mean().detach().cpu().numpy()
                if name not in self.his_info.his_reward_dict:
                    self.his_info.his_reward_dict[name] = []
                self.his_info.his_reward_dict[name].append(greedy_reward_dict[name])
                self.his_info.his_reward_dict[name] = self.his_info.his_reward_dict[name][-2000:]
            if self.args.past_index >= 0:
                self._past = outputs[self.args.past_index]

        if self.args.seq_decode_model == 't5':
            model.generation_mode()
        elif self.args.seq_decode_model == 'bart':
            model.model.generation_mode()
        elif self.args.seq_decode_model == 'pegasus':
            model.model.generation_mode()

        if self.smoother is not None and labels_ is not None:
            loss = self.smoother(outputs, labels_)
        else:
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        return (loss, outputs, rl_loss, greedy_reward_dict) if return_outputs else loss


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

            raw_src_origin = raw_src
            raw_src = [e.split()[0] for e in raw_src]
            # Prediction step
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
            # self.metrs[_m] = getattr(Metrics,_m)(model_name=model_name, cfg=cfgs, customize_cfg=task_cfgs)
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
