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
from torch.utils.data.dataset import Dataset, IterableDataset
from transformers.file_utils import is_datasets_available
from transformers.trainer_pt_utils import IterableDatasetShard, LabelSmoother
from transformers.trainer_utils import TrainOutput
from data.tokenizer_utils import *
from torch import nn
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
import math 
import torch.distributed as dist
import matplotlib.pyplot as plt 
from trainer.Trainer import register_trainer

if is_datasets_available():
    import datasets
logger = logging.get_logger(__name__)

from transformers.integrations import AzureMLCallback

@register_trainer("common")
class Trainer(Trainer):
    def __init__(self, model, args, model_args, task_args, train_dataset, eval_dataset, auto_tokenizer):
        data_collator = CustomizeCollator(train_dataset,eval_dataset)
        #auto_tokenizer = prepare_tokenizer(model_args._name_or_path, args.cache_dir, special_tokens=args.special_tokens)
        # resize embedding, will do nothing if `old_num_tokens==new_num_tokens`
        model.resize_token_embeddings(len(auto_tokenizer))
        print('2')

        if args.do_train:
            if args.eval_metrics == "eval_loss":
                metrics_calculator = None
                args.metric_for_best_model = "eval_loss"
            else:
                #metrics_calculator = MetricsCalculator(args.eval_metrics, model_args._name_or_path, args, task_args) if args.eval_metrics else None
                metrics_calculator = MetricsCalculator(args.eval_metrics, model_args._name_or_path, args, task_args,
                                                       auto_tokenizer) if args.eval_metrics else None

                args.metric_for_best_model = MetricsCalculator.cvt_mtc(args.eval_metrics.split(",")[0], False)
        else:
            metrics_calculator = None
        if args.do_predict:
            print(args.result_header) #query
            self.result_header = args.result_header.split(
                ",") if "," in args.result_header else eval_dataset.feature_extractor.model_header
            print(self.result_header) #query:doc:
            #assert 1 == 0
            #self.outputter = getattr(Outputter,args.output_type)(args,model_args._name_or_path)
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

        super().__init__(
            model = model,
            args = args,
            train_dataset = train_dataset.get_dataset() if train_dataset else None,
            eval_dataset = eval_dataset.get_dataset() if eval_dataset else None,
            data_collator = data_collator,
            compute_metrics = metrics_calculator
            )
        self.train_start_time = time.time()
        self.model.tokenizer = auto_tokenizer
        az_logger = AmlLogger()
        if az_logger.active:
            self.add_callback(AmlLogger)
    
    @staticmethod
    def _get_aml_logger():
        """
        If azureml.core package is installed, return an AML logger; otherwise return a dummy logger that does nothing
        """
        try:
            from azureml.core.run import Run
            run = Run.get_context(allow_offline=False)

            return run.log
        except:
            def dummy_logger(name, value):
                pass
            return dummy_logger

    def num_examples(self,dataloader):
        if hasattr(dataloader.dataset,"num_examples"):
            return dataloader.dataset.num_examples
        else:
            return len(dataloader.dataset)

    def realtime_prepare_model(self):
        self.model = self._wrap_model(self.model, training=False)
        self.model.eval()

    def realtime_predict(self,features,outputter):
        loss, logits, labels = self.prediction_step(self.model,features,False)
        return outputter.realtime_output(features,logits)

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
                #print(inputs.keys())
                #assert 1==0
                batch = {}
                # keep batch features and drop the corresponding ones in inputs
                for fn in self.result_header:
                    if fn in inputs:
                        batch[fn] = inputs.pop(fn)
                #print(batch)

                loss, logits, labels = self.prediction_step(model, inputs,False)
                self.outputter(batch,logits)
                # print(labels.shape)
                # print(logits.shape)
                # print(res)
                if self.args.logging_steps > 0 and step % self.args.logging_steps == 0 and step > 0:
                    current_speed = speed_metrics("inference", start_time, step*self.args.per_device_eval_batch_size * self.args.world_size)
                    current_speed["job_progress"] = step*self.args.per_device_eval_batch_size * self.args.world_size / num_examples
                    self.log(current_speed)

                #profiler.step()
                #if step == 10:
                #    print(profiler.key_averages())
        self.log(speed_metrics("inference", start_time, num_examples))
        #self.outputter.writer_process()
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

        metrics = None
        if self.control.should_evaluate:
            self.model.config.decoding_method = 'seq'
            metrics = self.evaluate()
            num_samples = self.args.per_device_train_batch_size * self.args.world_size * self.state.global_step
            metrics["eval_num_samples_consumed"] = num_samples
            self._report_to_hp_search(trial, epoch, metrics)

            if self.args.eval_non_seq:
                self.model.config.decoding_method = 'non_seq'
                metrics = self.evaluate()
                num_samples = self.args.per_device_train_batch_size * self.args.world_size * self.state.global_step
                metrics["eval_num_samples_consumed"] = num_samples
                self._report_to_hp_search(trial, epoch, metrics)


        # if self.control.should_save:
        #     self._save_checkpoint(model, trial, metrics=metrics)
        #     self.control = self.callback_handler.on_save(self.args, self.state, self.control)


class MetricsCalculator:
    def __init__(self, metrics, model_name, cfgs, task_cfgs, tokenizer):
        self.metrs = {}
        for m in metrics.split(","):
            _m = self.cvt_mtc(m, True)
            self.metrs[_m] = getattr(Metrics,_m)(model_name=model_name, cfg=cfgs, customize_cfg=task_cfgs)
            self.metrs[_m] = getattr(Metrics, _m)(model_name=model_name, cfg=cfgs, customize_cfg=task_cfgs, tokenizer=tokenizer)

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
