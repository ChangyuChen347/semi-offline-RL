from typing import DefaultDict
from collections import defaultdict
from . import register_metrics
from transformers import AutoTokenizer
import nltk
from rouge import Rouge as RougeMetrics # use py-rouge package
from rouge_score import rouge_scorer
from lmqg.automatic_evaluation_tool import text_normalization
import numpy as np
from transformers import logging
import scipy
import datasets
from nltk.util import ngrams
from functools import reduce
import operator
import pickle as pkl
logger = logging.get_logger(__name__)
logger.setLevel('INFO')
@register_metrics("bleu")
class Bleu():
    def __init__(self, model_name, cfg, **kwargs):
        tokenizer = kwargs.pop("tokenizer", None)
        self.tokenizer = tokenizer if tokenizer is not None else AutoTokenizer.from_pretrained(model_name)
        self.cfg = cfg
        self.print_instance = 10
        
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')

    def __call__(self,EvalPredict):
        if EvalPredict.predictions.ndim == 3:
            res = EvalPredict.predictions
        else:
            res = np.expand_dims(EvalPredict.predictions,axis=-2)
        bleu = []
        bleu1=[]
        bleu2 = []
        bleu3 = []
        bleu4 = []
        length = []
        all_pred = []
        for i,label in enumerate(EvalPredict.label_ids):
            label = np.clip(label,0,None)
            truth = self.tokenizer.decode(label,skip_special_tokens=True,clean_up_tokenization_spaces=False)
            # reference text (groundtruth) could be empty for DisplayURL scenario
            # in this case, consider empty string as one token
            if len(truth) == 0:
                truth = "EOS"
            pred = []
            for r in res[i]:
                curr_pred = self.tokenizer.decode(r,skip_special_tokens=True,clean_up_tokenization_spaces=False)
                if len(curr_pred) == 0:
                    curr_pred = "EOS"

                pred.append(curr_pred)
                all_pred.append(curr_pred)

            bleu.append(max([nltk.translate.bleu_score.sentence_bleu([nltk.word_tokenize(truth.lower())],
                                                                     nltk.word_tokenize(p.lower()),
                                                                     weights=[0.25, 0.25, 0.25, 0.25],
                                                                     smoothing_function=nltk.translate.bleu_score.SmoothingFunction().method1)
                             for p in pred]))

            bleu1.append(max([nltk.translate.bleu_score.sentence_bleu([nltk.word_tokenize(truth.lower())],
                                                                     nltk.word_tokenize(p.lower()),
                                                                     smoothing_function=nltk.translate.bleu_score.SmoothingFunction().method1,
                                                                     weights=[1, 0, 0, 0])
                             for p in pred]))



            bleu2.append(max([nltk.translate.bleu_score.sentence_bleu([nltk.word_tokenize(truth.lower())],
                                                                     nltk.word_tokenize(p.lower()),
                                                                     smoothing_function=nltk.translate.bleu_score.SmoothingFunction().method1,weights=[0,1,0,0])
                             for p in pred]))

            bleu3.append(max([nltk.translate.bleu_score.sentence_bleu([nltk.word_tokenize(truth.lower())],
                                                                     nltk.word_tokenize(p.lower()),
                                                                     smoothing_function=nltk.translate.bleu_score.SmoothingFunction().method1,
                                                                     weights=[0, 0, 1, 0])
                             for p in pred]))

            bleu4.append(max([nltk.translate.bleu_score.sentence_bleu([nltk.word_tokenize(truth.lower())],
                                                                     nltk.word_tokenize(p.lower()),
                                                                     smoothing_function=nltk.translate.bleu_score.SmoothingFunction().method1,
                                                                     weights=[0, 0, 0, 1])
                             for p in pred]))

            for t in pred:
                length.append(len(t.split()))

        scores = {'bleu': np.mean(bleu), 'b1': np.mean(bleu1), 'b2': np.mean(bleu2), 'b3': np.mean(bleu3), 'b4': np.mean(bleu4), 'length': np.mean(length)}

from collections import Counter
from nltk.translate import bleu_score
from nltk.translate.bleu_score import SmoothingFunction
from lmqg.automatic_evaluation_tool.bleu.bleu import Bleu as bleu_squad
from lmqg.automatic_evaluation_tool.rouge import Rouge as rouge_squad
bleu_squad = bleu_squad()
rouge_squad = rouge_squad()


@register_metrics("squad_bleu")
class SQUAD_BLEU():
    def __init__(self, model_name, cfg, **kwargs):
        tokenizer = kwargs.pop("tokenizer", None)
        self.tokenizer = tokenizer if tokenizer is not None else AutoTokenizer.from_pretrained(model_name)
        self.cfg = cfg

        if self.cfg.eval_dir == 'static_data/squad/squad_valid.tsv':
            self.src = open('sentence-valid.txt').readlines()
            self.tgt = open('question-valid.txt').readlines()
        elif self.cfg.eval_dir == 'static_data/squad/squad_test.tsv':
            self.src = open('sentence-test.txt').readlines()
            self.tgt = open('question-test.txt').readlines()
        elif self.cfg.eval_dir == 'static_data/squad/squad_test.5.tsv':
            self.src = open('sentence-test.txt').readlines()
            self.tgt = open('question-test.txt').readlines()
        else:
            raise NotImplementedError("the path for the eval_dir not implemented yet.")

        self.print_instance = 10
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
    def __call__(self, EvalPredict):
        if EvalPredict.predictions.ndim == 3:
            res = EvalPredict.predictions
        else:
            res = np.expand_dims(EvalPredict.predictions, axis=-2)
        bleu = []
        bleu1 = []
        bleu2 = []
        bleu3 = []
        bleu4 = []
        length = []
        all_pred = []

        to_eval_hyp_dict = {}
        to_eval_ref_dict = {}
        for i, label in enumerate(EvalPredict.label_ids):
            label = np.clip(label, 0, None)
            truth = self.tokenizer.decode(label, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            # reference text (groundtruth) could be empty for DisplayURL scenario
            # in this case, consider empty string as one token
            if len(truth) == 0:
                truth = "EOS"
            pred = []
            for r in res[i]:
                curr_pred = self.tokenizer.decode(r, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                if len(curr_pred) == 0:
                    curr_pred = "EOS"
                # curr_pred = ' '.join(nltk.word_tokenize(curr_pred))
                pred.append(curr_pred)
                all_pred.append(curr_pred)
            if self.src[i] in to_eval_hyp_dict:
                to_eval_ref_dict[self.src[i]].append(self.tgt[i].strip().lower().encode('utf-8'))
            else:
                to_eval_ref_dict[self.src[i]] = [self.tgt[i].strip().lower().encode('utf-8')]
                to_eval_hyp_dict[self.src[i]] = [curr_pred.strip().lower().encode('utf-8')]
            for t in pred:
                length.append(len(t.split()))

        bleu_score, _ = bleu_squad.compute_score(to_eval_ref_dict, to_eval_hyp_dict)
        return bleu_score


@register_metrics("squad_rouge")
class SQUAD_ROUGE():
    def __init__(self, model_name, cfg, **kwargs):
        tokenizer = kwargs.pop("tokenizer", None)
        self.tokenizer = tokenizer if tokenizer is not None else AutoTokenizer.from_pretrained(model_name)
        self.cfg = cfg

        if self.cfg.eval_dir == 'static_data/squad/squad_valid.tsv':
            self.src = open('sentence-valid.txt').readlines()
            self.tgt = open('question-valid.txt').readlines()
        elif self.cfg.eval_dir == 'static_data/squad/squad_test.tsv':
            self.src = open('sentence-test.txt').readlines()
            self.tgt = open('question-test.txt').readlines()
        elif self.cfg.eval_dir == 'static_data/squad/squad_test.5.tsv':
            self.src = open('sentence-test.txt').readlines()
            self.tgt = open('question-test.txt').readlines()
        else:
            raise NotImplementedError("the path for the eval_dir not implemented yet.")

        self.print_instance = 10

        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')


    def __call__(self, EvalPredict):
        if EvalPredict.predictions.ndim == 3:
            res = EvalPredict.predictions
        else:
            res = np.expand_dims(EvalPredict.predictions, axis=-2)
        scores = []
        length = []
        all_pred = []
        src = open('sentence-test.txt').readlines()
        tgt = open('question-test.txt').readlines()
        to_eval_hyp_dict = {}
        to_eval_ref_dict = {}
        for i, label in enumerate(EvalPredict.label_ids):
            label = np.clip(label, 0, None)
            truth = self.tokenizer.decode(label, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            if len(truth) == 0:
                truth = "EOS"
            pred = []
            for r in res[i]:
                curr_pred = self.tokenizer.decode(r, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                if len(curr_pred) == 0:
                    curr_pred = "EOS"
                # curr_pred = ' '.join(nltk.word_tokenize(curr_pred))
                pred.append(curr_pred)
                all_pred.append(curr_pred)
            if self.src[i] in to_eval_hyp_dict:
                to_eval_ref_dict[self.src[i]].append(self.tgt[i].strip().lower().encode('utf-8'))
            else:
                to_eval_ref_dict[self.src[i]] = [self.tgt[i].strip().lower().encode('utf-8')]
                to_eval_hyp_dict[self.src[i]] = [curr_pred.strip().lower().encode('utf-8')]
            for t in pred:
                length.append(len(t.split()))
        import pickle as pkl
        pkl.dump([to_eval_ref_dict, to_eval_hyp_dict], open('to_eval_pkl', 'wb'))
        score, _ = rouge_squad.compute_score(to_eval_ref_dict, to_eval_hyp_dict)
        return score



@register_metrics("rouge")
class Rouge():
    def __init__(self, model_name, cfg, which_rouge=None, max_n=1, **kwargs):
        """
        Calculate Rouge F1 metrics

        which_rouge: str or list of str, for example "rouge-n", "rouge-l", "rouge-w"
        max_n: N-grams for ROUGE-N if specify. Default:1
        """
        self.cfg = cfg

        self.print_instance = 10

        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        tokenizer = kwargs.pop("tokenizer", None)
        self.tokenizer = tokenizer if tokenizer is not None else AutoTokenizer.from_pretrained(model_name)
        if which_rouge is None:
            self.rouge = RougeMetrics(metrics=['rouge-n', 'rouge-l'], max_n=2)
        else:
            if isinstance(which_rouge, str):
                self.rouge = RougeMetrics(metrics=[which_rouge], max_n=max_n)
            else:
                self.rouge = RougeMetrics(metrics=which_rouge, max_n=max_n)

    def __call__(self,EvalPredict):
        if EvalPredict.predictions.ndim == 3:
            res = EvalPredict.predictions
        else:
            res = np.expand_dims(EvalPredict.predictions,axis=-2)

        rouge_list = defaultdict(list)
        debug_pkl = []
        preds = []
        truths = []
        for i,label in enumerate(EvalPredict.label_ids):
            label = np.clip(label,0,None)
            truth = self.tokenizer.decode(label,skip_special_tokens=True,clean_up_tokenization_spaces=False)
            # reference text (groundtruth) could be empty for DisplayURL scenario
            # in this case, consider empty string as one token
            if len(truth) == 0:
                truth = "EOS"
            truths.append(truth)
            tmp_rouge = defaultdict(list)
            pred = []
            # calculate the maximum rouge from multiple beam paths
            for r in res[i]:
                curr_pred = self.tokenizer.decode(r,skip_special_tokens=True,clean_up_tokenization_spaces=False)
                curr_tokens = self.tokenizer.convert_ids_to_tokens(r)

                if len(curr_pred) == 0 :
                    curr_pred = "EOS"
                pred.append(curr_pred)
                preds.append(curr_pred)
                curr_pred = ' '.join(nltk.word_tokenize(curr_pred))
                truth = ' '.join(nltk.word_tokenize(truth))
                rouge_score = self.rouge.get_scores(curr_pred.lower(), truth.lower())

                for k in rouge_score:
                    tmp_rouge[k].append(rouge_score[k]["f"])
            for k in tmp_rouge:
                rouge_list[k].append(max(tmp_rouge[k]))

            if i < self.print_instance and self.cfg.local_rank <= 0:
                logger.info("Truth: "+truth+" Predict: "+" ### ".join(pred))


        scores = dict()
        for k in rouge_list:
            scores[k] = np.mean(rouge_list[k])

        length = []
        for t in preds:
            length.append(len(t.split()))
        scores['length'] = np.mean(length)
        length = []
        for t in truths:
            length.append(len(t.split()))
        scores['gt_length'] = np.mean(length)
        if len(scores) == 1:
            return list(scores.values())[0]
        else:
            return scores


@register_metrics("rouges")
class Rouges():
    def __init__(self, model_name, cfg, which_rouge=None, max_n=1, **kwargs):
        """
        Calculate Rouge F1 metrics

        which_rouge: str or list of str, for example "rouge-n", "rouge-l", "rouge-w"
        max_n: N-grams for ROUGE-N if specify. Default:1
        """
        self.cfg = cfg
        self.print_instance = 10
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        tokenizer = kwargs.pop("tokenizer", None)
        self.tokenizer = tokenizer if tokenizer is not None else AutoTokenizer.from_pretrained(model_name)
        self.rouge = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL', 'rougeLsum'], use_stemmer=True)

    def __call__(self, EvalPredict):
        if EvalPredict.predictions.ndim == 3:
            res = EvalPredict.predictions
        else:
            res = np.expand_dims(EvalPredict.predictions, axis=-2)
        lang = EvalPredict.predictions
        rouge_list = defaultdict(list)
        debug_pkl = []
        preds = []
        cont_tot = 0
        heshe_tot = 0
        they_tot = 0
        for i, label in enumerate(EvalPredict.label_ids):
            label = np.clip(label, 0, None)
            truth = self.tokenizer.decode(label, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            # reference text (groundtruth) could be empty for DisplayURL scenario
            # in this case, consider empty string as one token
            if len(truth) == 0:
                truth = "EOS"
            tmp_rouge = defaultdict(list)
            pred = []
            # calculate the maximum rouge from multiple beam paths
            for r in res[i]:
                curr_pred = self.tokenizer.decode(r, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                if len(curr_pred) == 0:
                    curr_pred = "EOS"
                pred.append(curr_pred)
                curr_pred = "\n".join(nltk.sent_tokenize(" ".join(nltk.word_tokenize(curr_pred))))
                preds.append(curr_pred)
                curr_pred = "\n".join(nltk.sent_tokenize(curr_pred))
                truth = "\n".join(nltk.sent_tokenize(" ".join(nltk.word_tokenize(truth))))
                res_rouge = self.rouge.score(curr_pred.lower(), truth.lower())
                for k in res_rouge:
                    tmp_rouge[k].append(res_rouge[k].fmeasure)
            for k in tmp_rouge:
                rouge_list[k].append(max(tmp_rouge[k]))
            if i < self.print_instance and self.cfg.local_rank <= 0:
                print("Truth: " + truth + " Predict: " + " ### ".join(pred))
        scores = dict()
        for k in rouge_list:
            scores[k] = np.mean(rouge_list[k])
        length = []
        for t in preds:
            length.append(len(t.split()))
        scores['length'] = np.mean(length)
        if len(scores) == 1:
            return list(scores.values())[0]
        else:
            return scores


