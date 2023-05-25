from typing import DefaultDict
from collections import defaultdict
from . import register_metrics
from transformers import AutoTokenizer
import nltk
from rouge import Rouge as RougeMetrics # use py-rouge package
from rouge_score import rouge_scorer
from automatic_evaluation_tool import text_normalization
import numpy as np
from transformers import logging
import scipy
logger = logging.get_logger(__name__)

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
                # curr_pred = ' '.join(nltk.word_tokenize(curr_pred))
                pred.append(curr_pred)
                all_pred.append(curr_pred)
           # bleu.append(max([nltk.translate.bleu_score.sentence_bleu([nltk.word_tokenize(truth.lower())],nltk.word_tokenize(p.lower()),smoothing_function=nltk.translate.bleu_score.SmoothingFunction().method1) for p in pred]))
           #  truth = ' '.join(nltk.word_tokenize(truth))
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
            #if i < self.print_instance and self.cfg.local_rank <= 0:
            #    logger.info("Truth: "+truth+" Predict: "+" ### ".join(pred))

            for t in pred:
                length.append(len(t.split()))

        to_write_file = open('squad_test', 'w')
        to_write_file.write('\n'.join(all_pred))
        assert 1==0
        return [np.mean(bleu), np.mean(bleu1), np.mean(bleu2), np.mean(bleu3), np.mean(bleu4), np.mean(length)]

from collections import Counter

from nltk.translate import bleu_score
from nltk.translate.bleu_score import SmoothingFunction
@register_metrics("chat_bleu")
class ChatBleu():
    def __init__(self, model_name, cfg, **kwargs):
        tokenizer = kwargs.pop("tokenizer", None)
        self.tokenizer = tokenizer if tokenizer is not None else AutoTokenizer.from_pretrained(model_name)
        self.cfg = cfg
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
        refs = []
        for i, label in enumerate(EvalPredict.label_ids):
            label = np.clip(label, 0, None)
            truth = self.tokenizer.decode(label, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            # reference text (groundtruth) could be empty for DisplayURL scenario
            # in this case, consider empty string as one token
            if len(truth) == 0:
                truth = "EOS"
            refs.append(truth)
            pred = []
            for r in res[i]:
                curr_pred = self.tokenizer.decode(r, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                if len(curr_pred) == 0:
                    curr_pred = "EOS"
                # curr_pred = ' '.join(nltk.word_tokenize(curr_pred))
                pred.append(curr_pred)
                all_pred.append(curr_pred)

            for t in pred:
                length.append(len(t.split()))

        # to_write_file = open('squad_test', 'w')
        # to_write_file.write('\n'.join(all_pred))
        # assert 1==0
        def distinct(seqs):
            """ Calculate intra/inter distinct 1/2. """
            batch_size = len(seqs)
            intra_dist1, intra_dist2 = [], []
            unigrams_all, bigrams_all = Counter(), Counter()
            for seq in seqs:
                unigrams = Counter(seq)
                bigrams = Counter(zip(seq, seq[1:]))
                intra_dist1.append((len(unigrams) + 1e-12) / (len(seq) + 1e-5))
                intra_dist2.append((len(bigrams) + 1e-12) / (max(0, len(seq) - 1) + 1e-5))

                unigrams_all.update(unigrams)
                bigrams_all.update(bigrams)

            inter_dist1 = (len(unigrams_all) + 1e-12) / (sum(unigrams_all.values()) + 1e-5)
            inter_dist2 = (len(bigrams_all) + 1e-12) / (sum(bigrams_all.values()) + 1e-5)
            intra_dist1 = np.average(intra_dist1)
            intra_dist2 = np.average(intra_dist2)
            return intra_dist1, intra_dist2, inter_dist1, inter_dist2


        def bleu(hyps, refs):
            """ Calculate bleu 1/2. """
            bleu_1 = []
            bleu_2 = []
            for hyp, ref in zip(hyps, refs):
                try:
                    score = bleu_score.sentence_bleu(
                        [ref], hyp,
                        smoothing_function=SmoothingFunction().method7,
                        weights=[1, 0, 0, 0])
                except:
                    score = 0
                bleu_1.append(score)
                try:
                    score = bleu_score.sentence_bleu(
                        [ref], hyp,
                        smoothing_function=SmoothingFunction().method7,
                        weights=[0.5, 0.5, 0, 0])
                except:
                    score = 0
                bleu_2.append(score)
            bleu_1 = np.average(bleu_1)
            bleu_2 = np.average(bleu_2)
            return bleu_1, bleu_2

        all_pred = [line.strip().split(" ") for line in all_pred]
        refs = [line.strip().split(" ") for line in refs]
        bleu1, bleu2 = bleu(all_pred, refs)
        intra_dist1, intra_dist2, inter_dist1, inter_dist2 = distinct(all_pred)
        return [bleu1 * 100., bleu2 * 100., inter_dist1, inter_dist2, np.mean(length)]



from automatic_evaluation_tool.bleu.bleu import Bleu as bleu_squad
bleu_squad = bleu_squad()
from automatic_evaluation_tool.rouge import Rouge as rouge_squad
rouge_squad = rouge_squad()


@register_metrics("squad_bleu_s")
class SQUAD_BLEU_S():
    def __init__(self, model_name, cfg, **kwargs):
        tokenizer = kwargs.pop("tokenizer", None)
        self.tokenizer = tokenizer if tokenizer is not None else AutoTokenizer.from_pretrained(model_name)
        self.cfg = cfg
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
        length = []
        all_pred = []
        src = open('sentence-test.txt').readlines()
        tgt = open('question-test.txt').readlines()
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

            to_eval_ref_dict[i] = [tgt[i].strip().lower().encode('utf-8')]
            to_eval_hyp_dict[i] = [curr_pred.strip().lower().encode('utf-8')]
            for t in pred:
                length.append(len(t.split()))

        # to_write_file = open('squad_test', 'w')
        # to_write_file.write('\n'.join(all_pred))
        # assert 1 == 0
        bleu_score, _ = bleu_squad.compute_score(to_eval_ref_dict, to_eval_hyp_dict)
        return bleu_score
@register_metrics("squad_bleu")
class SQUAD_BLEU():
    def __init__(self, model_name, cfg, **kwargs):
        tokenizer = kwargs.pop("tokenizer", None)
        self.tokenizer = tokenizer if tokenizer is not None else AutoTokenizer.from_pretrained(model_name)
        self.cfg = cfg
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
        src = open('sentence-test.txt').readlines()
        tgt = open('question-test.txt').readlines()
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
            if src[i] in to_eval_hyp_dict:
                to_eval_ref_dict[src[i]].append(tgt[i].strip().lower().encode('utf-8'))
            else:
                to_eval_ref_dict[src[i]] = [tgt[i].strip().lower().encode('utf-8')]
                to_eval_hyp_dict[src[i]] = [curr_pred.strip().lower().encode('utf-8')]
            for t in pred:
                length.append(len(t.split()))

        # to_write_file = open('squad_test', 'w')
        # to_write_file.write('\n'.join(all_pred))
        # assert 1 == 0
        bleu_score, _ = bleu_squad.compute_score(to_eval_ref_dict, to_eval_hyp_dict)
        return bleu_score

@register_metrics("squad_meteor")
class SQUAD_METEOR():
    def __init__(self, model_name, cfg, **kwargs):
        from automatic_evaluation_tool.meteor.meteor import Meteor as meteor_squad
        self.meteor_squad = meteor_squad()
        tokenizer = kwargs.pop("tokenizer", None)
        self.tokenizer = tokenizer if tokenizer is not None else AutoTokenizer.from_pretrained(model_name)
        self.cfg = cfg
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
        src = open('sentence-test.txt').readlines()
        tgt = open('question-test.txt').readlines()
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
            if src[i] in to_eval_hyp_dict:
                to_eval_ref_dict[src[i]].append(tgt[i].strip().lower().encode('utf-8'))
            else:
                to_eval_ref_dict[src[i]] = [tgt[i].strip().lower().encode('utf-8')]
                to_eval_hyp_dict[src[i]] = [curr_pred.strip().lower().encode('utf-8')]
            for t in pred:
                length.append(len(t.split()))

        # to_write_file = open('squad_test', 'w')
        # to_write_file.write('\n'.join(all_pred))
        # assert 1 == 0
        bleu_score, _ = self.meteor_squad.compute_score(to_eval_ref_dict, to_eval_hyp_dict)
        return bleu_score

@register_metrics("squad_rouge")
class SQUAD_ROUGE():
    def __init__(self, model_name, cfg, **kwargs):
        tokenizer = kwargs.pop("tokenizer", None)
        self.tokenizer = tokenizer if tokenizer is not None else AutoTokenizer.from_pretrained(model_name)
        self.cfg = cfg
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
            if src[i] in to_eval_hyp_dict:
                to_eval_ref_dict[src[i]].append(tgt[i].strip().lower().encode('utf-8'))
            else:
                to_eval_ref_dict[src[i]] = [tgt[i].strip().lower().encode('utf-8')]
                to_eval_hyp_dict[src[i]] = [curr_pred.strip().lower().encode('utf-8')]
            # _, rouge_scores = rouge_squad.compute_score(to_eval_ref_dict, to_eval_hyp_dict)
            for t in pred:
                length.append(len(t.split()))
        # to_write_file = open('squad_test', 'w')
        # to_write_file.write('\n'.join(all_pred))
        # assert 1 == 0
        import pickle as pkl
        pkl.dump([to_eval_ref_dict, to_eval_hyp_dict], open('to_eval_pkl', 'wb'))
        score, _ = rouge_squad.compute_score(to_eval_ref_dict, to_eval_hyp_dict)
        print(score)
        return score

@register_metrics("squad_rouge_s")
class SQUAD_ROUGE_S():
    def __init__(self, model_name, cfg, **kwargs):
        tokenizer = kwargs.pop("tokenizer", None)
        self.tokenizer = tokenizer if tokenizer is not None else AutoTokenizer.from_pretrained(model_name)
        self.cfg = cfg
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
        truth = open
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
            to_eval_ref_dict[i] = [tgt[i].strip().lower().encode('utf-8')]
            to_eval_hyp_dict[i] = [curr_pred.strip().lower().encode('utf-8')]
            # _, rouge_scores = rouge_squad.compute_score(to_eval_ref_dict, to_eval_hyp_dict)


            # print(to_eval_ref_dict[0])
            # print(to_eval_hyp_dict[0])
            # print(score)
            # assert 1==0
            for t in pred:
                length.append(len(t.split()))
        # to_write_file = open('squad_test', 'w')
        # to_write_file.write('\n'.join(all_pred))
        # assert 1 == 0
        score, _ = rouge_squad.compute_score(to_eval_ref_dict, to_eval_hyp_dict)
        return score

import datasets
@register_metrics("scarebleu")
class scarebleu():
    def __init__(self, model_name, cfg, **kwargs):
        # if model_name == 'Yale-LILY/brio-xsum-cased':
        #     model_name = 'google/pegasus-xsum'
        # if model_name == 'Yale-LILY/brio-cnndm-uncased':
        #     model_name = 'facebook/bart-large-cnn'
        tokenizer = kwargs.pop("tokenizer", None)
        self.tokenizer = tokenizer if tokenizer is not None else AutoTokenizer.from_pretrained(model_name)
        self.cfg = cfg
        self.print_instance = 10

        self.sacrebleu = datasets.load_metric('sacrebleu')
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
        all_preds = []
        all_refs = []
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
                pred.append(curr_pred)
                all_preds.append(curr_pred)
                all_refs.append(truth)
            if i < self.print_instance and self.cfg.local_rank <= 0:
                logger.info("Truth: " + truth + " Predict: " + " ### ".join(pred))
        results = self.sacrebleu.compute(predictions=all_preds, references=[[t] for t in all_refs])
        return results['score']


from nltk.translate.meteor_score import meteor_score
@register_metrics("meteor")
class meteor():
    def __init__(self, model_name, cfg, **kwargs):
        # if model_name == 'Yale-LILY/brio-xsum-cased':
        #     model_name = 'google/pegasus-xsum'
        # if model_name == 'Yale-LILY/brio-cnndm-uncased':
        #     model_name = 'facebook/bart-large-cnn'
        tokenizer = kwargs.pop("tokenizer", None)
        self.tokenizer = tokenizer if tokenizer is not None else AutoTokenizer.from_pretrained(model_name)
        self.cfg = cfg
        self.print_instance = 10

        self.sacrebleu = datasets.load_metric('sacrebleu')
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
        all_preds = []
        all_refs = []
        results = []
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
                pred.append(curr_pred)
                all_preds.append(curr_pred)
                all_refs.append(truth)
                results.append(meteor_score([text_normalization(truth).lower().split()], text_normalization(curr_pred).lower().split()))
            if i < self.print_instance and self.cfg.local_rank <= 0:
                logger.info("Truth: " + truth + " Predict: " + " ### ".join(pred))
        return np.mean(results)


from nltk.util import ngrams
from functools import reduce
import operator
import pickle as pkl
@register_metrics("rouge")
class Rouge():
    def __init__(self, model_name, cfg, which_rouge=None, max_n=1, **kwargs):
        """
        Calculate Rouge F1 metrics

        which_rouge: str or list of str, for example "rouge-n", "rouge-l", "rouge-w"
        max_n: N-grams for ROUGE-N if specify. Default:1
        """
        self.cfg = cfg
        self.ml_eval = cfg.eval_dir == 'sample/test_ml.sample.tsv'
        self.base_model_predict_file = cfg.base_model_predict_file
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

    def distinct_k(self, sentences):
        unigram = []
        bigram = []
        trigram = []
        for sent in sentences:
            s = sent.split()
            unigram.append(s)
            bigram.append(list(ngrams(s, 2)))
            trigram.append(list(ngrams(s, 3)))
        unigram = reduce(operator.concat, unigram)
        bigram = reduce(operator.concat, bigram)
        trigram = reduce(operator.concat, trigram)
        epss = 0.0000000000001
        d1 = len(set(unigram)) / (len(unigram) + epss)
        d2 = len(set(bigram)) / (len(bigram) + epss)
        d3 = len(set(trigram)) / (len(trigram) + epss)
        return d1, d2, d3

    def group_distinct_k(self, sentences):
        d1s = []
        d2s = []
        d3s = []
        for sent in sentences:
            unigram = []
            bigram = []
            trigram = []
            s = sent.split()
            unigram.append(s)
            bigram.append(list(ngrams(s, 2)))
            trigram.append(list(ngrams(s, 3)))
            unigram = reduce(operator.concat, unigram)
            bigram = reduce(operator.concat, bigram)
            trigram = reduce(operator.concat, trigram)
            epss = 0.0000000000001
            d1 = len(set(unigram)) / (len(unigram) + epss)
            d2 = len(set(bigram)) / (len(bigram) + epss)
            d3 = len(set(trigram)) / (len(trigram) + epss)
            d1s.append(d1)
            d2s.append(d2)
            d3s.append(d3)
        d1 = np.mean(d1s)
        d2 = np.mean(d2s)
        d3 = np.mean(d3s)
        return d1, d2, d3, d1s, d2s, d2s
    def get_token_dist(self, lines):
        word_count = {}
        word_count_by_line = {}
        total_words = 0
        for txt_id, text in enumerate(lines):
            words = text.split()
            words_set = set(words)
            for word in words:
                if word not in word_count:
                    word_count[word] = 1
                else:
                    word_count[word] += 1
                total_words += 1
            for word in words_set:
                if word not in word_count_by_line:
                    word_count_by_line[word] = 1
                else:
                    word_count_by_line[word] += 1
        word_count_per_line = {}
        for k, v in word_count.items():
            word_count_per_line[k] = v / len(lines)
        return word_count, word_count_per_line, word_count_by_line
    def get_dif(self, word_count_per_line_hyp, word_count_per_line_ref,  word_count_hyp, word_count_ref, word_count_by_line_hyp, word_count_by_line_ref):
        rate = {}
        for k, v in word_count_hyp.items():
            if k not in word_count_ref:
                rate[k] = v
                word_count_ref[k] = 0

            else:
                rate[k] = v - word_count_ref[k]
            if k not in word_count_by_line_ref:
                word_count_by_line_ref[k] = 0
            if k not in word_count_by_line_hyp:
                word_count_by_line_hyp[k] = 0
        word_dict = rate
        word_dict = sorted(word_dict.items(), key=lambda item: -item[1])
        tot10 = 0
        tot50 = 0
        tot100 = 0
        tot500 = 0
        tot1000 = 0
        for k, v in word_dict:
            if v > 10:
                tot10 += 1
            if v > 50:
                tot50 += 1
            if v > 100:
                tot100 += 1
            if v > 500:
                tot500 += 1
            if v > 1000:
                tot1000 += 1
        tot50 /= len(word_dict)
        tot500 /= len(word_dict)
        tot10 /= len(word_dict)
        tot100 /= len(word_dict)
        tot1000 /= len(word_dict)
        print('10: {};50: {};100: {}; 500: {}; 1000: {}'.format(tot10, tot50, tot100, tot500, tot1000))

        for k, v in word_dict[:10]:
            print(k, format(v, '.5f'), word_count_hyp[k], word_count_ref[k], word_count_by_line_hyp[k], word_count_by_line_ref[k])
    def get_max(self, word_count_per_line_hyp,  word_count_hyp,
                word_count_by_line_hyp):
        word_dict = sorted(word_count_hyp.items(), key=lambda item: -item[1])
        tot = sum(list(word_count_hyp.values()))
        for k, v in word_dict[:20]:
            print(k, v, format(v/tot, '.5f'))
        word_dict = sorted(word_count_by_line_hyp.items(), key=lambda item: -item[1])
        tot = sum(list(word_count_by_line_hyp.values()))
        for k, v in word_dict[:20]:
            print(k, v, format(v / tot, '.5f'))

    def get_kl(self, word_count_hyp, word_count_ref, word_count_ref2=None, name=''):
        word_count = {}
        if word_count_ref2 is not None:
            for k, v in word_count_ref.items():
                if k not in word_count_ref2:
                    word_count_ref2[k] = 0
                word_count[k] = max(v, word_count_ref2[k])
            for k, v in word_count_ref2.items():
                if k not in word_count_ref:
                    word_count_ref[k] = 0
                word_count[k] = max(v, word_count_ref[k])
        else:
            word_count = word_count_ref
        word_count_hyp_list = []
        word_count_ref_list = []
        for k, v in word_count.items():
            if k not in word_count_hyp:
                word_count_hyp[k] = 0
            word_count_hyp_list.append(word_count_hyp[k])
            word_count_ref_list.append(v)
        KL = scipy.stats.entropy(word_count_hyp_list, word_count_ref_list)
        print('KL:{}'.format(name), KL)
    def get_vocab(self, hyp):
        #ref0_file = 'ex1_predict.txt'
        #ref0_file = 'ex173_predict.txt'
        #ref0 = [r.strip().split('\t')[2] for r in ref0]
        #base_file = 'base_predict.txt'
        base_file = self.base_model_predict_file
        ref1 = open(base_file).readlines()
        gt0 = [r.strip().split('\t')[1] for r in ref1]
        ref1 = [r.strip().split('\t')[2] for r in ref1]
        #word_count_ref, word_count_per_line_ref, word_count_ref_by_line = self.get_token_dist(ref0)
        word_count_ref1, word_count_per_line_ref1, word_count_ref1_by_line = self.get_token_dist(ref1)
        word_count_hyp, word_count_per_line_hyp, word_count_hyp_by_line = self.get_token_dist(hyp)
        word_count_gt, word_count_per_line_gt, word_count_gt_by_line = self.get_token_dist(gt0)
        print('hyp tot:')
        self.get_max(word_count_per_line_hyp,  word_count_hyp, word_count_ref1_by_line)
        self.get_kl(word_count_hyp, word_count_gt, name='gt')
        self.get_kl(word_count_hyp, word_count_ref1, name='ref')
        self.get_kl(word_count_hyp, word_count_ref1, word_count_ref2=word_count_gt, name='refgt')
        #print('ex1:')
        #self.get_dif(word_count_per_line_hyp, word_count_per_line_ref, word_count_hyp, word_count_ref, word_count_hyp_by_line, word_count_ref_by_line)
        print('base:')
        self.get_dif(word_count_per_line_hyp, word_count_per_line_ref1, word_count_hyp, word_count_ref1, word_count_hyp_by_line, word_count_ref1_by_line)
        print('gt:')
        self.get_dif(word_count_per_line_hyp, word_count_per_line_gt, word_count_hyp, word_count_gt, word_count_hyp_by_line, word_count_gt_by_line)

    def split_by_lang(self, langs, res):
        res_dict = {}
        # {'n': []}
        for i, l in enumerate(langs):
            if l not in res_dict:
                res_dict[l] = [res[i]]
            else:
                res_dict[l].append(res[i])
        res_dict = {k: np.mean(v) for k, v in res_dict.items()}
        d = {'DA': 6541, 'DE': 47669, 'EN': 207984, 'ES': 8795, 'FR': 27620, 'IT': 8386, 'NL': 27448, 'SV': 178}
        d_sum = sum(list(d.values()))
        d = {k: 1.0 * v / d_sum for k, v in d.items()}
        res_dict = {k: np.mean(v) for k, v in res_dict.items()}
        res_dict['avg'] = 0
        res_dict['w_avg'] = 0
        for k, v in res_dict.items():
            if k != 'avg' and k != 'w_avg':
                res_dict['avg'] += v / len(d)
                res_dict['w_avg'] += d[k] * v

        return res_dict
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
                #debug_pkl.append([curr_pred, truth, rouge_score])
                for k in rouge_score:
                    tmp_rouge[k].append(rouge_score[k]["f"])
            for k in tmp_rouge:
                rouge_list[k].append(max(tmp_rouge[k]))
            
            if i < self.print_instance and self.cfg.local_rank <= 0:
                logger.info("Truth: "+truth+" Predict: "+" ### ".join(pred))
                # print('Tokens: ', curr_tokens )
        self.get_vocab(preds)
        #pkl.dump(debug_pkl, open('debug_pkl', 'wb'))
        dist = self.distinct_k(preds)
        d1, d2, d3, d1s, d2s, d3s = self.group_distinct_k(preds)
        #pkl.dump(preds, open('/mnt/shared_data/zr_output/xsum_pegasus_res.pkl', 'wb'))
        #assert 1==0
        # truths = []
        scores = dict()
        for k in rouge_list:
            scores[k] = np.mean(rouge_list[k])
            if self.ml_eval:
                res_dict=self.split_by_lang(langs=EvalPredict.raw_src, res=rouge_list[k])
                print(k, res_dict)
        scores['dist'] = dist
        scores['gdist'] = (d1, d2, d3)
        if self.ml_eval:
            res_dict=self.split_by_lang(langs=EvalPredict.raw_src, res=d1s)
            print('d1', res_dict)
            res_dict = self.split_by_lang(langs=EvalPredict.raw_src, res=d2s)
            print('d2', res_dict)
            res_dict = self.split_by_lang(langs=EvalPredict.raw_src, res=d3s)
            print('d3', res_dict)
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
    def distinct_k(self, sentences):
        unigram = []
        bigram = []
        trigram = []
        for sent in sentences:
            sent = sent.lower()
            s = sent.split()
            unigram.append(s)
            bigram.append(list(ngrams(s, 2)))
            trigram.append(list(ngrams(s, 3)))
        unigram = reduce(operator.concat, unigram)
        bigram = reduce(operator.concat, bigram)
        trigram = reduce(operator.concat, trigram)
        epss = 0.0000000000001
        d1 = len(set(unigram)) / (len(unigram) + epss)
        d2 = len(set(bigram)) / (len(bigram) + epss)
        d3 = len(set(trigram)) / (len(trigram) + epss)
        return d1, d2, d3

    def group_distinct_k(self, sentences):
        d1s = []
        d2s = []
        d3s = []
        for sent in sentences:
            unigram = []
            bigram = []
            trigram = []
            sent = sent.lower()
            s = sent.split()
            unigram.append(s)
            bigram.append(list(ngrams(s, 2)))
            trigram.append(list(ngrams(s, 3)))
            unigram = reduce(operator.concat, unigram)
            bigram = reduce(operator.concat, bigram)
            trigram = reduce(operator.concat, trigram)
            epss = 0.0000000000001
            d1 = len(set(unigram)) / (len(unigram) + epss)
            d2 = len(set(bigram)) / (len(bigram) + epss)
            d3 = len(set(trigram)) / (len(trigram) + epss)
            d1s.append(d1)
            d2s.append(d2)
            d3s.append(d3)
        d1 = np.mean(d1s)
        d2 = np.mean(d2s)
        d3 = np.mean(d3s)
        return d1, d2, d3

    def get_token_dist(self, lines):
        word_count = {}
        word_count_by_line = {}
        total_words = 0
        for txt_id, text in enumerate(lines):
            words = text.split()
            words_set = set(words)
            for word in words:
                if word not in word_count:
                    word_count[word] = 1
                else:
                    word_count[word] += 1
                total_words += 1
            for word in words_set:
                if word not in word_count_by_line:
                    word_count_by_line[word] = 1
                else:
                    word_count_by_line[word] += 1
        word_count_per_line = {}
        for k, v in word_count.items():
            word_count_per_line[k] = v / len(lines)
        return word_count, word_count_per_line, word_count_by_line

    def get_dif(self, word_count_per_line_hyp, word_count_per_line_ref, word_count_hyp, word_count_ref,
                word_count_by_line_hyp, word_count_by_line_ref):
        rate = {}
        for k, v in word_count_hyp.items():
            if k not in word_count_ref:
                rate[k] = v
                word_count_ref[k] = 0
                word_count_by_line_ref[k] = 0
            else:
                rate[k] = v - word_count_ref[k]
        word_dict = rate
        word_dict = sorted(word_dict.items(), key=lambda item: -item[1])
        tot10 = 0
        tot50 = 0
        tot100 = 0
        tot500 = 0
        tot1000 = 0
        for k, v in word_dict:
            if v > 10:
                tot10 += 1
            if v > 50:
                tot50 += 1
            if v > 100:
                tot100 += 1
            if v > 500:
                tot500 += 1
            if v > 1000:
                tot1000 += 1
        tot50 /= len(word_dict)
        tot500 /= len(word_dict)
        tot10 /= len(word_dict)
        tot100 /= len(word_dict)
        tot1000 /= len(word_dict)
        print('10: {};50: {};100: {}; 500: {}; 1000: {}'.format(tot10, tot50, tot100, tot500, tot1000))
        for k, v in word_dict[:10]:
            print(k, format(v, '.5f'), word_count_hyp[k], word_count_ref[k], word_count_by_line_hyp[k],
                  word_count_by_line_ref[k])

    def get_max(self, word_count_per_line_hyp, word_count_hyp,
                word_count_by_line_hyp):
        word_dict = sorted(word_count_hyp.items(), key=lambda item: -item[1])
        tot = sum(list(word_count_hyp.values()))
        for k, v in word_dict[:20]:
            print(k, v, format(v / tot, '.5f'))
        word_dict = sorted(word_count_by_line_hyp.items(), key=lambda item: -item[1])
        tot = sum(list(word_count_by_line_hyp.values()))
        for k, v in word_dict[:20]:
            print(k, v, format(v / tot, '.5f'))

    def get_vocab(self, hyp):
        ref0_file = 'ex1_predict.txt'
        ref0_file = 'ex173_predict.txt'
        ref0 = open(ref0_file).readlines()
        gt0 = [r.strip().split('\t')[1] for r in ref0]
        ref0 = [r.strip().split('\t')[2] for r in ref0]
        base_file = 'base_predict.txt'
        base_file = 'b_predict.txt'
        ref1 = open(base_file).readlines()
        ref1 = [r.strip().split('\t')[2] for r in ref1]
        word_count_ref, word_count_per_line_ref, word_count_ref_by_line = self.get_token_dist(ref0)
        word_count_ref1, word_count_per_line_ref1, word_count_ref1_by_line = self.get_token_dist(ref1)
        word_count_hyp, word_count_per_line_hyp, word_count_hyp_by_line = self.get_token_dist(hyp)
        word_count_gt, word_count_per_line_gt, word_count_gt_by_line = self.get_token_dist(gt0)
        print('hyp tot:')
        self.get_max(word_count_per_line_hyp, word_count_hyp, word_count_ref_by_line)
        print('ex1:')
        self.get_dif(word_count_per_line_hyp, word_count_per_line_ref, word_count_hyp, word_count_ref,
                     word_count_hyp_by_line, word_count_ref_by_line)
        print('base:')
        self.get_dif(word_count_per_line_hyp, word_count_per_line_ref1, word_count_hyp, word_count_ref1,
                     word_count_hyp_by_line, word_count_ref1_by_line)
        print('gt:')
        self.get_dif(word_count_per_line_hyp, word_count_per_line_gt, word_count_hyp, word_count_gt,
                     word_count_hyp_by_line, word_count_gt_by_line)

    def __call__(self, EvalPredict):
        if EvalPredict.predictions.ndim == 3:
            res = EvalPredict.predictions
        else:
            res = np.expand_dims(EvalPredict.predictions, axis=-2)
        lang = EvalPredict.predictions

        rouge_list = defaultdict(list)
        debug_pkl = []
        preds = []
        #print(len(res))
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
                if '..' in curr_pred:
                    cont_tot += 1
                if '. he says.' in curr_pred.lower() or '. she says.' in curr_pred.lower():
                    heshe_tot += 1
                if '. they say.' in curr_pred.lower():
                    they_tot += 1
                pred.append(curr_pred)
                curr_pred = "\n".join(nltk.sent_tokenize(" ".join(nltk.word_tokenize(curr_pred))))
                preds.append(curr_pred)
                curr_pred = "\n".join(nltk.sent_tokenize(curr_pred))
                # truth = "\n".join(nltk.sent_tokenize(truth))
                truth = "\n".join(nltk.sent_tokenize(" ".join(nltk.word_tokenize(truth))))
                res_rouge = self.rouge.score(curr_pred.lower(), truth.lower())
                for k in res_rouge:
                    tmp_rouge[k].append(res_rouge[k].fmeasure)
            for k in tmp_rouge:
                rouge_list[k].append(max(tmp_rouge[k]))
            if i < self.print_instance and self.cfg.local_rank <= 0:
                # logger.info("Truth: " + truth + " Predict: " + " ### ".join(pred))
                print("Truth: " + truth + " Predict: " + " ### ".join(pred))
        # pkl.dump(preds, open('base_file.pkl', 'wb'))
        scores = dict()
        for k in rouge_list:
            scores[k] = np.mean(rouge_list[k])
        length = []
        for t in preds:
            length.append(len(t.split()))
        scores['length'] = np.mean(length)
        scores['cont_tot'] = cont_tot
        scores['heshe_tot'] = heshe_tot
        scores['they_tot'] = they_tot
        print(scores)
        if len(scores) == 1:
            return list(scores.values())[0]
        else:
            return scores


@register_metrics("rouge1f")
class Rouge1F1(Rouge):
    """Only difference with Rouge class is that it only returns Rouge-1 F1 number for
    selecting checkpoint during training
    """
    def __init__(self, model_name, cfg, **kwargs):
        super().__init__(model_name, cfg, "rouge-n", 1)
