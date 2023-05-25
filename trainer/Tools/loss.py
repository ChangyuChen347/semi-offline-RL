from logging import log
import torch 
from dataclasses import dataclass
from torch.nn import CrossEntropyLoss
from typing import Any 


def hinge(z): 
    return torch.clip(1 - z, min=0)

def exponential(z): 
    return torch.exp(-z)

def logistic(z): 
    return torch.log(1 + torch.exp(-z))

GAIN_FUNC = {
    "hinge" : hinge, 
    "exponential" : exponential, 
    "logistic" : logistic
}

@dataclass
class LabelSmoother:
    """
    Adds label-smoothing on a pre-computed output from a Transformers model.

    Args:
        epsilon (:obj:`float`, `optional`, defaults to 0.1):
            The label smoothing factor.
        ignore_index (:obj:`int`, `optional`, defaults to -100):
            The index in the labels to ignore when computing the loss.
    """

    epsilon: float = 0.1
    ignore_index: int = -100

    def __call__(self, model_output, labels, return_losses=False):
        logits = model_output["logits"] if isinstance(model_output, dict) else model_output[0]
        log_probs = -torch.nn.functional.log_softmax(logits, dim=-1)
        if labels.dim() == log_probs.dim() - 1:
            labels = labels.unsqueeze(-1)

        padding_mask = labels.eq(self.ignore_index)
        # In case the ignore_index is -100, the gather will fail, so we replace labels by 0. The padding_mask
        # will ignore them in any case.
        labels.clamp_min_(0)
        nll_loss = log_probs.gather(dim=-1, index=labels)
        # works for fp16 input tensor too, by internally upcasting it to fp32
        smoothed_loss = log_probs.sum(dim=-1, keepdim=True, dtype=torch.float32)

        nll_loss.masked_fill_(padding_mask, 0.0)
        smoothed_loss.masked_fill_(padding_mask, 0.0)

        # Take the mean over the label dimensions, then divide by the number of active elements (i.e. not-padded):
        num_active_elements = padding_mask.numel() - padding_mask.long().sum()
        nll_loss = nll_loss.sum() / num_active_elements
        smoothed_loss = smoothed_loss.sum() / (num_active_elements * log_probs.shape[-1])
        weighted_loss = (1 - self.epsilon) * nll_loss + self.epsilon * smoothed_loss

        batch_num_active_elements = padding_mask.shape[1] - padding_mask.long().sum(dim=-1)
        batch_nll_loss = nll_loss.sum(dim=-1) / batch_num_active_elements
        batch_smoothed_loss = smoothed_loss.sum(dim=-1) / (batch_num_active_elements * log_probs.shape[-1])
        batch_loss = (1 - self.epsilon) * batch_nll_loss + self.epsilon * batch_smoothed_loss
        
        return weighted_loss, batch_loss 

@dataclass
class LabelLoss : 
    """
    regular cross-entropy loss with output of both loss and batch loss
    """
    ignore_index : int = -100 

    def __call__(self, model_output, labels, return_losses=False) : 
        assert "logits" in model_output.keys() or "loss" in model_output.keys()
        
        # no logits output in evaluation mode 
        if "logits" in model_output.keys() : 
            logits = model_output["logits"] if isinstance(model_output, dict) else model_output[0]

            padding_mask = labels.eq(self.ignore_index)
            num_active_elements = padding_mask.numel() - padding_mask.long().sum()
            batch_num_active_elements = padding_mask.shape[1] - padding_mask.long().sum(dim=-1)

            assert labels is not None, print("labels is missing for loss compute !")
            batch_loss_fct = CrossEntropyLoss(ignore_index=-100, reduction="none")
            logits = logits.permute(0, 2, 1)
            batch_loss = batch_loss_fct(logits, labels)
            loss = batch_loss.sum() / num_active_elements
            batch_loss = batch_loss.sum(-1) / batch_num_active_elements
            return loss, batch_loss 
            
        elif "loss" in model_output.keys() : 
            return model_output["loss"]
        
        
class PairWiseRankLoss: 
    """
    Pairwise ranking loss, focused on pair-comparison  
    referenced from https://proceedings.neurips.cc/paper/2009/file/2f55707d4193dc27118a0f19a1985716-Paper.pdf
    """
    def __call__(self, model_output, rank_labels, gain_fn=hinge) -> Any:
        logits = model_output["logits"]
        
        logits = logits.unsqueeze(-1)
        rank_labels = rank_labels.unsqueeze(-1)
        score_mtx = logits.permute(0, 2, 1) - logits 
        pair_rank_mtx = rank_labels.permute(0, 2, 1) - rank_labels
        low_rank_mask = pair_rank_mtx > 0  
        atd_active_score = gain_fn(score_mtx) * low_rank_mask
        loss = atd_active_score.sum() / atd_active_score.shape[0]

        return loss 

class ListWiseRankLoss: 
    """
    Listwise ranking loss, focused on (i, lower-rank-list(i)) comparison
    referenced from https://proceedings.neurips.cc/paper/2009/file/2f55707d4193dc27118a0f19a1985716-Paper.pdf
    """
    def __call__(self, model_output, rank_labels) -> Any:
        logits = model_output["logits"]
        rep_logits = logits.unsqueeze(-1)
        rank_labels = rank_labels.unsqueeze(-1)

        rep_logits = rep_logits.repeat(1, 1, logits.shape[-1])
        act_rank_mtx = (rank_labels.permute(0, 2, 1) - rank_labels) >= 0 
        act_rank_mtx = act_rank_mtx.clone()

        low_rank_loss = torch.log((torch.exp(rep_logits) * act_rank_mtx).sum(1))
        loss = (low_rank_loss - logits).sum() / low_rank_loss.shape[0]

        return loss 
