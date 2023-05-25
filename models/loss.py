import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
import numpy as np

class SelectorGenLoss(_Loss):
    def __init__(self, sample_num, sample_loss_weight, latent_num, latent_loss_weight):
        super().__init__()
        self.sample_num = sample_num
        self.sample_loss_weight = sample_loss_weight
        self.latent_num = latent_num
        self.latent_loss_weight = latent_loss_weight

    def forward(self,input,target,sample_logits,select_mask,latent_logits):
        # input         (page_num*sample_num*latent_num)*(tgt_seq_len-2)*|V|
        # target        (page_num*sample_num*latent_num)*(tgt_seq_len-2)
        # sample_logits page_num*sample_num
        # select_mask   page_num*sample_num
        # latent_logits (page_num*sample_num)*latent_num
        select_mask_expand = torch.repeat_interleave(select_mask, self.latent_num, dim=1)
        loss_lm = F.cross_entropy(input.transpose(-1,-2),target,reduction='none')
        loss_lm = (torch.sum(loss_lm,dim=-1)/torch.count_nonzero(target+100,dim=-1)).view(-1,self.sample_num*self.latent_num)
        loss_lm[select_mask_expand==0] = float('inf')
        min_idx = torch.argmin(loss_lm,dim=-1)
        sample_min_idx = torch.div(min_idx, self.latent_num, rounding_mode='floor')
        loss = torch.min(loss_lm,dim=-1).values
        if self.sample_loss_weight > 0: # seletor loss function
            sample_logits = torch.sigmoid(sample_logits)
            loss_sample = torch.abs(torch.ones_like(sample_logits).scatter_(1, sample_min_idx.unsqueeze(-1), 0) - sample_logits)
            loss_sample = -torch.log(loss_sample)
            loss_sample *= select_mask
            loss_sample = torch.sum(loss_sample, dim=-1) / torch.sum(select_mask, dim=-1)
            loss += loss_sample * self.sample_loss_weight
        if self.latent_loss_weight > 0: # calculate the latent classifier loss and check if match
            latent_mask = torch.repeat_interleave(torch.zeros_like(sample_logits).scatter_(1, sample_min_idx.unsqueeze(-1), 1), self.latent_num, dim=1)
            latent_logits = latent_logits.view(-1,self.sample_num*self.latent_num)
            latent_logits[latent_mask == 0] = float('-inf')
            loss_latent = torch.softmax(latent_logits, dim=-1)
            loss_latent = torch.gather(loss_latent, dim=-1, index=min_idx.unsqueeze(-1)).squeeze(-1)
            loss_latent = -torch.log(loss_latent)
            max_idx = torch.argmax(latent_logits,dim=-1)
            loss = torch.where(torch.eq(max_idx,min_idx),loss,torch.tensor(0.0,dtype=loss.dtype).to(loss.device))
            loss += loss_latent * self.latent_loss_weight
        return torch.mean(loss)

class LatentGuidedLoss(_Loss):
    def __init__(self,latent_class, cls_weight):
        super().__init__()
        self.latent_class = latent_class
        self.cls_weight = cls_weight

    def forward(self,input,target,cls_logits,enc_logits):
        loss_lm = F.cross_entropy(input.transpose(-1,-2),target,reduction='none')
        loss_lm = (torch.sum(loss_lm,dim=-1)/torch.count_nonzero(target+100,dim=-1)).view(-1,self.latent_class)
        min_idx = torch.argmin(loss_lm,dim=-1)
        loss = torch.min(loss_lm,dim=-1).values
        if self.cls_weight > 0:
            loss_cls = F.cross_entropy(cls_logits,min_idx) + F.cross_entropy(enc_logits,min_idx)
            max_cls = torch.argmax(cls_logits,dim=-1)
            #print(max_cls.dtype)
            loss = torch.where(torch.eq(max_cls,min_idx),loss,torch.tensor(0.0,dtype=loss.dtype).to(loss.device))
            loss += loss_cls * self.cls_weight
        return torch.mean(loss)



if __name__ == '__main__':
    loss_fct = LatentGuidedLoss(2,1)
    input = torch.randn(10,10,5)
    target = torch.randint(4,[10,10])
    target[0,5:] = -100
    cls_logits = torch.randn(5,2)
    print("01.Input",input,target,cls_logits)
    loss_fct(input,target,cls_logits)

