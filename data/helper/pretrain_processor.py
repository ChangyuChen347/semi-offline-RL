import torch
from transformers.data.data_collator import *

class MLMCollator(DataCollatorForLanguageModeling):
    #def __init__(self,idx,out_name,model,cfg=None):
    #    super().__init__(idx,out_name,model,cfg)
    #    self.mlm = getattr(cfg,"mlm",True) if cfg else True
    #    self.mlm_probability = getattr(cfg,"mlm_prob",0.15) if cfg else 0.15
    #    self.dc = DataCollatorForLanguageModeling(tokenizer=self.fn,mlm=self.mlm,mlm_probability=self.mlm_prob)
    def __call__(self,col0,col1,maxlen):
        ids = self.tokenizer.encode_plus(col0,text_pair=col1,max_length=maxlen,truncation=True)
        torch_ids = torch.tensor([ids["input_ids"]],dtype=torch.int64,device=torch.device("cpu"))
        res,label = self.mask_tokens(torch_ids)
        return {"input_ids":res.numpy()[0],"labels":label.numpy()[0],"token_type_ids":ids["token_type_ids"],"attention_mask":ids["attention_mask"]}

class WholeWordMLMCollator(DataCollatorForWholeWordMask):
    def __call__(self,col0,col1,maxlen):
        ids = self.tokenizer.encode_plus(col0,text_pair=col1,max_length=maxlen,truncation=True)
        ref_tokens = []
        for idx in ids["input_ids"]:
            token = self.tokenizer._convert_id_to_token(idx)
            ref_tokens.append(token)
        mask_labels = self._whole_word_mask(ref_tokens)
        torch_ids = torch.tensor([ids["input_ids"]],dtype=torch.int64,device=torch.device("cpu"))
        torch_mask = torch.tensor([mask_labels],dtype=torch.int64,device=torch.device("cpu"))
        res,label = self.mask_tokens(torch_ids,torch_mask)
        return {"input_ids":res.numpy()[0],"labels":label.numpy()[0],"token_type_ids":ids["token_type_ids"],"attention_mask":ids["attention_mask"]}

if __name__ == "__main__":
    from transformers import AutoTokenizer
    fn = AutoTokenizer.from_pretrained("bert-base-uncased")
    mc = WholeWordMLMCollator(fn,True,0.15)

    print(mc("amanda is very angry","hahahaha",10))
