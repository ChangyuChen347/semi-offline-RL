import torch
import os
import numpy as np
from scipy.special import softmax

from . import register_outputter
from transformers import AutoTokenizer
from data.helper.formatter import Formatter
#from multiprocessing import Process,Queue
from pathos.helpers import mp
from typing import List, Union, Tuple 


class DummyOutQueue:

    def __init__(self, output_f):
        self.output_f = output_f
    
    def put(self, line):
        self.output_f.write(line)


class MPOutputter:
    """  
    Outputter for generatl task. Worker flow is 
    `pred_value(token_ids)` put in queue 
    ==> postprocess worker fetch queue and decode 
    ==> writer worker fetch from out_queue and write out 
    """
    def __init__(self,args,model_name=None, **kwargs):
        self.output_dir = args.output_dir
        self.result_file = args.result_file
        self.result_header = list(filter(lambda x: len(x) > 0,args.result_header.split(",")))
        self.kept_header = set(self.result_header)
        tokenizer = kwargs.pop("tokenizer", None)
        self.tokenizer = tokenizer if tokenizer is not None else AutoTokenizer.from_pretrained(model_name)

        if args.world_size > 1:
            self.of =open(os.path.join(self.output_dir,self.result_file+"_"+str(args.local_rank)),mode='w',encoding="utf-8")
        else:
            self.of =open(os.path.join(self.output_dir,self.result_file),mode='w',encoding="utf-8")
        print("\033[0;37;41m\t{}\033[0m".format(self.of))

        if args.outputter_num_workers is None:
            self.workers = args.dataloader_num_workers
        else:
            self.workers = args.outputter_num_workers
        if self.workers == 0: # disable multi-processing
            self.in_queue = None
            self.out_queue = DummyOutQueue(self.of)
        else:
            self.in_queue = mp.Queue()
            self.out_queue = mp.Queue()
            self.end_process_cnt = 0
    
    def realtime_output(self,batch,pred):
        if isinstance(pred,dict):
            pred_value = {key:pred[key].cpu() for key in pred if pred[key] != None}
        elif isinstance(pred,tuple):
            pred_value = [p.cpu() for p in pred if p!=None]
        else:
            pred_value = pred.cpu()
        return self.return_res(batch,pred_value)

    def return_res(self,batch,pred):
        NotImplementedError

    def __call__(self,batch,pred):
        if batch == None:
            self.in_queue.put((None,None))
            return
        batch_value = {key:batch[key] for key in self.kept_header}
        if isinstance(pred,dict):
            pred_value = {key:pred[key].cpu() for key in pred if pred[key] != None}
        elif isinstance(pred,tuple):
            pred_value = [p.cpu() for p in pred]
        else:
            pred_value = pred.cpu()
        
        if self.workers == 0:
            self.out_batch(batch_value,pred_value)
        else:
            self.in_queue.put((batch_value,pred_value))
    
    def writer_process(self):
        #print(len(self.out_queue))
        while True:
            line = self.out_queue.get()
            if line == None:
                self.end_process_cnt += 1
                if self.end_process_cnt == self.workers:
                    print("Writer recieve end notice!")
                    break
            else:
                self.of.write(line)
                self.of.flush()
    
    def post_process(self):
        while True:
            batch,pred = self.in_queue.get()
            if batch == None:
                print("Recieve end notice!")
                self.out_queue.put(None)
                break

            self.out_batch(batch,pred)
    
    def start_process(self):
        if self.workers ==0:
            pass

        self.proc = []
        for i in range(0,self.workers):
            self.proc.append(mp.Process(target=self.post_process, args=()))
            self.proc[i].start()
        self.writer_proc = mp.Process(target=self.writer_process,args=())
        self.writer_proc.start()
    
    def close(self):
        if self.workers > 0:
            for i in range(0,self.workers):
                self(None,None)
            for p in self.proc:
                p.join()
            self.writer_proc.join()
        self.of.close()

class BasicOutputter:
    def __init__(self,args,model_name=None):
        self.output_dir = args.output_dir
        self.result_file = args.result_file
        self.result_header = args.result_header.split(",")
        self.of = open(os.path.join(self.output_dir,self.result_file),mode='w',encoding='utf-8')
    def output(self,batch,pred):
        self.out_batch(batch,pred)
    def close(self):
        self.of.close()

@register_outputter("score")
class ClassificationOutputter(MPOutputter):
    def __init__(self,args,model_name):
        super().__init__(args,model_name)
        if self.workers:
            self.start_process()
    def out_batch(self,batch,res):
        if res.ndim == 2:
            res = softmax(res,axis=-1)
        res = res.numpy()

        for i in range(0,len(res)):
            line = []
            for key in self.result_header:
                line.append(batch[key][i])
            if res.ndim == 1:
                line.append(str(res[i]))
            else:
                line.append(",".join([str(score) for score in res[i]]))
            newline = "\t".join(line)+"\n"
            self.out_queue.put(newline)

@register_outputter("single_score")
class MultiResultOutputter(MPOutputter):
    """If results contains multiple 1d tensors, e.g., intermediate representation, only write the first one"""
    def __init__(self,args,model_name):
        super().__init__(args,model_name)
        if self.workers:
            self.start_process()

    def out_batch(self,batch,res):
        """
        :param batch: dict
        :param res: tuple of tensors|tensor with shape (batch_dim, other_dim)
        """
        def tensor2str(t):
            t = t.numpy()
            if t.ndim == 1:
                t = t[:, np.newaxis]
            return [",".join(str(score) for score in t[i]) for i in range(t.shape[0])]

        if isinstance(res, torch.Tensor):
            serialized_res = tensor2str(res)
        elif isinstance(res, (tuple, list)):
            serialized_res = tensor2str(res[0])  # first element
        else:
            raise NotImplementedError(f"res type = {type(res)} not supported")

        for i in range(0,len(serialized_res)):
            line = []
            for key in self.result_header:
                line.append(batch[key][i])

            line.append(serialized_res[i])
            newline = "\t".join(line)+"\n"
            self.out_queue.put(newline)

@register_outputter("score_and_more")
class MultiResultOutputter(MPOutputter):
    """Results contains multiple 1d tensors, e.g., intermediate representation"""

    def __init__(self, args, model_name, **kwargs):
        super().__init__(args, model_name, **kwargs)
        if self.workers:
            self.start_process()

    def out_batch(self,batch,res):
        """
        organize output, put tensors in both batch and prediction output to str form 

        Arguments: 
            :param batch: dict
            :param res: tuple of tensors|tensor with shape (batch_dim, other_dim)
        """
        def tensor2str(t):
            t = t.numpy()
            if t.ndim == 1:
                t = t[:, np.newaxis]
            return [",".join(str(score) for score in t[i]) for i in range(t.shape[0])]

        def cvt_res_to_str(res: Union[List[str], torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor]]) : 
            if isinstance(res[0], str) : 
                serialized_res = res
            elif isinstance(res, torch.Tensor):
                serialized_res = tensor2str(res)
            elif isinstance(res, (tuple, list)):
                each_serialized_element = []
                for r in res:
                    each_serialized_element.append(tensor2str(r))
                element_per_batch = zip(*each_serialized_element)
                serialized_res = ["\t".join(l) for l in element_per_batch] # each element in each is a row
            else:
                raise NotImplementedError(f"res type = {type(res)} not supported")

            return serialized_res

        serialized_res = cvt_res_to_str(res)
        for i in range(0,len(serialized_res)):
            line = []
            for key in self.result_header:
                _key_batch = cvt_res_to_str(batch[key])
                line.append(_key_batch[i])

            line.append(serialized_res[i])
            newline = "\t".join(line)+"\n"
            self.out_queue.put(newline)

@register_outputter("multi_class")
class MultiClassOutputter(MPOutputter):
    def __init__(self, args, model_name, **kwargs):
        super().__init__(args, model_name, **kwargs)
        self.mapping = self.parse_class_cfg(args.output_mapping)
        if self.workers:
            self.start_process()
    def parse_class_cfg(self,s):
        res = {}
        clss = s.split(",")
        for c in clss:
            key,value=c.split(":")
            res[int(key)] = value
        return res
    def out_batch(self,batch,res):
        res = res.numpy()
        for i in range(0,len(res)):
            line = []
            for key in self.result_header:
                line.append(batch[key][i])
            idx = np.argmax(res[i])
            line.append(self.mapping[idx])
            newline = "\t".join(line)+"\n"
            self.out_queue.put(newline)


@register_outputter("generation")
class Seq2SeqOutputter(MPOutputter):
    def __init__(self, args, model_name, **kwargs):
        super().__init__(args, model_name, **kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.workers:
            self.start_process()

    def return_res(self,batch,res):
        assert 1==0
        rewrite = res[1] # 0 is fake loss
        res = []
        print(len(rewrite))
        for i in range(0,len(rewrite)):
            print(len(rewrite[i]))
            for j in range(0,len(rewrite[i])):
                newline = [self.tokenizer.decode(rewrite[i][j],skip_special_tokens=True,clean_up_tokenization_spaces=False),str(j)]
                res.append(newline)
        return res
    
    def out_batch(self,batch,res):
        rewrite = res[1] # 0 is fake loss
        for i in range(0,len(rewrite)):
            line = []
            for key in self.result_header:
                line.append(batch[key][i])
            for j in range(0, len(rewrite[i])):
                #output_tokens = self.tokenizer.convert_ids_to_tokens(rewrite[i][j])
                #print(output_tokens)
                res_score = [self.tokenizer.decode(rewrite[i][j],skip_special_tokens=True,clean_up_tokenization_spaces=False),str(j)]
                #print(res_score)
                newline = "\t".join(line + res_score)+"\n"
                self.out_queue.put(newline)

@register_outputter("selector_generation")
class Seq2SeqOutputter(MPOutputter):
    def __init__(self, args, model_name, **kwargs):
        super().__init__(args, model_name, **kwargs)
        self.sen_sep = args.sen_sep
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.workers:
            self.start_process()
    def return_res(self,batch,res):
        rewrite = res[1] # 0 is fake loss
        assert 1==0
        res = []
        for i in range(0,len(rewrite)):
            for j in range(0,len(rewrite[i])):
                newline = [self.tokenizer.decode(rewrite[i][j],skip_special_tokens=True,clean_up_tokenization_spaces=False),str(j)]
                res.append(newline)
        return res
    def out_batch(self,batch,res):
        rewrite = res[1] # 0 is fake loss
        selected_idxs = res[2]
        selected_probs = res[3]
        latent_num = res[4]
        for i in range(0,len(rewrite)):
            line = []
            for key in self.result_header:
                line.append(batch[key][i])
            for j in range(0,len(rewrite[i])):
                res_score = [self.tokenizer.decode(rewrite[i][j],skip_special_tokens=True,clean_up_tokenization_spaces=False),str(j)]
                res_score += [str(selected_probs[i][j//latent_num].item()), batch['lpsents'][i].split(self.sen_sep)[selected_idxs[i][j//latent_num]]]
                newline = "\t".join(line+res_score)+"\n"
                self.out_queue.put(newline)

@register_outputter("formatter")
class FormatOutputter(MPOutputter):
    def __init__(self, args, model_name, **kwargs):
        super().__init__(args, model_name, **kwargs)
        self.formatter = Formatter(model_name,args,save=False)
        self.kept_header.add("query")
        if self.workers:
            self.start_process()
    def out_batch(self,batch,res):
        res = res["tag"]
        for i in range(0,len(res)):
            line = []
            for key in self.result_header:
                line.append(batch[key][i])
            pred = np.argmax(res[i],axis=-1)
            query = batch["query"][i]
            pred = self.formatter.realize(query,pred)
            newline = "\t".join(line+[pred]) + "\n"
            self.out_queue.put(newline)
