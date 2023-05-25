from . import register_outputter
from trainer.Outputter.basic_outputter import MPOutputter
from transformers import AutoTokenizer

@register_outputter("selector")
class SelectorOutputter(MPOutputter):
    def __init__(self,args,model_name):
        super().__init__(args,model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.workers:
            self.start_process()
    
    def out_batch(self,batch,res):
        selected_id, sentence_cnt = res[1:] # 0 is fake loss
        if len(selected_id.shape) == 2: 
            sources = self.tokenizer.batch_decode(selected_id,skip_special_tokens=True,clean_up_tokenization_spaces=False)

            for i in range(0,len(sources)):
                line = []
                for key in self.result_header:
                    line.append(batch[key][i])
                line.append(sources[i])
                self.out_queue.put("\t".join(line)+"\n")

        elif len(selected_id.shape) == 3: 
            bs, sample_num, seq_len = selected_id.shape
            for i in range(bs): 
                line = []
                for key in self.result_header: 
                    line.append(batch[key][i])
                for j in range(sample_num): 
                    source = self.tokenizer.decode(selected_id[i][j],skip_special_tokens=True,clean_up_tokenization_spaces=False)
                    self.out_queue.put("\t".join(line + [source, str(j)]) + "\n")
                    #print(line + [source, str(j)])