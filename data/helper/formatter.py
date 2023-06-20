from transformers import AutoTokenizer
import unicodedata
from enum import IntEnum
import collections
from transformers import logging
logger = logging.get_logger(__name__)
class Tags:#(IntEnum):
    noChange = 0
    capFirst = 1
    capSecond = 2
    capFirstThird = 3
    allCap = 4
    capExLast = 5
    capFirstTwo = 6
    capExceptFirst = 7
    lookUp = 10
    leftSpace = 11
    rightSpace = 12
    noSpace = 13
    allSpace = 14

def _is_punctuation(char):
  """Checks whether `chars` is a punctuation character."""
  cp = ord(char)
  # We treat all non-letter/number ASCII as punctuation.
  # Characters such as "^", "$", and "`" are not in the Unicode
  # Punctuation class but we treat them as punctuation anyways, for
  # consistency.
  if ((cp >= 33 and cp <= 47) or (cp >= 58 and cp <= 64) or
      (cp >= 91 and cp <= 96) or (cp >= 123 and cp <= 126)):
    return True
  cat = unicodedata.category(char)
  if cat.startswith("P"):
    return True
  return False

def _run_strip_accents(text):
    """Strips accents from a piece of text."""
    text = unicodedata.normalize("NFD", text)
    output = []
    for char in text:
        cat = unicodedata.category(char)
        if cat == "Mn":
            continue
        output.append(char)
    return "".join(output)

class Formatter():
    def __init__(self,model,cfg=None,save=True,debug=False):
        additional_special_tokens=cfg.special_tokens.strip().split(",")
        self.tokenizer = AutoTokenizer.from_pretrained(model,cache_dir=cfg.cache_dir if cfg else None,additional_special_tokens=additional_special_tokens)
        self.tokenizer.save_pretrained(cfg.output_dir)
        self.special_tokens = set(self.tokenizer.all_special_tokens_extended)
        self.format_vocab_file = (cfg.previous_dir + "/format_vocab.txt") if cfg else None
        self.format_vocab = dict()
        self.new_vocab = collections.defaultdict(collections.Counter)
        self.format_level = cfg.format_level
        self.debug = debug
        self.load_vocab(self.format_vocab_file)
        self.format_ops = {
            Tags.noChange: lambda x: x,
            Tags.capFirst: lambda x: x[0].upper()+x[1:] if len(x)>=1 else x,
            Tags.capSecond: lambda x: x[0] + x[1].upper() + x[2:] if len(x) >= 2 else x,
            Tags.capFirstThird: lambda x: x[0].upper() + x[1] + x[2].upper() + x[3:] if len(x) >= 3 else x,
            Tags.allCap: lambda x: x.upper(),
            Tags.capExLast: lambda x: x[:-1].upper() + x[-1] if len(x) >= 1 else x,
            Tags.capFirstTwo: lambda x: x[:2].upper() + x[2:] if len(x) >= 2 else x,
            Tags.capExceptFirst: lambda x: x[0] + x[1:].upper() if len(x) >= 2 else x,
            Tags.noSpace: lambda x: x,
            Tags.leftSpace: lambda x: " " + x,
            Tags.rightSpace: lambda x: x + " ",
            Tags.allSpace: lambda x: " " + x + " ",

        }
        if save and not os.path.exists(cfg.output_dir+"/format_vocab.txt"):
            os.copy(cfg.previous_dir+"/format_vocab.txt",cfg.output_dir+"/format_vocab.txt")
            #self.save_vocab(cfg.output_dir + "./format_vocab.txt")
    def load_vocab(self,fn):
        if fn and os.path.exists(fn):
            for line in open(fn,encoding='utf-8'):
                line = line.strip("\n")
                self.format_vocab[_run_strip_accents(line.lower())] = line

    def tagging(self,seq):
        label,encode = [],[]
        terms = seq.split(" ")
        for term_id,t in enumerate(terms):
            if t in self.special_tokens: #Modify!!!
                label.append(Tags.noChange)
                encode.append(t)
            else:
                for tks,pos in self.tokenizer._tokenizer.pre_tokenizer.pre_tokenize_str(t):
                    #print(tks,pos)
                    if len(tks) == 1 and _is_punctuation(tks):
                        if term_id != 0 and len(t) == 1 and term_id != len(terms)-1:
                            label.append(Tags.allSpace)
                        elif term_id != 0 and pos[0] == 0:
                            label.append(Tags.leftSpace)
                        elif term_id != len(terms)-1 and pos[1] == len(t):
                            label.append(Tags.rightSpace)
                        else:
                            label.append(Tags.noSpace)
                        encode.append(tks)

                    else:
                        wps = self.tokenizer._tokenizer.encode(tks,add_special_tokens=False).tokens
                        #print(wps)
                        l = self._token_label(wps,tks)
                        label.extend(l)
                        encode.extend(wps)
        if self.debug:
            logger.info(encode)
            logger.info(label)
        assert len(encode) == len(label)
        return {"input_ids":[i if i < self.tokenizer.vocab_size else i % self.tokenizer.vocab_size + 1 for i in self.tokenizer.convert_tokens_to_ids(encode)],
                "attention_mask": [1] * len(label),
                "labels": label
                }
    
    def _token_label(self,wps,tks):
        recover = "".join(wps[i] if i == 0 else wps[i][2:] for i in range(0,len(wps)))
        expected_len = len(wps)
        #01. Word piece processor
        if self.format_level == "wordpiece":
            tmp = []
            idx = 0
            for i,t in enumerate(wps):
                t = t[2:] if i != 0 else t
                for ops in self.format_ops:
                    if self.format_ops[ops](t) == _run_strip_accents(tks[idx:idx+len(t)]):
                        tmp.append(ops)
                        idx += len(t)
                        break
                    if len(tmp) < i:
                        break
            if idx == len(tks):
                return tmp
        #02. Whole word processor
        else:
            for ops in self.format_ops:
                if self.format_ops[ops](recover) == tks:
                    return [ops]*expected_len #+ [Tags.noChange] * (expected_len-1)
        
        #03. Lookup dictionary - pending implementation
        
        self.new_vocab[recover][tks] += 1
        
        return [Tags.lookUp] * expected_len


    def realize(self,source,labels):
        #print(source,labels)
        #tokenized_idx =self.tokenizer.encode(source)[1:-1]
        #recover_tokens = self.tokenizer.convert_ids_to_tokens(tokenized_idx)
        #print(recover_tokens)
        recover_tokens = []
        terms = source.split(" ")
        for term in terms:
            if term in self.special_tokens:
                recover_tokens.append(term)
                continue
            term = term.lower()
            for tks,pos in self.tokenizer._tokenizer.pre_tokenizer.pre_tokenize_str(term):
                idx = 0
                if self.debug:
                    logger.debug(self.tokenizer._tokenizer.encode(tks,add_special_tokens=False).tokens)
                for t in self.tokenizer._tokenizer.encode(tks,add_special_tokens=False).tokens:
                    if t == "[UNK]":
                        recover_tokens.append(tks)
                        continue
                    prefix = "##" if idx else ""
                    l = len(t)-2 if idx else len(t)
                    recover_tokens.append(prefix + tks[idx:idx+l])
                    idx += l
        if self.debug:
            logger.debug(recover_tokens)
        res = ""
        space = False
        i = 0
        while i < len(recover_tokens):
            label,token = labels[i],recover_tokens[i]
            if space and not token.startswith("##") and label <= 10:
                res += " "
            if label == Tags.lookUp and not recover_tokens[i].startswith("##"):
                i += 1
                while i < len(recover_tokens) and recover_tokens[i].startswith("##"):
                    token += recover_tokens[i][2:]
                    i += 1
                res += self.format_vocab.get(token,token)
            else:
                i += 1
                if self.format_level == "term":
                    while i < len(recover_tokens) and recover_tokens[i].startswith("##"):
                        token += recover_tokens[i][2:]
                        i += 1
                token = token[2:] if token.startswith("##") else token
                if label in self.format_ops:
                    res += self.format_ops[label](token)
                else:
                    res += token
            space = True if label <= Tags.lookUp else False
        return res.replace("  "," ")

    def save_vocab(self,fn):
        of = open(fn,'w')
        for term in self.new_vocab:
            of.write(max(self.new_vocab[term])+"\n")
        of.close()


def generate_dictionary(file,idx,out_dir,formatter):
    A = formatter
    allcnt,lookup = 0,0
    for line in open(file):
        line = line.rstrip("\n").split("\t")[idx]
        line = " ".join(t for t in line.split(" ") if t)
        encdec = A.realize(line,A.tagging(line)["labels"])
        if unicodedata.normalize("NFKD",line) != encdec and line != encdec:
            lookup += 1
            logger.warning(line)
            logger.warning(encdec)
        allcnt += 1
    A.save_vocab(out_dir+"/format_vocab.txt")
    logger.info("Build Dictionary Done, Lookup Rate: " + str(lookup) + "," + str(allcnt) + "," + str(lookup / allcnt))

if __name__ == "__main__":
    import argparse
    logging.set_verbosity_info()
    logging.enable_explicit_format()
    import logging as local_logging
    local_logging.basicConfig(format="[%(levelname)s|%(filename)s:%(lineno)s] %(asctime)s >> %(message)s",level=logging.INFO)
    parser = argparse.ArgumentParser(description="Formatter Dictionary Generation")
    parser.add_argument('--tokenizer',default='bert-base-uncased',help="tokenizer to use")
    parser.add_argument('--output_dir',default='./',help="output dir")
    parser.add_argument('--input_file',default='',help="input file")
    parser.add_argument('--column',default=0,type=int,help="target column")
    parser.add_argument('--debug',default=False,action="store_true",help="debug mode")
    parser.add_argument('--special_tokens',default="",help="add special token")
    parser.add_argument('--format_level',default="wordpiece",help="format level: term/wordpiece")
    args = parser.parse_args()
    #Test
    import sys
    class Config():
        def __init__(self):
            self.cache_dir = None
            self.previous_dir = args.output_dir
            self.output_dir = args.output_dir
            self.format_level = args.format_level
            self.special_tokens=args.special_tokens
    if not tf.io.gfile.exists(args.output_dir):
        tf.io.gfile.makedirs(args.output_dir)
    cfg = Config()
    A = Formatter(args.tokenizer,cfg,debug=args.debug,save=False)
    cases = ["Amanda likes AutoEncoder",
             "Hello! World!!",
             "##hashtag",
             "Mid-term exam is comming soon!",
             "HaHaHa!!!!!",
             "I want to BuyiPhone",
             "How about SUVs",
             "macBook, iPhone, iPad, AppleWatch",
             "Neareast mcDonalds!",
             "amaNDA^_^ CoOL!!",
             "VetriScience For Dogs",
             "TravelSouthDakota.com",
             "WorldWide",
             "Lässige und elegante Hosen in vielen Schnitten bei heine!",
             "N'attends plus, commande des tenues ou d'autres articles pour homme !",
             "aBC !",
             "Een nieuwe vorm van esthetiek. Ontdek de eerste officiële foto´s.",
             "Reinventare l'automazione. Disponibile ora !",
             "Le etichette di spedizione GLS Etichette ad un prezzo competitivo.",
             "Dynamo´s voor alle auto types en van diverse merken.",
             "Download onze gratis software, en laat je creativiteit de vrije loop.",
             "Alle de gemakkelijkste manieren om naar het Hotel en Disneyland®Paris.",
             "Ontwerp Nu Je Eigen Vans® Gepersonaliseerde Look Online!",
             "Håndværkerveste i Lækker Kvalitet. Alle De Kendte Mærker - Køb Her.",
             "Amanda [DoNotSplit] !^_^"]
    logger.info("Pass 1:")
    for c in cases:
        res = A.realize(c,A.tagging(c)["labels"])
        label = "[Wrong]" if c != res else "[Right]"
        logger.info(label + "\n" + c + "\n" + res)
    A.save_vocab(args.output_dir + "/format_vocab.txt")
    if args.input_file:
        generate_dictionary(args.input_file,args.column,args.output_dir,A)
    logger.info("Pass 2:")
    A = Formatter(args.tokenizer,cfg,debug=args.debug,save=False)
    for c in cases:
        res = A.realize(c,A.tagging(c)["labels"])
        label = "[Wrong]" if c != res else "[Right]"
        logger.info(label + "\n" + c + "\n" + res)
