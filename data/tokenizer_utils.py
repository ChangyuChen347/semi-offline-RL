from transformers import AutoTokenizer
import torch
def prepare_tokenizer(model, cache_dir, **kwargs):
    special_tokens = kwargs.pop("special_tokens", None)
    if special_tokens:
        special_tokens = special_tokens.split(",")
    #auto_tokenizer = AutoTokenizer.from_pretrained(model, cache_dir=cache_dir, additional_special_tokens=special_tokens)
    # auto_tokenizer = AutoTokenizer.from_pretrained(model, cache_dir=cache_dir, additional_special_tokens=special_tokens)
    if model == 'Yale-LILY/brio-xsum-cased':
        model = 'google/pegasus-xsum'
    if model == 'Yale-LILY/brio-cnndm-uncased':
        model = 'facebook/bart-large-cnn'
    print('model', model)
    auto_tokenizer = AutoTokenizer.from_pretrained(model, cache_dir=cache_dir)
    print('len(auto_tokenizer)', len(auto_tokenizer))
    auto_tokenizer.add_tokens(special_tokens)
    print(len(auto_tokenizer))
    print(special_tokens)
    return auto_tokenizer