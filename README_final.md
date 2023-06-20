# Semi-Offline Reinforcement Learning for Optimized Text Generation
## Usage

# Python Install
```
git clone https://github.com/ChangyuChen347/semi-offline-RL
cd semi-offline-RL
pip install -r requirement.txt
```

### Train
```console
bash run_cnn.sh
```

### Evaluate
The evaluation (word tokenization and metric computation) of CNN/DM and XSum is following [BRIO](https://github.com/yixinL7/BRIO) and the evaluation of SQuAD is following [LMQG](https://github.com/asahi417/lm-question-generation).  
### Checkpoints and static datasets
|        | BASE (M-FT)                 | RL                        | 
|--------|-----------------------------|---------------------------|
| CNN/DM | [cnndm_bart_base_model]()   | [cnndm_bart_rl_model]()   | 
| SAMSum | [samsum_bart_base_model]()  | [samsum_bart_rl_model]()  |  
| SQuAD  | [t5_squad_base_model]()     | [t5_squad_rl_model]()     |   
| XSum   | [xsum_pegasus_base_model]() | [xsum_pegasus_rl_model]() |  

|        | Train                | validation         | Test                |
|--------|----------------------|--------------------|---------------------|
| CNN/DM | [cnn_train.tsv]()    | [cnn_val.tsv]()    | [cnn_test.tsv]()    |
| SAMSum | [samsum_train.tsv]() | [samsum_val.tsv]() | [samsum_test.tsv]() |
| SQuAD  | [squad_train.tsv]()  | [squad_val.tsv]()  | [squad_test.tsv]()  |
| XSum   | [xsum_train.tsv]()   | [xsum_val.tsv]()   | [xsum_test.tsv]()   |

### The format of the dataset

|        | SRC                                    | Groundtruth                          | Decoded results                                                                    |
|--------|----------------------------------------|--------------------------------------|------------------------------------------------------------------------------------|
| CNN/DM | By. Sara Malm. PUBLISHED:. 17:34 ES... | Fehmina Chaudhry was kidnapped as... | Body of Fehmina Chaudhry ... <#SCORE#>0.5459186331197582<#SEP#>Fehmina Chaudhry... |

