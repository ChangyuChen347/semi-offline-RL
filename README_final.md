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
The evaluation of CNN/DM and XSum is following [BRIO](https://github.com/yixinL7/BRIO). The evaluation 
### Checkpoints and static datasets
|        | BASE (M-FT) | RL | dataset |
|--------|-------------|----|---------|
| CNN/DM | [BART]()    |    |         |
| SAMSum | [BART]()    |    |         |
| SQuAD  | [T5]()      |    |         |
| XSum   | [PEGASUS]() |    |         |

|        | Train       | validation | Test |
|--------|-------------|------------|------|
| CNN/DM | [BART]()    |            |      |
| SAMSum | [BART]()    |            |      |
| SQuAD  | [T5]()      |            |      |
| XSum   | [PEGASUS]() |            |      |

### The format of the dataset

|        | SRC                                    | Groundtruth                          | Decoded results                                                                    |
|--------|----------------------------------------|--------------------------------------|------------------------------------------------------------------------------------|
| CNN/DM | By. Sara Malm. PUBLISHED:. 17:34 ES... | Fehmina Chaudhry was kidnapped as... | Body of Fehmina Chaudhry ... <#SCORE#>0.5459186331197582<#SEP#>Fehmina Chaudhry... |

