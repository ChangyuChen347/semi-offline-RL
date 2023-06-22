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
```console
CUDA_VISIBLE_DEVICES=0 python main.py \
    --do_train \
    --scene bart_cnn_generation \
    --use_logit True \
    --report_to tensorboard \
    --seed 2022 \
    --smooth 0.1 \
    --trainer rl \
    --save_steps 10000000 \
    --learning_rate 0.000001 \
    --num_train_epochs 60 \
    --max_grad_norm 1 \
    --print_every 1000 \
    --save_every 4000 \
    --eval_steps 2000 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --per_device_eval_batch_size 16 \
    --length_normalize_4_rl True \
    --training_length_penalty 1 \
    --train_dir sample/cnndm_train.tsv \
    --eval_dir sample/cnndm_test.tsv \
    --cand_pos_remove_sp_tk True \
    --recover cnndm_base_model \
    --exp_name demo_cnndm \
    --rewards 'rouge' \
    --rouge_type 12l \
    --reward_type rouges \
    --rl_weight 20 \
    --sample_num 63 \
    --mask_rate 0.4 \ 
    --kd_inputs_worst True \
    --eval_metrics rouges \ 
```

### Evaluate

We use the PTB tokenizer provided by Standford [CoreNLP](https://stanfordnlp.github.io/CoreNLP/index.html) ([download here](https://repo1.maven.org/maven2/edu/stanford/nlp/stanford-corenlp/3.8.0/stanford-corenlp-3.8.0.jar)). Please note that tokenized texts are *only* used for evaluation.
To tokenize a file, you may run (using test.source as an example)
```console
export CLASSPATH=/your_path/stanford-corenlp-3.8.0.jar
cat test.source | java edu.stanford.nlp.process.PTBTokenizer -ioFileList -preserveLines > test.source.tokenized
```

The evaluation (word tokenization and metric computation) of CNN/DM and XSum is following [BRIO](https://github.com/yixinL7/BRIO) and the evaluation of SQuAD is following [LMQG](https://github.com/asahi417/lm-question-generation).  

For ROUGE calculation, we use the standard ROUGE Perl package from [here](https://github.com/summanlp/evaluation/tree/master/ROUGE-RELEASE-1.5.5) in our paper. We lowercased and tokenized (using PTB Tokenizer) texts before calculating the ROUGE scores. Please note that the scores calculated by this package would be sightly *different* from the ROUGE scores calculated/reported during training/intermidiate stage of evalution, because we use a pure python-based ROUGE implementation to calculate those scores for better efficiency. 

If you encounter problems when setting up the ROUGE Perl package (unfortunately it happens a lot :( ), you may consider using pure Python-based ROUGE package such as the one we used from the [compare-mt](https://github.com/neulab/compare-mt) package.

We provide the evaluation script in `cal_rouge.py`. If you are going to use Perl ROUGE package, please change line 13 into the path of your perl ROUGE package.
```python
_ROUGE_PATH = '/YOUR-ABSOLUTE-PATH/ROUGE-RELEASE-1.5.5/'
```

To evaluate the model performance, please first use the following command to generate the summaries.
```console
python main.py --cuda --gpuid [single gpu] --config [name of the config (cnndm/xsum)] -e --model_pt [model path] -g [evaluate the model as a generator] -r [evaluate the model as a scorer/reranker]
```
model path should be a subdirectory in the `./cache` directory, e.g. `cnndm/model.pt` (it shouldn't contain the prefix `./cache/`).
The output will be saved in a subfolder of `./result` having the same name of the checkpoint folder.

#### Example: evaluating the model as a generator on CNNDM
```console
# write the system-generated files to a file: ./result/cnndm/test.out
python main.py --cuda --gpuid 0 --config cnndm -e --model_pt cnndm/model_generation.bin -g

# tokenize the output file -> ./result/cnndm/test.out.tokenized (you may use other tokenizers)
export CLASSPATH=/your_path/stanford-corenlp-3.8.0.jar
cat ./result/cnndm/test.out | java edu.stanford.nlp.process.PTBTokenizer -ioFileList -preserveLines > ./result/cnndm/test.out.tokenized

# calculate the ROUGE scores using ROUGE Perl Package
python cal_rouge.py --ref ./cnndm/test.target.tokenized --hyp ./result/cnndm/test.out.tokenized -l

# calculate the ROUGE scores using ROUGE Python Implementation
python cal_rouge.py --ref ./cnndm/test.target.tokenized --hyp ./result/cnndm/test.out.tokenized -l -p

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

