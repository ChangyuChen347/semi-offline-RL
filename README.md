## Usage
* Command: `accelerate launch --multi_gpu --num_processes [n_gpu] --num_machines 1 main.py --scene [scene_name]`
* Parameter Priority: command line > config/SceneConfigs > config/ArgmentClass
* Data Preprocess Header Schema: --train_model_header, --eval_model_header, --predict_model_header, cols split by ':'
    - num_col = 1：input/output name (Example:`--train_model_header query`)
    - num_col = 2: preprocessor, input/output name (Example: `--train_model_header convert2floatlist:labels`)
    - num_col >= 3: preprocessor, input0, input1,..., output name (Example: `--train_model_header cls_tokenize:query:doc:input`)
* Evaluation Metrics to Use `--eval_metrics a,b,c`, the first metric is used to select best model
* Output Schema
    - `--result_header` to set all the fields want to be kept, also append then in `--predict_model_header`
    - `--output_type` controls the postprocessing function applied to the prediction result
    - Example refer to config/SceneConfgs/bart_generation
* Setting
    - `--datareader_streaming` will enable the streaming input mode, which will not index and shuffle the data
    - `--dataloader_num_workers` will control input multiprocessing workers and output multiprocessing workers
    - `--outputter_num_workers` will override `dataloader_num_workers` for output multiprocessing workers if set. 0 is for not using multiprocess 
    - It support n gpu (n>=2) inference, and it will output n files naming `result_file + "_" + (0...n-1)`. The last batchs are filled with duplicate data to avoid potential issue, dedup is needed after merge. 
* Build Docker and Push to DeepGen Container
    - Create a branch base on the master branch, branch name: `branch_name`
    - Create a Docker file or Modify the docker file: ./docker/Dockerfile_`branch_name` -> Please note that the suffix should be exact the same with branch name
    - Create a Pull Request to push the new branch to master
    - In the PR page, run optional check "Build DeepGen Docker"
    - If the pipeline successed, the docker can be access with: Registry: deepgen.azurecr.io, DockerImage: deepgen.azurecr.io/deepgen:`branch_name`
## Support List
* Model Task
  - SequenceClassification (HF Native, class: AutoModelForSequenceClassification) `--task seq2label`
    - Regression: `num_labels = 1, loss = MSELoss(logits,labels)`
    - Single label classification: `num_labels > 1, labels.dtype==long/int, loss = CrossEntropyLoss(logits,labels)` 
    - Multi label classification: `num_labels > 1, labels is vector, loss = BCEWithLogitsLoss(logits,labels)`
  - Sequence2Sequence (HF Native, class: AutoModelForSeq2SeqLM) `--task seq2seq`
  - Customize Tasks
    - Sequence2Sequence with latent variable `--task guided_seq2seq`
* Pretrain Model
  - Model list could be found in [Huggingface Models](https://huggingface.co/models)
* Evaluation Metrics
  - Classification
    - Macro Avg AUC: `auc`
    - AUC of each class: `auc_list`
    - Accuracy: `acc`
  - Geneartion
    - Bleu-4: `bleu` 
    - Rouge: `rouge`
* Data Preprocessor
  - Basic Preprocessor
    - convert2int
    - convert2float
    - convert2floatlist: assuming floatlist in format: 0.5,0.2,0.3
    - label_mapping: `--label_mapping Good:1,Bad:0,Fair:0.5`
  - Tokenization Preprocessor
    - cls_tokenize: `cls_tokenize:query:doc:input, cls_tokenize:query:input`
    - twinbert_tokenize: `twinbert_tokenize:query:doc:input, twinbert_tokenize:query:input`
    - s2s_tokenize: `s2s_tokenize:query:doc:input, s2s_tokenize:query:input`

## Development Guide
* Before pushing to dev - run e2e test: `bash tests/test.sh`
* Adding a Model - If AutoClasses/Huggingface model is not support
  - Add a Model Task which can load different pretrain models
    - Refer to models/seq2softlabel.py
    - Add a task in ModelArguments.py, refer to model_seq2softlabel
  - Add a Model Task supports a specific pretrained models
    - Refer to models/unilm.py or BartForLatentGuidedGeneration in models/bart.py
    - Decorate the main class with `@register_model("task_name","ModelConfig")`
    - Add the task to ModelArguments.py, with no `auto_model` member
  - Add a pretrained model not support by Huggingface
    - Refer to models/twinbert.py
    - Add a model config class inherit PretrainedConfig in models/model_to_add.py
    - In config/ArgumentClass/ModelArguments.py, add it to configuration_auto.Config_MAPPING, tokenization_auto.TOKENIZER_MAPPING
    - Implement different task for the new model
* Customize Trainer
  - Add a trainer class in `trainer/Trainer`
  - Register the class with decorator `@register_trainer`, the original trainer is given name as `common` as in BasicArguments.py.
* New Input Preprocessor
  - Add a class to data/Processor folder, refer to data/Processor/BasicProcessor.py
  - Register the class with `@register_processor("processor_name_used_in_data_preprocssor_command")`
  - Input variables
    - `idx`: a list of a scalar of indexes of input columns
    - `columns`: all the column of a single sample
* Training Metrics
  - Add a class to trainer/Metrics folder
  - Register the class with `@register_metrics("metrics_name_to_use_in_eval_metrics_command")` 
  - Input: EvalPredict(predictions, labels_ids)
* Outputter
  - Add a class to trainer/Outputter folder
  - Register the class with `@register_outputter("outputter_name_to_use_in_output_type")`
* Replace HF Native model/module/any class
  - Refer to models/t5.py
  -  `from config.decoration import replace`
  - Decorate the class with `@replace("class_to_replace")`
* Add your scene to e2e test
  - Add sample data to ./sample folder
  - Config Scene file to be able to run diretely
  - Add the scene name to tests/test.sh

## ITP Features
* Viewing training metrics from ITP
  - Click "job name" link as shown below from ITP portal to go to Azure ML page
![ITP portal](docs/ITP_portal.png)

  - Click "Metrics" tab to view training metrics
![AML metrics](docs/AML_metrics.png)

  - **note:** integration with Azure ML metrics only works when you start python without sudo.

* Run job on ITP without sudo
  - When you start a Python process on ITP without sudo, sometimes it fail because that the process cannot access Huggingface's cache folder, which is /home/youralias/.cache
  - The trick to solve this issue is to mount Azure blob to this path when you start the ITP job
  - See the example below and Aether module: 3ca95749-4fa3-459b-b310-e4542265d21b
![Aether ITP example](docs/MountPathExample.png)
