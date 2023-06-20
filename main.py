
import models
import torch
from transformers import logging
from config.parse_args import *
from data.data_reader import *
import trainer.Trainer as Trainer
logging.set_verbosity_info()
logging.enable_explicit_format()
import logging as local_logging
logger = logging.get_logger(__name__)
local_logging.basicConfig(format="[%(levelname)s|%(filename)s:%(lineno)s] %(asctime)s >> %(message)s",level=logging.INFO)

from data.tokenizer_utils import prepare_tokenizer
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED, as_completed, ALL_COMPLETED


# import time
def main():
    base_args,train_args,model_args,task_args = parse_args()
    print(model_args._name_or_path)

    auto_tokenizer = prepare_tokenizer(model_args._name_or_path, train_args.cache_dir,
                                       special_tokens=train_args.special_tokens)


    train_input, eval_input, predict_input = input_builder(model_args._name_or_path, train_args, task_args,
                                                           auto_tokenizer)

    if hasattr(task_args, 'auto_model'):
        print(task_args.auto_model) #transformers.models.auto.modeling_auto.AutoModelForSeq2SeqLM
    else:
        print(train_args.task)
        print(getattr(models,train_args.task))
        print(identifier(model_args))

    auto_model = task_args.auto_model if hasattr(task_args,'auto_model') else getattr(models,train_args.task)[identifier(model_args)]
    kwargs = {}
    if hasattr(auto_model,'set_cfg'):
        kwargs["customize_cfg"] = task_args
        kwargs["train_cfg"] = train_args

    model = auto_model.from_pretrained(
        model_args._name_or_path,
        from_tf = train_args.from_tf,
        config=model_args,
        cache_dir=train_args.cache_dir,
        **kwargs
        )

    trainer = getattr(Trainer, train_args.trainer)(
        model=model,
        args = train_args,
        model_args = model_args,
        train_dataset = train_input,
        eval_dataset = eval_input if not train_args.do_predict else predict_input,
        task_args = task_args,
        auto_tokenizer=auto_tokenizer
    )

    if train_args.do_train:
        trainer.train()
    if train_args.do_predict:
        trainer.predict()



if __name__ == "__main__":
    #os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    logger.info("checking GPU")
    if not torch.cuda.is_available():
        logger.warning("torch.cuda.is_available() Fail")
    else:
        logger.info("torch.cuda.is_available() Succeed")
    main()
