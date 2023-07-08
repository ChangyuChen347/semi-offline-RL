""" SQuAD QG evaluation (sentence/answer level) """
import logging
import argparse
import json

from automatic_evaluation import evaluate


logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')


def get_options():
    parser = argparse.ArgumentParser(description='QG evaluation on SQuAD.')
    parser.add_argument('-m', '--model-checkpoint', default=None, type=str)
    parser.add_argument('--batch-size', default=16, type=int)
    parser.add_argument('--n-beams', default=4, type=int)
    parser.add_argument('-e', '--export-dir', required=True, type=str)
    parser.add_argument('--hyp-test', default=None, type=str)
    parser.add_argument('--hyp-dev', default=None, type=str)
    parser.add_argument('--max-length', default=512, type=int, help='')
    parser.add_argument('--max-length-output', default=64, type=int, help='')
    parser.add_argument('-d', '--dataset-path', help='huggingface datasets alias', default='lmqg/qg_squad', type=str)
    parser.add_argument('--dataset-name', help='huggingface datasets name', default='default', type=str)
    parser.add_argument('-i', '--input-type', help='', default='paragraph_answer', type=str)
    parser.add_argument('-o', '--output-type', help='', default='question', type=str)
    parser.add_argument('-l', '--language', help='', default='en', type=str)
    parser.add_argument('--test-split', help='the name of test split', default='test', type=str)
    parser.add_argument('--validation-split', help='the name of validation split', default='validation', type=str)
    parser.add_argument('--overwrite', help='', action='store_true')
    parser.add_argument('--bleu-only', help='', action='store_true')
    parser.add_argument('--use-auth-token', help='', action='store_true')
    parser.add_argument('--device-map', help='', default=None, type=str)
    parser.add_argument('--low-cpu-mem-usage', help='', action='store_true')
    parser.add_argument('--prediction-aggregation', default=None, type=str)
    parser.add_argument('--prediction-level', default=None, type=str)
    return parser.parse_args()


def main():
    opt = get_options()
    prediction_aggregation = 'first' if opt.prediction_aggregation is None else opt.prediction_aggregation
    prediction_level = 'sentence'  # qg
    if opt.input_type == 'paragraph' and opt.output_type == 'questions_answers':  # e2e qag
        prediction_level = 'answer'
    elif opt.output_type == 'answer':  # answer extraction, question answering
        prediction_level = 'answer'
    prediction_level = opt.prediction_level if opt.prediction_level is not None else prediction_level
    assert opt.model_checkpoint or opt.hyp_test or opt.hyp_dev
    assert prediction_aggregation in ['first', 'last', 'long', 'short', 'middle']
    metric = evaluate(
        export_dir=opt.export_dir,
        batch_size=opt.batch_size,
        n_beams=opt.n_beams,
        hypothesis_file_dev=opt.hyp_dev,
        hypothesis_file_test=opt.hyp_test,
        model=opt.model_checkpoint,
        max_length=opt.max_length,
        max_length_output=opt.max_length_output,
        dataset_path=opt.dataset_path,
        dataset_name=opt.dataset_name,
        input_type=opt.input_type,
        output_type=opt.output_type,
        prediction_aggregation=prediction_aggregation,
        prediction_level=prediction_level,
        overwrite=opt.overwrite,
        language=opt.language,
        bleu_only=opt.bleu_only,
        use_auth_token=opt.use_auth_token,
        device_map=opt.device_map,
        low_cpu_mem_usage=opt.low_cpu_mem_usage
    )
    print(json.dumps(metric, indent=4, sort_keys=True))

main()
