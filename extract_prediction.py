import argparse

parser = argparse.ArgumentParser(description='Parameters')
parser.add_argument("--dataset", type=str, help="squad xsum cnndm samsum", required=True)
parser.add_argument('--exp_name', type=str, help="base/rl/...", required=True)
parser.add_argument('--output_dir_path', type=str, help="eval_output/")


args = parser.parse_args()
path = args.output_dir_path + '/' + args.dataset + '/' + args.exp_name + '/'  # eval_output/samsum/base/

predict_file = path + 'predict.txt'
x = open(predict_file).readlines()
x = [e.strip().split('\t') for e in x]
extract_predction = [e[2] for e in x]
output_file = path + 'pred.txt'
out = open(output_file, 'w')
if args.dataset == 'cnndm' or args.dataset == 'samsum' or args.dataset == 'xsum':
    extract_predction = [e.lower() for e in extract_predction]
out.write('\n'.join(extract_predction))

extract_ref = [e[1] for e in x]
output_file = path + 'ref.txt'
out = open(output_file, 'w')
if args.dataset == 'cnndm' or args.dataset == 'samsum' or args.dataset == 'xsum':
    extract_ref = [e.lower() for e in extract_ref]
out.write('\n'.join(extract_ref))


