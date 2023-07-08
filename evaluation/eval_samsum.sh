exp_name=rl
output_dir_path=eval_output
model_dir_path=model_output
dataset=samsum
export _ROUGE_PATH=./ROUGE-RELEASE-1.5.5
bash evaluation/run_test_samsum.sh ${exp_name} ${output_dir_path} ${model_dir_path}
python extract_prediction.py --dataset ${dataset} --exp_name ${exp_name} --output_dir_path ${output_dir_path}
python cal_rouge.py --ref  ${output_dir_path}/${dataset}/${exp_name}/ref.txt --hyp ${output_dir_path}/${dataset}/${exp_name}/pred.txt
python cal_rouge.py --ref  ${output_dir_path}/${dataset}/${exp_name}/ref.txt --hyp ${output_dir_path}/${dataset}/${exp_name}/pred.txt -p
