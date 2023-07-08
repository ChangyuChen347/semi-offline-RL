exp_name=rl2
output_dir_path=eval_output
model_dir_path=model_output
dataset=squad
bash evaluation/run_test_squad.sh ${exp_name} ${output_dir_path} ${model_dir_path}
python extract_prediction.py --dataset ${dataset} --exp_name ${exp_name} --output_dir_path ${output_dir_path}
rm -rf ./eval_metrics
python lmqg/model_evaluation.py --hyp-test ${output_dir_path}/${dataset}/${exp_name}/pred.txt -e "./eval_metrics" -d "lmqg/qg_squad" -l "en"
