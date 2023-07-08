
exp_name=rl
output_dir_path=eval_output
model_dir_path=model_output
dataset=cnndm
export _ROUGE_PATH=./ROUGE-RELEASE-1.5.5
export CLASSPATH=./stanford-corenlp-3.8.0.jar
bash evaluation/run_test_cnndm.sh ${exp_name} ${output_dir_path} ${model_dir_path}
python extract_prediction.py --dataset ${dataset} --exp_name ${exp_name} --output_dir_path ${output_dir_path}
cat ${output_dir_path}/${dataset}/${exp_name}/pred.txt | java edu.stanford.nlp.process.PTBTokenizer -ioFileList -preserveLines > ${output_dir_path}/${dataset}/${exp_name}/pred.txt.token
cat ${output_dir_path}/${dataset}/${exp_name}/ref.txt | java edu.stanford.nlp.process.PTBTokenizer -ioFileList -preserveLines > ${output_dir_path}/${dataset}/${exp_name}/ref.txt.token
python cal_rouge.py --ref  ${output_dir_path}/${dataset}/${exp_name}/ref.txt.token --hyp ${output_dir_path}/${dataset}/${exp_name}/pred.txt.token
python cal_rouge.py --ref  ${output_dir_path}/${dataset}/${exp_name}/ref.txt.token --hyp ${output_dir_path}/${dataset}/${exp_name}/pred.txt.token -p
