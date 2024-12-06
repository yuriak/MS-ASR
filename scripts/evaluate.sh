batch_job_file="../data/video_series_all.json"
result_root="../output"
dataset_path="../dataset/conference/2/acl_6060"
vl_model="THUDM/cogagent-vqa-hf"
analyze_mode="general_qa"
pe_mode="visual"
device="cuda:0"


vl_model_name_=`echo ${vl_model} | cut -d'/' -f2`
vl_model_name_=${vl_model_name_//-/_}
result_dir="${result_root}/${vl_model_name_}-${analyze_mode}-${pe_mode}/"
eval_output_dir="../output/${vl_model_name_}-${analyze_mode}-${pe_mode}/"
mkdir $eval_output_dir -p


# Standard evaluation without SWER
python ../src/evaluation/evaluate.py \
    --tmp_dir ../tmp \
    --dataset_path $dataset_path \
    --result_dir $result_dir \
    --output_dir $eval_output_dir \
    --split all



# Evaluation with SWER
# openai_api_key="<api_key>"
# python ../src/evaluation/evaluate.py \
#     --tmp_dir ../tmp \
#     --dataset_path $dataset_path \
#     --result_dir $result_dir \
#     --output_dir $eval_output_dir \
#     --split all \
#     --compute_swer \
#     --annotator gpt-4o \
#     --openai_api_key $openai_api_key

