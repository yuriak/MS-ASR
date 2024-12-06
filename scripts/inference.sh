batch_job_file="../data/video_series_all.json"
result_root="../output"
vl_model="THUDM/cogagent-vqa-hf"
analyze_mode="general_qa"
pe_mode="visual"
device="cuda:0"


# Please use deploy_vllm_server.sh first to deploy the server
api_address="http://<ip:port>/v1"
api_key="<api_key>"
llm_type="api"
llm_in_use="meta-llama/Meta-Llama-3-70B-Instruct"


vl_model_name_=`echo ${vl_model} | cut -d'/' -f2`
vl_model_name_=${vl_model_name_//-/_}
output_dir="${result_root}/${vl_model_name_}-${analyze_mode}-${pe_mode}/"
mkdir $output_dir -p
echo ${output_dir}
mkdir -p ${output_dir}
python ../src/scivasr/cli.py \
    --llm_reasoner_model $llm_in_use \
    --api_key $api_key \
    --api_address $api_address \
    --llm_type $llm_type \
    --video_batch_job $batch_job_file \
    --analyzer_device ${device} \
    --feature_extractor_device ${device} \
    --analyzer_type local \
    --analyzer_model $vl_model \
    --post_edit_mode $pe_mode \
    --page_level_analyze_mode $analyze_mode \
    --output_dir $output_dir