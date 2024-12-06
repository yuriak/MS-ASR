python -m vllm.entrypoints.openai.api_server \
 --model meta-llama/Meta-Llama-3-70B-Instruct \
 --dtype bfloat16 \
 --api-key scivasr \
 --tensor-parallel-size 2 \
 --host 0.0.0.0 \
 --port 9876 \
 --max-model-len 16384