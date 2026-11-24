modelscope download --dataset kanliu/oscar-en-10k-megatron oscar-en-10k.jsonl --local_dir examples/llama2/data/dataset
modelscope download --model LLM-Research/llama-2-7b special_tokens_map.json tokenizer.json tokenizer_config.json tokenizer.model --local_dir examples/llama2/data/tokenizer

python 3rdparty/Megatron-LM/tools/preprocess_data.py \
    --tokenizer-type HuggingFaceTokenizer \
    --tokenizer-model examples/llama2/data/tokenizer \
    --input examples/llama2/data/dataset/oscar-en-10k.jsonl \
    --output-prefix examples/llama2/data/dataset/oscar-en-10k \
    --append-eod \
    --workers 16