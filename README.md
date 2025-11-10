## finetuning_subtitle_generation

Ukaz v terminal:
docker run --gpus all -it --rm \
  -v $(pwd)/instruct_training_data3.jsonl:/workspace/instruct_training_data3.jsonl \
  subtitles-container \
  python3 gams_instruct_lora_train.py \
    --model "cjvt/GaMS-2B-Instruct" \
    --data "instruct_training_data3.jsonl"

