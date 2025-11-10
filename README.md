# Subtitle Generation with GaMS-instruct and LoRA Fine-Tuning 

Ta repozitorij vsebuje vse potrebno za LoRA fine-tuning modela **GaMS-2B-Instruct** s paketoma `transformers` in `trl`. Trening se izvaja v Docker kontainerju.

## Vsebina repozitorija

- `Dockerfile` – definicija containerja  
- `.dockerignore` – datoteke, ki se ne kopirajo v container  
- `requirements.txt` – Python paketi  
- `gams_instruct_lora_train.py` – trening skripta  
- README.md – ta dokument  

> ** Učna množica in model nista vključena, slednji se sproti downloada iz HuggingFace platforme.

---

## Zahteve

- Docker Desktop (Mac / Windows / Linux)  
- Python 3.10+ (če je trening izven Dockerja)  
- WandB account za spremljanje treninga  

---

## Ukazi v terminal za zagon in v vrstnem redu

### Oblikovanje kontejnerja

docker build -t subtitles-container .

### Prijava v Wandb projekt

wandb login

(Ključ je možno dobiti na wandb.ai, kjer se po vpisu pomaknemo zgoraj skrajno desno na svoj profil, izberemo API key poglavje in kopiramo ključ, ki ga vidimo na monitorju.)

### Zagon finetuning skripte s parametri

docker run --gpus all -it --rm \
  -v $(pwd)/instruct_training_data3.jsonl:/workspace/instruct_training_data3.jsonl \
  subtitles-container \
  python3 gams_instruct_lora_train.py \
    --experiment_dir ./experiments/run1 \
    --model_input_path "cjvt/GaMS-2B-Instruct" \
    --model_output_path ./models/run1 \
    --data_path instruct_training_data3.jsonl \
    --run_name "experiment_1"


