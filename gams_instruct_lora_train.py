from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig, get_peft_model, TaskType  
from argparse import ArgumentParser
import os

def use_lora(model, rank=128):
    if rank == 0:
        return model
    
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=rank,
        lora_alpha=2*rank,
        lora_dropout=0.1,
        target_modules=[
            "q_proj", 
            "v_proj", 
            "k_proj", 
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    model = get_peft_model(model, lora_config)
    return model

def print_trainable_parameters(model):
    trainable_params = 0
    all_params = 0

    for param in model.parameters():
        all_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()

    print(f"Total Parameters: {all_params/1e6:.2f}M")
    print(f"Trainable Parameters (LoRA): {trainable_params/1e6:.2f}M")
    print(f"Percentage of Trainable Params: {(trainable_params/all_params) * 100:.4f}%")

def run_training(experiment_dir, model_output_path, model_input_path, run_name, resume_from_checkpoint=None):
    
   
    tokenizer = AutoTokenizer.from_pretrained(model_input_path)
    if tokenizer.pad_token is None:
        print("Warning: Pad token is not set, setting it to the eos token")
        tokenizer.pad_token = tokenizer.eos_token
    

    print("Loading dataset in streaming mode...")
    dataset = load_dataset(
        "json", 
        data_files="/shared/home/alenka.zumer/dataset_finetuning/training_data.jsonl",
        split="train",
        streaming=True  
    )
    

    train_val_split = dataset.train_test_split(test_size=0.05, seed=42)
    train_dataset = train_val_split["train"]
    val_dataset = train_val_split["test"]

 
    def tokenize_example(example):
        """Tokenize example - system prompt je že v training_data.jsonl"""
        prompt_ids = tokenizer.apply_chat_template(
            example["prompt"],
            add_generation_prompt=True
        )
        prompt_completion_ids = tokenizer.apply_chat_template(
            example["prompt"] + example["completion"]
        )
        
       
        completion_mask = [0] * len(prompt_ids) + [1] * (len(prompt_completion_ids) - len(prompt_ids))
        
        return {
            "input_ids": prompt_completion_ids,
            "completion_mask": completion_mask
        }
    

    print("Setting up streaming tokenization...")
    train_dataset = train_dataset.map(
        tokenize_example,
        remove_columns=["prompt", "completion"]  
    )
    val_dataset = val_dataset.map(
        tokenize_example,
        remove_columns=["prompt", "completion"]
    )


    micro_batch_size = 2  
    batch_size = 32 
    gradient_accumulation_steps = batch_size // micro_batch_size  

    print(f"Micro batch size: {micro_batch_size}")
    print(f"Effective batch size: {batch_size}")
    print(f"Gradient accumulation steps: {gradient_accumulation_steps}")


    num_epochs = 3
    eval_steps = 500  
    save_steps = 500
    warmup_steps = 500

    print("--------------------------------")
    print("Training parameters:")
    print(f"Run name: '{run_name}'")
    print(f"Num epochs: {num_epochs}")
    print(f"Evaluate each {eval_steps} steps")
    print(f"Save each {save_steps} steps")
    print(f"Warmup steps: {warmup_steps} steps")
    print("--------------------------------")
    

    training_args = SFTConfig(
        num_train_epochs=num_epochs,
        eval_strategy="steps",
        save_strategy="steps",
        save_steps=save_steps,
        eval_steps=eval_steps,
        save_total_limit=3,
        logging_strategy="steps",
        logging_dir=f"{experiment_dir}/logs",
        logging_first_step=True,
        output_dir=experiment_dir,
        prediction_loss_only=True,
        metric_for_best_model="eval_loss",
        load_best_model_at_end=True,
        greater_is_better=False,
        report_to="wandb",
        run_name=run_name,
        logging_steps=10,
        per_device_train_batch_size=micro_batch_size,
        per_device_eval_batch_size=micro_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=1e-6,
        warmup_steps=warmup_steps,
        weight_decay=0.1,
        adam_beta1=0.9,
        adam_beta2=0.95,
        adam_epsilon=1e-5,
        lr_scheduler_type="cosine_with_min_lr",
        lr_scheduler_kwargs={"min_lr": 1e-7},
        bf16=True,
        dataloader_num_workers=4,  
        push_to_hub=False,
        gradient_checkpointing=True,
        max_seq_length=2048, 
        completion_only_loss=True,
    )
    

    print("Initializing model")
    model = AutoModelForCausalLM.from_pretrained(model_input_path, attn_implementation='eager')
    print("Model initialized")
    

    print("Applying LoRA...")
    model = use_lora(model, rank=128)
    print_trainable_parameters(model)


    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset
    )


    if resume_from_checkpoint:
        print(f"Resuming training from checkpoint: {resume_from_checkpoint}")
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    else:
        print("Starting training from scratch")
        trainer.train()

    # Save model
    model.save_pretrained(model_output_path)
    tokenizer.save_pretrained(model_output_path)
    print(f"Model saved to: {model_output_path}")


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--experiment_dir", type=str, required=True)
    parser.add_argument("--model_input_path", type=str, required=True)
    parser.add_argument("--model_output_path", type=str, required=True)
    parser.add_argument("--run_name", type=str, required=True)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_training(
        args.experiment_dir, 
        args.model_output_path, 
        args.model_input_path, 
        args.run_name, 
        args.resume_from_checkpoint
    )
