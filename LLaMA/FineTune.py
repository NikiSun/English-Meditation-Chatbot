import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from trl import SFTTrainer, setup_chat_format
from peft import LoraConfig

# Step 1: Load dataset
file_path = "" # Set file path
dataset = load_dataset("json", data_files=file_path, split="train")
print("Dataset loaded successfully!")

# Step 2: Preprocess conversations
def preprocess_conversations(batch):
    concatenated_messages = []
    weights = []

    for messages_batch in batch["messages"]:
        concatenated_text = ""
        weight = 1  # Default weight

        for message in messages_batch:
            concatenated_text += f"{message['role'].capitalize()}: {message['content']}\n"
            if 'weight' in message:
                weight = message['weight']
        
        concatenated_messages.append(concatenated_text.strip())
        weights.append(weight)

    return {"text": concatenated_messages, "weight": weights}

# Apply preprocessing
dataset = dataset.map(preprocess_conversations, batched=True)
print("Dataset preprocessed successfully!")


# Step 3: Check GPU availability
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
torch.cuda.empty_cache()

# Step 4: Load model and tokenizer
model_id = "" # Set model ID
model = AutoModelForCausalLM.from_pretrained(model_id).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Step 5: Set chat format 
model, tokenizer = setup_chat_format(model, tokenizer)

# Step 6: LoRA configuration 
peft_config = LoraConfig(
    lora_alpha=128,
    lora_dropout=0.1,
    r=256,
    bias="none",
    target_modules="all-linear",
    task_type="CAUSAL_LM",
)

# Step 7: Define training arguments. Choose your own hyperparameters.
output_dir = "" # Set output directory
args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=10,
    per_device_train_batch_size=2,  
    gradient_accumulation_steps=16,  
    learning_rate=5e-5, 
    max_grad_norm=0.5,
    warmup_steps=100,  
    weight_decay=0.01,
    lr_scheduler_type='cosine',   
    fp16=True,  
    seed=1,
    logging_steps=10,
    save_steps=500,
    evaluation_strategy="steps",
    eval_steps=500,
    push_to_hub=True,
    report_to="tensorboard"
)


# Step 8: Define the formatting function for dataset preparation
def formatting_func(example):
    # Ensure the text is of type str
    if isinstance(example["text"], str):
        input_text = example["text"]
    elif isinstance(example["text"], list):
        input_text = " ".join(example["text"])  # Convert list of strings to a single string
    else:
        raise ValueError(f"Unexpected input type: {type(example['text'])}")

    # Tokenize the input text with correct keyword arguments
    input_ids = tokenizer(input_text, truncation=True, padding="max_length", max_length=64)["input_ids"]

    
    return {"input_ids": input_ids}


# Apply the formatting function to the dataset
dataset = dataset.map(formatting_func, batched=False)
print("Dataset formatted successfully!")

# Step 9: Set up and train with SFTTrainer
trainer = SFTTrainer(
    model=model,
    args=args,
    train_dataset=dataset,
    peft_config=peft_config,
    tokenizer=tokenizer,
    max_seq_length=128,
    packing=False,  
    formatting_func=formatting_func,
)

# Step 10: Start training
trainer.train()

# Step 11: Save the fine-tuned model
trainer.save_model()
trainer.push_to_hub()

print("Training and model save completed successfully!")
