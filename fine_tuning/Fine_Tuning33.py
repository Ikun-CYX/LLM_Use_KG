from datasets import Dataset
import pandas as pd
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForSeq2Seq,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig
)
import torch
import os
from peft import LoraConfig, TaskType, get_peft_model
import warnings

warnings.filterwarnings("ignore", category=UserWarning)  # 忽略告警

if torch.cuda.is_bf16_supported():
    compute_dtype = torch.bfloat16
    attn_implementation = 'flash_attention_2'
else:
    compute_dtype = torch.float16
    attn_implementation = 'sdpa'

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# 模型文件路径
model_name = "FlagAlpha/Llama3-Chinese-8B-Instruct"
# model_name = "D:\demo\llama3_fine_tuning7.1\merge_model"

# 训练过程数据保存路径
name = 'dataset'
output_dir = f'./output/Mistral-7B-{name}'
# 是否从上次断点处接着训练
train_with_checkpoint = False

# 加载tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, add_eos_token=True, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.padding_side = 'left'

# 加载数据集
df = pd.read_json(f'./fine_tuning/datasets/slotcraft/{name}.json')

ds = Dataset.from_pandas(df)

# 对数据集进行处理，需要将数据集的内容按大模型的对话格式进行处理
def process_func_mistral(example):
    MAX_LENGTH = 384  # Llama分词器会将一个中文字切分为多个token，因此需要放开一些最大长度，保证数据的完整性
    instruction = tokenizer(
        f"<s>[INST] <<SYS>>\n\n<</SYS>>\n\n{example['instruction'] + example['input']}[/INST]",
        add_special_tokens=False)  # add_special_tokens 不在开头加 special_tokens
    response = tokenizer(f"{example['output']}", add_special_tokens=False)
    input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
    attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1]  # 因为pad_token_id咱们也是要关注的所以 补充为1
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.pad_token_id]

    if len(input_ids) > MAX_LENGTH:  # 做一个截断
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

inputs_id = ds.map(process_func_mistral, remove_columns=ds.column_names)
# inputs_id = ds

# 加载模型
bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
)
model = AutoModelForCausalLM.from_pretrained(
          model_name, quantization_config=bnb_config, device_map={"": 0}, attn_implementation=attn_implementation
)

print(model)

print(f"Model size: {model.get_memory_footprint():,} bytes")

model.enable_input_require_grads()  # 开启梯度检查点时，要执行该方法
config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    inference_mode=False,  # 训练模式
    r=8,  # Lora 秩
    lora_alpha=32,  # Lora alaph，具体作用参见 Lora 原理
    lora_dropout=0.1  # Dropout 比例
)


model = get_peft_model(model, config)

model.print_trainable_parameters()

args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=2,
    logging_steps=5,
    num_train_epochs=5,
    save_steps=1,
    save_total_limit=5,
    learning_rate=1e-4,
    save_on_each_node=True,
    gradient_checkpointing=True
)
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=inputs_id,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
)

warnings.filterwarnings("ignore", category=FutureWarning)

# 如果训练中断了，还可以从上次中断保存的位置继续开始训练
if train_with_checkpoint:
    checkpoint = [file for file in os.listdir(output_dir) if 'checkpoint' in file][-1]
    last_checkpoint = f'{output_dir}/{checkpoint}'
    print(last_checkpoint)
    trainer.train(resume_from_checkpoint=last_checkpoint)
else:
    trainer.train()