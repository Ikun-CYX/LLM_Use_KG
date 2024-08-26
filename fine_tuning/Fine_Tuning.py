import argparse
import os
import torch
import pandas as pd
import warnings
from datasets import Dataset
from typing import Optional

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForSeq2Seq,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
    AutoModelForSequenceClassification
)
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model
)

from parse_fine_tuning import parse_fine_tuning
class FineTuning(object):
    def __init__(self,
                 model_name: Optional[str] = None,
                 dataset_name: Optional[str] = None,
                 dataset_dir: Optional[str] = None,
                 output_dir: Optional[str] = None,
                 train_with_checkpoint: Optional[bool] = None,
                 ) -> None:
        if torch.cuda.is_bf16_supported():
            self.compute_dtype = torch.bfloat16
            self.attn_implementation = 'flash_attention_2'
        else:
            self.compute_dtype = torch.float16
            self.attn_implementation = 'sdpa'

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # 模型文件路径
        self.model_name = model_name
        print(self.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, add_eos_token=True, use_fast=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.padding_side = 'left'

        self.output_dir = f"{output_dir}/{dataset_name}"
        self.dataset_dir = f"{dataset_dir}/{dataset_name}.json"

        self.train_with_checkpoint = train_with_checkpoint

        # 加载模型
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=self.compute_dtype,
            bnb_4bit_use_double_quant=True,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, quantization_config=bnb_config, device_map={"": 0}, attn_implementation=self.attn_implementation
        )

    def Load_Dataset(self):

        df = pd.read_json(self.dataset_dir)
        ds = Dataset.from_pandas(df)
        inputs_id = ds.map(self.process_func_mistral, remove_columns=ds.column_names)
        return inputs_id

    def process_func_mistral(self, example):
        MAX_LENGTH = 384  # Llama分词器会将一个中文字切分为多个token，因此需要放开一些最大长度，保证数据的完整性
        instruction = self.tokenizer(
            f"<s>[INST] <<SYS>>\n\n<</SYS>>\n\n{example['instruction'] + example['input']}[/INST]",
            add_special_tokens=False)  # add_special_tokens 不在开头加 special_tokens
        response = self.tokenizer(f"{example['output']}", add_special_tokens=False)
        input_ids = instruction["input_ids"] + response["input_ids"] + [self.tokenizer.pad_token_id]
        attention_mask = instruction["attention_mask"] + response["attention_mask"] + [
            1]  # 因为pad_token_id咱们也是要关注的所以 补充为1
        labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [self.tokenizer.pad_token_id]

        if len(input_ids) > MAX_LENGTH:  # 做一个截断
            input_ids = input_ids[:MAX_LENGTH]
            attention_mask = attention_mask[:MAX_LENGTH]
            labels = labels[:MAX_LENGTH]
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }
    def Train(self):

        self.model.enable_input_require_grads()  # 开启梯度检查点时，要执行该方法
        config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            inference_mode=False,  # 训练模式
            r=8,  # Lora 秩
            lora_alpha=32,  # Lora alaph，具体作用参见 Lora 原理
            lora_dropout=0.1  # Dropout 比例
        )

        self.model = get_peft_model(self.model, config)

        args = TrainingArguments(
            output_dir=self.output_dir,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=2,
            logging_steps=10,
            num_train_epochs=20,
            save_steps=10,
            save_total_limit=5,
            learning_rate=1e-4,
            save_on_each_node=True,
            gradient_checkpointing=True
        )

        inputs_id = self.Load_Dataset()

        print(inputs_id)

        trainer = Trainer(
            model=self.model,
            args=args,
            train_dataset=inputs_id,
            data_collator=DataCollatorForSeq2Seq(tokenizer=self.tokenizer, padding=True),
        )

        warnings.filterwarnings("ignore", category=FutureWarning)

        # 如果训练中断了，还可以从上次中断保存的位置继续开始训练
        if self.train_with_checkpoint:
            checkpoint = [file for file in os.listdir(self.output_dir) if 'checkpoint' in file][-1]
            last_checkpoint = f'{self.output_dir}/{checkpoint}'
            print(last_checkpoint)
            trainer.train(resume_from_checkpoint=last_checkpoint)
        else:
            trainer.train()

    def checkpoint_to_lora(
            self,
            lora_path: Optional[str] = None,
            model_path: Optional[str] = None,
    ) -> None:
        lora_path = "D:\demo\llama3_Knowledge_Graph\output\lora\Llama3-Chinese-8B-Instruct_slotcraft"

        # 模型路径
        model_path = "FlagAlpha/Llama3-Chinese-8B-Instruct"

        # 检查点路径
        checkpoint_dir = "D:\demo\llama3_Knowledge_Graph\output\dataset"

        checkpoint = [file for file in os.listdir(checkpoint_dir) if 'checkpoint-' in file][-1]  # 选择更新日期最新的检查点

        model = AutoModelForSequenceClassification.from_pretrained(
            f'D:\demo\llama3_Knowledge_Graph\output\dataset/{checkpoint}')
        # 保存模型
        model.save_pretrained(lora_path)

        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
        # 保存tokenizer
        tokenizer.save_pretrained(lora_path)

def main():

    warnings.filterwarnings("ignore", category=UserWarning)  # 忽略告警

    config = vars(parse_fine_tuning())

    FT = FineTuning(
        model_name=config['model_name'],
        dataset_name=config['dataset_name'],
        dataset_dir=config['dataset_dir'],
        output_dir=config['output_dir'],
        train_with_checkpoint=config['train_with_checkpoint'],
    )
    FT.Train()
    # FT.checkpoint_to_lora()


if __name__ == '__main__':
    main()