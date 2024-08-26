import argparse

def parse_fine_tuning():
    parser = argparse.ArgumentParser()

    # 路径
    parser.add_argument('--model_name', default=r"D:\model_and_weights_cache\modelscope\Meta-Llama-3.1-8B-Instruct", type=str, help='本地模型路径')
    parser.add_argument('--dataset_name', default="dataset", type=str, help='数据集名称')
    parser.add_argument('--dataset_dir', default="./fine_tuning/datasets/slotcraft_hh", type=str, help='数据集路径')
    parser.add_argument('--output_dir', default="./output", type=str, help='lora保存路径')

    #lora配置
    parser.add_argument('--Lora_r', default=8, type=int, help='Lora的秩')
    parser.add_argument('--lora_alpha', default=32, type=int, help='Lora的alaph_权重占比')
    parser.add_argument('--lora_dropout', default=0.1, type=int, help='Dropout的比例')

    #训练配置
    parser.add_argument('--per_device_train_batch_size', default=2, type=int, help='每个训练设备上的批量大小')
    parser.add_argument('--gradient_accumulation_steps', default=2, type=int, help='梯度累积步骤，用于更大的批次训练')
    parser.add_argument('--logging_steps', default=5, type=int, help='定义多少个更新步骤打印一次训练日志。')
    parser.add_argument('--num_train_epochs', default=5, type=int, help='训练的轮数')
    parser.add_argument('--save_steps', default=5, type=int, help='定义多少个更新步骤保存一次模型')
    parser.add_argument('--save_total_limit', default=5, type=int, help='保存的最大模型数量')
    parser.add_argument('--learning_rate', default=5, type=int, help='学习率')
    parser.add_argument('--save_on_each_node', default=5, type=int, help='保存的最大模型数量')
    parser.add_argument('--gradient_checkpointing', default=5, type=int, help='保存的最大模型数量')


    parser.add_argument('--train_with_checkpoint', action='store_true', help='是否从上次断点处接着训练')
    config = parser.parse_args()
    return config