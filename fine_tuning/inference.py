from transformers import AutoModelForCausalLM, AutoTokenizer
# from prompt_setting import alpaca_prompt
# import torch
# model_id = "D:\demo\llama3_Knowledge_Graph\output\merge_model"
#
# tokenizer = AutoTokenizer.from_pretrained(model_id)
# model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, load_in_8bit=True, device_map={"": 0})
#
# print(f"Model size: {model.get_memory_footprint():,} bytes")
#
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#
# inputs = tokenizer(
#     [
#         alpaca_prompt.format(
#             "slotcraft含有几个大类？",  # instruction
#             "",  # input
#             "",  # output - leave this blank for generation!
#         )
#     ], return_tensors="pt"
# ).to(device)
#
# outputs = model.generate(**inputs, max_new_tokens=64, use_cache=True)
# print(tokenizer.batch_decode(outputs))
# print(type(tokenizer.batch_decode(outputs)))
#
# print("AI: " + tokenizer.batch_decode(outputs))
#
# while True:
#         test_text = input("Kun Peng：")
#
#         inputs = tokenizer(
#             [
#                 alpaca_prompt.format(
#                     test_text,  # instruction
#                     "",  # input
#                     "",  # output - leave this blank for generation!
#                 )
#             ], return_tensors="pt"
#         ).to(device)
#         if test_text == "exit":
#                 break;
#
#         outputs = model.generate(**inputs, max_new_tokens=64, use_cache=True)
#
#         print("AI: " + tokenizer.batch_decode(outputs))

import transformers
import torch

model_id = "D:\demo\llama3_Knowledge_Graph\output\merge_model"

pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.float16},
    device="cuda",
)

# model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, load_in_8bit=True, device_map={"": 0})
# print(model)

messages = [{"role": "system", "content": ""}]

while True:
    test_text = input("Human：")
    messages.append(
        {"role": "user", "content": test_text}
    )

    prompt = pipeline.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

    terminators = [
            pipeline.tokenizer.eos_token_id,
            pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
    outputs = pipeline(
        prompt,
        max_new_tokens=512,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.6,
        top_p=0.9
    )

    content = outputs[0]["generated_text"][len(prompt):]

    print("AI："+ content)