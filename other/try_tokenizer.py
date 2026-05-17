from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("/data/lqc/Practice/llm/from_zero_to_one/tokenizers/minimind")

prompt = [
        {"role": "user", "content": "你好"},
        {"role": "assistant", "content": "你好！"},
        {"role": "user", "content": "再见"},
        {"role": "assistant", "content": "再见！"}
    ]

prompt = tokenizer.apply_chat_template(prompt,  tokenize = False)
print(prompt)