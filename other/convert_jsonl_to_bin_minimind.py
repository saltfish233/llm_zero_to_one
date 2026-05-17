import tiktoken
import numpy as np
import json
from tqdm import tqdm
import os
from concurrent.futures import ProcessPoolExecutor
from transformers import AutoTokenizer


def check_bin():
    tokenizer = AutoTokenizer.from_pretrained(
        "/data/lqc/Practice/llm/from_zero_to_one/tokenizers/minimind"
    )
    # ========== 步骤 2：验证聊天模板 ==========
    # 创建测试用的多轮对话消息
    #   包含 system、user、assistant 三种角色，模拟真实的对话场景
    messages = [
        {"role": "system", "content": "你是一个优秀的聊天机器人，总是给我正确的回应！"},
        {"role": "user", "content": "你来自哪里？"},
        {"role": "assistant", "content": "我来自地球"},
    ]
    # 使用聊天模板格式化消息
    #   apply_chat_template 会使用 tokenizer_config.json 中的 chat_template
    #   将消息列表转换为模型输入格式
    new_prompt = tokenizer.apply_chat_template(
        messages,
        # tokenize=False: 直接返回格式化后的字符串，而不是 token ID 序列
        #   这样可以直观地看到格式化结果，便于验证模板是否正确
        tokenize=False,
    )
    # 打印格式化后的结果，应该看到类似：
    #   <|im_start|>system
    #   你是一个优秀的聊天机器人，总是给我正确的回应！<|im_end|>
    #   <|im_start|>user
    #   你来自哪里？<|im_end|>
    #   <|im_start|>assistant
    #   我来自地球<|im_end|>
    print(new_prompt)

    actual_vocab_size = len(tokenizer)
    print("tokenizer实际词表长度：", actual_vocab_size)
    # 预期输出：tokenizer实际词表长度： 6400

    # 编码测试：将文本转换为 token ID 序列
    #   tokenizer() 会返回一个字典，包含 'input_ids'、'attention_mask' 等字段
    new_prompt = "我爱吃屎"
    model_inputs = tokenizer(new_prompt)
    print(model_inputs["input_ids"] + [tokenizer.eos_token_id] * 3)


check_bin()
# main()
