import tiktoken
import numpy as np
import json
from tqdm import tqdm
import os
from concurrent.futures import ProcessPoolExecutor


def process_line_batch(lines):
    enc = tiktoken.get_encoding("gpt2")
    eos = enc.encode("<|endoftext|>", allowed_special={"<|endoftext|>"})
    batch_tokens = []
    for line in lines:
        try:
            raw = json.loads(line)["text"]
            raw = enc.encode(raw) + eos
            batch_tokens.extend(raw)
        except Exception as e:
            print("跳过错误行")
    return np.array(batch_tokens, dtype=np.uint16)


def main():
    fp = "/data/lqc/Practice/llm/from_zero_to_one/corpus/pretrain_t2t.jsonl"
    max_line = 2000
    max_workers = 16
    out_train_file = (
        "/data/lqc/Practice/llm/from_zero_to_one/corpus/pretrain_t2t_train.bin"
    )
    out_val_file = "/data/lqc/Practice/llm/from_zero_to_one/corpus/pretrain_t2t_val.bin"

    fcount = 0
    with open(fp, "rb") as f:
        for _ in f:
            fcount += 1

    train_num = int(fcount * 0.9)

    print(f"train 数量 {train_num}, val 数量 {fcount - train_num}")

    with open(out_train_file, "ab") as out_train_bin, open(
        out_val_file, "ab"
    ) as out_val_bin:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            batch = []
            current_out = out_train_bin
            with open(fp) as f:
                for i, line in enumerate(tqdm(f, total=fcount, desc="processing:")):
                    if i == train_num:
                        futures.append(executor.submit(process_line_batch, batch))
                        batch = []
                        for future in futures:
                            future.result().tofile(current_out)
                        current_out = out_val_bin

                    batch.append(line)

                    if len(batch) >= max_line:
                        futures.append(executor.submit(process_line_batch, batch))
                        batch = []
                    if len(futures) > 2 * max_workers:
                        res = futures.pop(0).result()
                        res.tofile(current_out)

                if batch:
                    futures.append(executor.submit(process_line_batch, batch))
                for future in futures:
                    future.result().tofile(current_out)


def check_bin():
    bin_file = "/data/lqc/Practice/llm/from_zero_to_one/corpus/pretrain_t2t_val.bin"
    a = np.fromfile(bin_file, dtype=np.uint16)
    print(a[:20])


check_bin()
# main()
