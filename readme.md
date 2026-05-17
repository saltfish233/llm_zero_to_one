## Introduction
为了深入对LLM的理解，参考了minimind项目以及b站up主“chaofa用代码打点酱油”，应用新技术实现LLM并持续性优化。
目前阶段不打算拆分不同类到不同文件夹

# How to use
目前只考虑多卡DDP训练，根据自身情况修改里面的一些属性：
*preprocess*
python preprocess/convert_jsonl_to_bin_minimind_onlytrain.py
*pretrain*
sh train_20260515.sh
*sft*
sh sft_20260515.sh

## Insight
1. generate中的采样策略非常影响回答质量，原本写的一个简单的采样策略发现回答效果很差，参考minimind的generate采样策略后发现生成质量提升非常大。因此不一定是模型差劲，也可能是采样策略没写好。

## Todo
1. 搞明白generate策略
2. 实现kvcache
3. 实现moe
4. 实现yarn
5. 拓展实现omni-llm
6. etc