import json
# import os
# import torch
# from transformers import AutoModelForCausalLM, AutoModel, AutoTokenizer
# from transformers.generation.utils import GenerationConfig
# import numpy as np
# import re
# from random import shuffle
    

def load_json(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return [json.loads(line.strip()) for line in f.readlines()]
    

def save_json(file_path, json_data):
    with open(file_path, "w", encoding="utf-8") as f:
        for jd in json_data:
            f.write(json.dumps(jd, ensure_ascii=False) + "\n")


def fix_name(file_name):
    """
    给标签取名的问题，标签名含'/'，无法用做存放shots文件的路径，打个补丁
    For example: 0-竞品/网站/APP或外部引流.json --> 0-竞品网站APP或外部引流.json
    """
    return file_name.replace("/", "")


def estimate_length(history, encoding):
    length = 0
    system_message = history[0]
    system_length = len(encoding.encode(system_message["content"]))
    for i in range(1, len(history)):
        length += len(encoding.encode(history[i]["content"]))
    # 预估一下给模型的query和模型的answer的长度
    print(f"History length: {length + system_length} With {len(history) // 2} messages")
    length *= (len(history) + 1) / (len(history) - 1)
    length = int(length) + system_length
    return length


def judge_label_static(data, label_name):
    pos_num, neg_num = 0, 0
    for json_data in data:
        if json_data["label"][label_name] == 1:
            pos_num += 1
        else:
            neg_num += 1
    print(f"Positive sample of label {label_name}: {pos_num}   Negative sample of label {label_name}: {neg_num}")


def get_output(judge_text):
    if "[[A]]" in judge_text[-10:]:
        return 1
    elif "[[B]]" in judge_text[-10:]:
        return 0
    else:
        count_a = judge_text.count('[[A]]')
        count_b = judge_text.count('[[B]]')
        if count_a > count_b:
            return 1
        else:
            return 0