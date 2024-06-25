from utils.utils import *
from utils.prompt import *
import argparse
from openai_azure_api.req import chatcompletions
from time import sleep
import pandas as pd
import tiktoken
import os
from concurrent.futures import ThreadPoolExecutor, Future, as_completed
from utils.qps_decorator import call_qps_limit
from tqdm import tqdm
from datetime import datetime


def parse_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data_zh/judge_labels/test/")
    parser.add_argument("--shot_path", type=str, default="data_zh/shot_v2/")
    parser.add_argument("--qps", type=float, default="0.9")
    parser.add_argument("--max_try", type=int, default="20")

    return parser.parse_args()
args = parse_argument()
        

def call_gpt(instruction, history, max_tokens):
    global acc, tp, fp, fn, answer_len, engine
    for _ in range(args.max_try):
        try:
            judge = chatcompletions(engine=engine,
                                    temperature=0,
                                    question=instruction,
                                    history=history[:],
                                    max_tokens=max_tokens)
            break
        except Exception as e:
            if "RateLimitError" in str(e):
                print("TPM、QPS限制")
                sleep(10)
                continue
            else:
                print(f"访问错误\n{str(e)}")
                if engine == "gpt-4-0613":
                    engine = "gpt-4-32k-0613"
                if engine == "gpt-4-32k-0613":
                    engine = "gpt-4-0125-preview"
                else:
                    print("长度超过128k或者其他错误")
                continue
    return judge


@call_qps_limit(args.qps)
def single_judge(json_data, history, task, label_name):
    global acc, tp, fp, fn, answer_len, engine
    instruction = get_judge_query_prompt(json_data["instruction"], json_data["answer"], task, label_name, reference=json_data["reference"] if "reference" in json_data else None)
    # zero-shot
    zero_shot_judge = call_gpt(instruction, [{"role": "system", "content": get_judge_sys_prompt(task, label_name)}], max(answer_len+300, 800))
    sleep(5)
    # few-shot
    few_shot_judge = call_gpt(instruction, history, max(answer_len+200, 700))
    zero_shot_output = get_output(zero_shot_judge)
    few_shot_output  = get_output(few_shot_judge)
    if zero_shot_output != few_shot_output:
        sleep(5)
        # 综合两轮，第三轮最终判断
        judge = call_gpt(f"你会发现，上面的两轮对话都是针对同一个指令和回答进行评判，但是他们可能提出了不同的观点和角度。请结合以上所有对话中的评判例子以及最开始提供的\"{get_criteria(task)[label_name]['name']}\"的解释，总结出\"{get_criteria(task)[label_name]['name']}\"的评估标准和尺度，重新对以下指令和回答判断一次：\n\n" + instruction,
                         history[:] + [{"role": "user", "content": instruction}, {"role": "assistant", "content": zero_shot_judge}, {"role": "user", "content": instruction}, {"role": "assistant", "content": few_shot_judge}],
                         max(answer_len+300, 900))
    else:
        judge = few_shot_judge
    
    if "output_label" not in json_data:
        json_data["output_label"] = {k: {"label": 0, "reason": ""} for k in json_data["label"].keys()}
    if "output_score" not in json_data:
        json_data["output_score"] = 2
    
    if not judge.startswith("访问错误"):
        l = json_data["label"][label_name]
        output = get_output(judge)
        acc += int(l == output)
        if l == 1 and output == 1:
            tp += 1
        if l == 0 and output == 1:
            fp += 1
        if l == 1 and output == 0:
            fn += 1
        if l != output:
            print(f"""label: {l}  output: {output}
**********************Instruction:**********************
{json_data['instruction']}
**********************Answer:**********************
{json_data['answer']}
**********************Zero-shot Judge:**********************
{zero_shot_judge}
**********************Few-shot Judge:**********************
{few_shot_judge}
**********************Final Judge:**********************
{judge}\n\n\n""")
            
        json_data["output_label"][label_name]["label"] = output
        json_data["output_label"][label_name]["reason"] = judge
        json_data["output_score"] = min(json_data["output_score"], int(label_name[0]) if output == 1 else 2)
    else:
        json_data["output_label"][label_name]["reason"] = judge

    return json_data
    

def gpt4_judge(data, history, task, label_name, output_path):
    global acc, tp, fp, fn, answer_len, engine
    pool = ThreadPoolExecutor(max_workers=50)
    futures = []
    for task_id, json_data in tqdm(enumerate(data)):
        future = pool.submit(single_judge, json_data, history, task, label_name)
        future.task_id = task_id
        futures.append(future)
    new_data, cur, total = [], 0, len(data)
    for future in as_completed(futures):
        json_data = future.result()
        cur += 1
        print(f"Progress: {cur}/{total}")
        new_data.append((future.task_id, json_data))
    pool.shutdown()

    acc /= len(data)
    print(f"{task} {label_name}  Acc: {acc}")
    print(f"{task} {label_name}  TP: {tp}  FP: {fp}  FN: {fn}")
    if tp + fp != 0 and tp + fn != 0 and tp != 0:
        p, r = tp / (tp + fp), tp / (tp + fn)
        f1 = (2 * p * r) / (p + r)
    else:
        p, r, f1 = 0, 0, 0
    print(f"{task} {label_name}  P: {p}  R: {r}  F1: {f1}")

    new_data = [json_data for task_id, json_data in sorted(new_data, key=lambda x: x[0])]
    save_json(output_path, new_data)
    with open(f"{args.data_path}/results.json", "a", encoding="utf-8") as f:
        f.write(json.dumps({
            "time": str(datetime.now()), "task": task, "label" : label_name,
            "acc": acc, "f1": f1, "tp" : tp, "fp": fp, "fn": fn, "p": p, "r": r
        }, ensure_ascii=False) + "\n")


def get_history(task, label_name):
    global acc, tp, fp, fn, answer_len, engine
    shots = load_json(f"{args.shot_path}/{task}/{fix_name(label_name)}.json")
    system_prompt = get_judge_sys_prompt(task, label_name)
    history = [{"role": "system", "content": system_prompt}]
    for shot in shots:
        history.append({
            "role": "user", 
            "content": get_judge_query_prompt(shot["instruction"], shot["answer"], task, label_name, reference=shot["reference"] if "reference" in shot else None)
        })
        history.append({
            "role": "assistant",
            "content": shot["reason"].strip()
        })
        answer_len += len(encoding.encode(shot["reason"].strip()))
    answer_len //= (len(history) - 1) // 2

    if estimate_length(history, encoding) < 7200:
        engine = "gpt-4-0613"
    else:
        engine = "gpt-4-32k-0613"
    return history


if __name__ == "__main__":
    tasks = ["knowledge_qa", "emotion_analyze_nlg", "title", "search_qa"]
    label_names = ["0-无翻译外文","0-竞品/网站/APP或外部引流","0-专有词汇错误",'0-影响阅读','0-拒绝回答','0-错误回答/不相关匹配结果','0-低价值内容','0-不符合要求','1-不符合要求','1-重复表达','1-语序不当','1-分段','1-主体不精准','1-答案不全','1-机械感','1-软文']
    encoding = tiktoken.encoding_for_model("gpt-4-0613")
    engine = "gpt-4-0613"

    for task in tasks:
        for label_name in label_names:
            print(f"Judge task: {task}  Judge label: {label_name}")
            if not os.path.exists(f"{args.shot_path}/{task}/{fix_name(label_name)}.json"):
                print(f"Task {task} don't have shots of label {label_name}. Skip the judge.")
                continue
            file_path = f"{args.data_path}/{task}.json"
            data = load_json(file_path)
            judge_label_static(data, label_name)

            acc, tp, fp, fn, answer_len = 0, 0, 0, 0, 0
            his = get_history(task, label_name)
            print(f"Use engine: {engine}")
            gpt4_judge(data, his, task, label_name, file_path)