import json
import pandas as pd


def load_json(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return [json.loads(line.strip()) for line in f.readlines()]
    

def get_human_labels(json_data):
    labels = []
    for k, v in json_data["label"].items():
        if v == 1:
            labels.append(k)
    return ",".join(labels)


def get_gpt_labels(json_data):
    labels = []
    reasons = []
    for k, v in json_data["output_label"].items():
        if v["label"] == 1:
            labels.append(k)
            reasons.append(v["reason"].replace("\n\n", "\n").strip())
        elif json_data["label"][k] == 1:
            reasons.append(v["reason"].replace("\n\n", "\n").strip())
    return ",".join(labels), "\n\n".join(reasons)


tasks = ["knowledge_qa", "emotion_analyze_nlg", "title", "search_qa"]
for task in tasks:
    data = load_json(f"judge_test_result_v3-shot_v2-多轮zero-shot/eval/{task}.json")
    df = pd.DataFrame(columns=["id", "引导词", "模型回答", "人工标注标签", "GPT4标注标签", "GPT4打标解释", "人工分数", "GPT4分数"])
    for jd in data:
        gpt_labels, gpt_reasons = get_gpt_labels(jd)
        row = [jd["id"], jd["instruction"], jd["answer"], get_human_labels(jd), gpt_labels, gpt_reasons, jd["score"], jd["output_score"]]
        # if row[-1] != row[-2]:  # score
        if row[3] != row[4]:   # label
            df.loc[len(df)] = row
    print(task, len(df), len(data), f"{len(df) / len(data):.2%}")
    # file_path = f"error_cases/{task}_score_test_v4.csv"
    file_path = f"error_cases/{task}_label_eval-shot_v2_加zero-shot.csv"
    df.to_csv(file_path, index=False)