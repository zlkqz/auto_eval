from scipy.stats import spearmanr, kendalltau, pearsonr
import json

def load_json(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return [json.loads(line.strip()) for line in f.readlines()]


dir_path = "test/Qwen_SFT_evalOS/test/"
# dir_path = "/mnt/bn/motor-nlp-team/mlx/users/zhangkaiqi.zlkqz/repo/5355/llm_auto_eval/history/judge_labels_v2-5.9_删除了知识安全问答的分段标签/"
# dir_path = "history/judge_test_result_v2，已评估，统计字数前/"
# tasks = ["knowledge_qa", "emotion_analyze_nlg", "dialogue_summary", "title", "search_qa"] #, "writing", "knowledge_qa", "emotion_analyze_nlg", "dialogue_summary", "title", "search_qa"]
tasks = ["emotion_analyze_nlg", "knowledge_qa", "search_qa", "title"]

for task in tasks:
    data1, data2 = [], []
    d = load_json(dir_path + task + ".json")
    for json_data in d:
        data1.append(json_data["score"])
        data2.append(json_data["output_score"])

    spearman, p_1 = spearmanr(data1, data2)
    pearson, p_2 = pearsonr(data1, data2)
    kendall, p_3 = kendalltau(data1, data2)

    # 打印结果
    print(f"Spearman correlation coefficient of {task}: {round(spearman, 4)}  p_value: {p_1}")
    print(f"Pearson correlation coefficient of {task}: {round(pearson, 4)}  p_value: {p_2}")
    print(f"Kendall correlation coefficient of {task}: {round(kendall, 4)}  p_value: {p_3}")
    print("\n")

