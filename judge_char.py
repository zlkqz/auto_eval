from utils.count_chars import *
# from utils.utils import *
import re
import json


def load_json(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return [json.loads(line.strip()) for line in f.readlines()]
    

def save_json(file_path, json_data):
    with open(file_path, "w", encoding="utf-8") as f:
        for jd in json_data:
            f.write(json.dumps(jd, ensure_ascii=False) + "\n")


def check_title_format4title(text):
    pattern = r'主标题：(.*?)\s*副标题：(.*?)'
    match = re.search(pattern, text, re.DOTALL)
    
    if match:
        main_title = match.group(1)
        sub_title = match.group(2)
        sub_match = re.search(r'\n+', sub_title)
        if sub_match:
            sub_title = sub_title[:sub_match.start()]
        return main_title, sub_title
    else:
        match = re.search(r'\n+', text)
        if match:
            # 获取分隔符的位置
            separator_start = match.start()
            separator_end = match.end()
            main_title, sub_title = text[:separator_start], text[separator_end:]
            sub_match = re.search(r'\n+', sub_title)
            if sub_match:
                sub_title = sub_title[:sub_match.start()]
            return main_title, sub_title
        else:
            return text, None    # 只有一个标题


def get_output_score(json_data):
    score = 2
    for label_name, num in json_data["output_label"].items():
        if num["label"] == 1 and label_name.startswith("0-"):
            score = 0
            break
        if num["label"] == 1 and label_name.startswith("1-"):
            score = 1
    return score


if __name__ == "__main__":
    tasks = {
        # "dialogue_summary": (100, 150), 
        "title": (17, 23, 50, 60), 
        "search_qa": (200, 250)
    }
    dir_path = "test/Qwen_SFT_evalOS"
    for dataset in ["test", "eval"]:
        for task, nums in tasks.items():
            data = load_json(f"{dir_path}/{dataset}/{task}.json")
            if task == "title":
                for json_data in data:
                    main_title, sub_title = check_title_format4title(json_data["answer"])
                    main_length = count_chars4title(main_title)
                    sub_length = count_chars4title(sub_title) if sub_title is not None else None
                    if  nums[0] < main_length <= nums[1]:
                        json_data["output_label"]["1-不符合要求"]["label"] = 1
                        json_data["output_label"]["1-不符合要求"]["reason"] += f"\n主标题字数为{main_length}，存在“1-不符合要求”问题"
                        json_data["output_score"] = min(json_data["output_score"], 1)
                        if json_data["label"]["1-不符合要求"] == 0:
                            json_data["label"]["1-不符合要求"] = 1
                            json_data["score"] = min(json_data["score"], 1)
                    elif main_length > nums[1]:
                        json_data["output_label"]["0-不符合要求"]["label"] = 1
                        json_data["output_label"]["0-不符合要求"]["reason"] += f"\n主标题字数为{main_length}，存在“0-不符合要求”问题"
                        json_data["output_score"] = min(json_data["output_score"], 0)
                        if json_data["label"]["0-不符合要求"] == 0:
                            json_data["label"]["0-不符合要求"] = 1
                            json_data["score"] = min(json_data["score"], 0)
                    if sub_length is not None:
                        if nums[2] < sub_length <= nums[3]:
                            json_data["output_label"]["1-不符合要求"]["label"] = 1
                            json_data["output_label"]["1-不符合要求"]["reason"] += f"\n副标题字数为{sub_length}，存在“1-不符合要求”问题"
                            json_data["output_score"] = min(json_data["output_score"], 1)
                            if json_data["label"]["1-不符合要求"] == 0:
                                json_data["label"]["1-不符合要求"] = 1
                                json_data["score"] = min(json_data["score"], 1)
                        elif sub_length > nums[3]:
                            json_data["output_label"]["0-不符合要求"]["label"] = 1
                            json_data["output_label"]["0-不符合要求"]["reason"] += f"\n副标题字数为{sub_length}，存在“0-不符合要求”问题"
                            json_data["output_score"] = min(json_data["output_score"], 0)
                            if json_data["label"]["0-不符合要求"] == 0:
                                json_data["label"]["0-不符合要求"] = 1
                                json_data["score"] = min(json_data["score"], 0)
            else:
                for json_data in data:
                    length = count_chars(json_data["answer"])
                    if  nums[0] < length <= nums[1]:
                        json_data["output_label"]["1-不符合要求"]["label"] = 1
                        json_data["output_label"]["1-不符合要求"]["reason"] += f"\n模型回答字数为{length}，存在“1-不符合要求”问题"
                        json_data["output_score"] = min(json_data["output_score"], 1)
                        if json_data["label"]["1-不符合要求"] == 0:
                            json_data["label"]["1-不符合要求"] = 1
                            json_data["score"] = min(json_data["score"], 1)
                    elif length > nums[1]:
                        json_data["output_label"]["0-不符合要求"]["label"] = 1
                        json_data["output_label"]["0-不符合要求"]["reason"] += f"\n模型回答字数为{length}，存在“0-不符合要求”问题"
                        json_data["output_score"] = min(json_data["output_score"], 0)
                        if json_data["label"]["0-不符合要求"] == 0:
                            json_data["label"]["0-不符合要求"] = 1
                            json_data["score"] = min(json_data["score"], 0)
            save_json(f"{dir_path}/{dataset}/{task}.json", data)