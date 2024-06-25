def get_criteria(task):
    criteria = {
        "general": {
            "0-无翻译外文": {"name": "存在翻译外文", "describe": "回答中存在大篇幅的非中文语言句子。若仅仅含有一些专业名词的非中文词语（如车系名、发动机型号等），但回答整体绝大多数都是中文，则并不存在该问题。"},
            "0-拒绝回答": {"name": "拒绝回答", "describe": "模型在明明可以回答指令中的问题的情况下，仍然显示地拒绝回答问题"},
            "0-不符合要求": {"name": "不符合要求", "describe": "回答未严格、明确执行指令中的生成要求。若指令中的要求包括字数要求，请忽略字数要求。"},
            "1-答案不全": {"name": "答案不全", "describe": "未能解决指令中的所有需求，或者回答没有将所有要点回答出来"},
            "1-机械感": {"name": "机械感", "describe": "回答中存在一些类似于机器人接收指令的内容，或者自问自答，给读者造成一种在和机器人生硬聊天的感觉."},
            "1-重复表达": {"name": "重复表达", "describe": "回答中有重复表达内容，重叠的信息多次出现。"},
            "0-错误回答/不相关匹配结果": {"name": "包含错误信息或生成不匹配回答", "describe": "回答中的答案是错误的，或者回答和指令中提的问题并不相关"},
            "0-影响阅读": {"name": "影响阅读", "describe": "回答中出现乱码，或者存在符号乱用，或者输出格式不正确。"},
            "1-主体不精准": {"name": "内容主体不精准", "describe": "回答中存在和指令要求不相关的冗余内容，或内容中描述的主体不清晰"},
            "0-竞品/网站/APP或外部引流": {"name": "存在外部链接或引流", "describe": "回答中存在网页链接、APP链接或者引导读者到线下店面的行为"},
        },
        # "general": {
        #     "0-Untranslated Text": {"name": "Untranslated Text", "describe": "Large portions of non-Chinese language in responses."},
        #     "0-Refusal to Answer": {"name": "Refusal to Answer", "describe": "The model explicitly shows a refusal to answer when it can."},
        #     "0-Not Meeting the Requirements": {"name": "Not Meeting the Requirements", "describe": "Answer failed to strictly enforce the requirements."},
        #     "1-Incomplete Answers": {"name": "Incomplete Answers", "describe": "Failure to address all needs in the directive"},
        #     "1-Stiffness": {"name": "Stiffness", "describe": "The response lacked anthropomorphic expression."},
        #     "1-Repetitive Expression": {"name": "Repetitive Expression", "describe": "Repeated expressions in the answer."},
        #     "0-Incorrect Answer/Unrelated Matching Results": {"name": "Incorrect Answer/Unrelated Matching Results", "describe": "Containing incorrect information or mismatched responses."},
        #     "0-Confusing Answers": {"name": "Confusing Answers", "describe": "Answers contain messy code or content that interferes with reading."},
        #     "1-Subject Imprecision": {"name": "Subject Imprecision", "describe": "Redundant content or unclear subject in the response"},
        #     "0-External Links or Diversions": {"name": "External Links or Diversions", "describe": "External links or obvious diversionary behavior in the answer."},
        # },
        "emotion_analyze_nlg": {
        },
        "knowledge_qa": {
        },
        "search_qa": {
        },
        "title": {
        },
    }
    criteria = {**criteria["general"], **criteria[task]}
    return criteria


def get_judge_sys_prompt(task ,label_name):
    return f"""你是一个评估大语言模型（Large Language Model, LLM）的评判者，对一个大语言模型对于某个指令给出的回答进行质量评估。
下面会给出一些指令（instruction），以及该指令对应的一个回答（response）。请你判断该回答是否有\"{get_criteria(task)[label_name]['name']}\"的问题
\"{get_criteria(task)[label_name]['name']}\"的解释为：{get_criteria(task)[label_name]['describe']}
请不要一开始就给出最终答案，必须先充分给出你的分析，最后再回答是否存在该问题，如果存在该问题，则请在最后输出[[A]]，反之则输出[[B]]"""


def get_judge_query_prompt(instruction, response, task, label_name, reference=None):
    if (reference is not None) and \
    ((task == "emotion_analyze_nlg" and label_name in ["0-错误回答/不相关匹配结果"]) or (task == "knowledge_qa" and label_name in ["0-错误回答/不相关匹配结果", "1-答案不全"])):
        if task == "emotion_analyze_nlg":
            reference = f"""以下是指令中给出的文本对于“懂车帝”的情感类型（emotion），作为参考答案，能对你的评判提供一些帮助：
<emotion>
{reference}
</emotion>"""
        elif task == "knowledge_qa":
            reference = f"""以下是指令中给出的问题的参考信息（reference），可能对你的评判提供一些帮助：
<reference>
{reference}
</reference>"""
        else:
            raise ValueError(f"Task {task} has no references.")
        return f"""以下是一条指令:
<instruction>
{instruction}
</instruction>


{reference}


以下是针对上面这条指令的回答
<response>
{response}
</response>


请你判断该回答是否存在上述的\"{get_criteria(task)[label_name]['name']}\"问题"""
        
    else:
        return f"""以下是一条指令:
<instruction>
{instruction}
</instruction>


以下是针对上面这条指令的回答
<response>
{response}
</response>


请你判断该回答是否存在上述的\"{get_criteria(task)[label_name]['name']}\"问题"""


if __name__ == "__main__":
    print(get_judge_sys_prompt("emotion_analyze_nlg", "0-无翻译外文"))
    print(get_judge_query_prompt("ins", "res", "emotion_analyze_nlg", "0-无翻译外文"))