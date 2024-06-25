# TALEC

[**Paper**]() (coming soon)

In this paper, we propose a model-based evaluation method: **TALEC**, which allows users to flexibly set their own evaluation criteria, and uses in-context learning (ICL) to teach judge model these in-house criteria. In addition, we try combining zero-shot and few-shot to make the judge model focus on more information. We also propose a prompt paradigm and an engineering approach to adjust and iterate the shots ,helping judge model to better understand the complex criteria. We then compare fine-tuning with ICL, finding that fine-tuning can be replaced by ICL. TALEC demonstrates a strong capability to accurately reflect human preferences and achieves a correlation of over 80% with human judgments, outperforming even the inter-human correlation in some tasks.
**Judge Process:**
![Judge Process](https://github.com/zlkqz/auto_eval/blob/master/img/judge_process.jpg)


### Quick Start
1. Fix `openai_azure_api/req.py`
2. `/bin/bash start_bash_demo.sh`


### Data
- We release 2 types of data here: **data_zh**(Chinese version) and **data_en**(English version)
- All the experiments are done on data_zh. To make it easier to understand, in the text, we translate data_zh to data_en, which means data_en is only for reference.
- This is our internal closed-source benchmark, which will be released soon. So we just release part of our data in this repo, just for the data protection needs.
- We split eval&&test datasets in data_zh/data_en
- There are 3 versions of shots in out paper:
    1. **shot_v1:** Which is named "Arbitrariness" in the paper. It uses arbitrary format to write shots.
    2. **shot_v2:** Which is named "Standard Prompt Paradigm"/"Repeat descriptions" in the paper. It uses standard format to write shots, which consistent format for both positive and negative examples and repeat descriptionf of label before judging.
    3. **shot_v3:** Which is named "Standard(Non-repetition)" in the paper. It is very similar to shot_v2, but without repetition of label's descriptionf.


### Judge Model and API
- We Openai Azure Api in our experiments, the code can be seen in `openai_azure_api/`
- But the contents of `openai_azure_api/` would give away the organization I belong to, so I omitted it. **You must implement your API calling method by yourself.**

### Prompt
- All the descriptions of our criteria are shown in `utils/prompt.py`
- Due to business confidentiality requirements, we only show part of the descriptions. **You can use your own criteria by fixing `utils/prompt.py`**
- Here is a brief description of our criteria:
![Criteria](https://github.com/zlkqz/auto_eval/blob/master/img/criteria.jpg)

### Judge Process
1. We mainly use `test_judge.py` or `test_judge_with_0shot.py` to judge. The former one is a typical few-shot method, and the latter one adds zero-shot and uses multi-turn method to judge. The former one is named "Multi-turn with Zero-shot" in the paper.
2. Then we use `judge_char.py` to judge the word count requirement if you want.
3. You can run `get_error_cases.py` to find cases of judge errors and can run `get_spearman.py` to get the correlation coefficient.




