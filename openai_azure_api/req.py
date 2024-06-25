#!/usr/bin/env python3
#coding:utf-8


import bytedlogger
import bytedmetrics
import thriftpy2
import traceback
import euler
from euler import base_compat_middleware
import json

base_thrift = thriftpy2.load("openai_azure_api/base.thrift", module_name="base_thrift")
from base_thrift import BaseResp  # type: ignore # noqa: E402

openai_api_in_house_thrift = thriftpy2.load("openai_azure_api/openai_api_in_house.thrift", module_name="openai_api_in_house_thrift")
from openai_api_in_house_thrift import OpenAI_API_Server, ChatCompletions_Req, ChatCompletions_Rsp  # type: ignore # noqa

timeout=2000
client = euler.Client(OpenAI_API_Server, 'sd://motor.nlp.llm_api?idc=lq&cluster=Bernard-Prod', timeout=timeout)
client.use(base_compat_middleware.client_middleware)


def chatcompletions(engine='gpt-3.5-turbo', temperature=1.0, max_tokens=20, best_of=1, username='', question="", logit_bias=None, history=None):
    req = ChatCompletions_Req()
    req.engine = engine
    req.temperature = temperature
    req.max_tokens = max_tokens
    req.username = username
    req.best_of = best_of
    req.timeout = timeout
    if logit_bias is not None:
        req.logit_bias = logit_bias
    if history is not None:
        req.messages = history
    else:
        req.messages = [{"role": "system", "content": "你是一个靠谱的助手。"}]
    req.messages.append({"role": "user", "content": question})

    res = client.chat_completions(req)
    if res.BaseResp.StatusCode != 0:
        raise Exception(f"OpenAI Chat Call Error: {res.BaseResp.StatusMessage}")

    res = json.loads(res.result)
    return res["choices"][0]["message"]["content"]


if __name__ == '__main__':
    question = """"""
    print(chatcompletions('gpt-4-0613', question=question, max_tokens=800, temperature=0))
