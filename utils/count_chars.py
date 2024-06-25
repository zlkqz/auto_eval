import re

def count_chars(string):
    total_chars = 0
    for char in string:
        if re.match(r'[\u4e00-\u9fff]', char):  # 匹配中文字符
            total_chars += 1
        elif re.match(r'\w', char):  # 匹配字母、数字和下划线
            total_chars += 1
        elif re.match(r'[^\u4e00-\u9fff\w\s]', char):  # 匹配标点符号
            total_chars += 1
    return total_chars


def count_chars4title(string):
    total_chars = 0
    for char in string:
        if re.match(r'[\u4e00-\u9fff]', char):  # 匹配中文字符
            total_chars += 1
        elif re.match(r'\w', char):  # 匹配字母、数字和下划线
            total_chars += 0.5
        elif re.match(r'[^\u4e00-\u9fff\w\s]', char):  # 匹配标点符号
            total_chars += 0.5
        elif char == ' ':  # 特殊处理空格
            total_chars += 0.5
    return total_chars


if __name__ == "__main__":
    s = """  a  fa这是一C'est条-_测établir试。\n\n月0\t\t134,.用チベ力   。"""
    print(count_chars(s))
    print(count_chars4title(s))
