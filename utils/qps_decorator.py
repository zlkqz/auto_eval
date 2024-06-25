import functools
import time
from threading import Lock


def call_qps_limit(qps=10):
    def decorator(func):
        # 函数调用外的变量，利用闭包函数的性质，类似全局变量
        lock = Lock()
        last_time = 0
        interval = 1.0 / qps

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            nonlocal last_time
            key = func.__name__
            # 必须加锁，否则多个任务获取到的last_time很可能就是初始last_time，导致瞬间执行完
            lock.acquire()
            try:
                now = time.time()
                # 初值为None，令其强制设置为now - 最小调用间隔，目的是可以直接强制执行
                if not last_time: last_time = now - interval
                delta = now - last_time
                if last_time and delta < interval:
                    # 等待，并修改调用时间
                    time.sleep(interval - delta)
                    # print("func {} should wait {}s".format(key, interval - delta))
                # 在调用前保存时间
                last_time = time.time()
            finally:
                lock.release()
            result = func(*args, **kwargs)
            return result

        return wrapper

    return decorator
