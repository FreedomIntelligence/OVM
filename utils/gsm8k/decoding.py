from contextlib import contextmanager
import signal
import torch
import json
import os
import re


# ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
ANS_RE = re.compile(r"The answer is (\-?[0-9\.\,]+)")
INVALID_ANS = "[invalid]"



def extract_answer(completion):
    match = ANS_RE.search(completion)
    if match:
        match_str = match.group(1).strip()
        st_str = standardize_value_str(match_str)
        try: eval(st_str); return st_str
        except: ...
    return INVALID_ANS

def extract_answers(completions):
    return [extract_answer(completion) for completion in completions]

def standardize_value_str(x):
    """Standardize numerical values"""
    y = x.replace(",", "")
    if '.' in y:
        y = y.rstrip('0')
        if y[-1] == '.':
            y = y[:-1]
    if not len(y):
        return INVALID_ANS
    if y[0] == '.':
        y = '0' + y
    if y[-1] == '%':
        y = str(eval(y[:-1]) / 100)
    return y.rstrip('.')

def get_answer_label(response_answer, gt):
    if response_answer == INVALID_ANS:
        return INVALID_ANS
    return response_answer == gt



# taken from
# https://stackoverflow.com/questions/492519/timeout-on-a-function-call
@contextmanager
def timeout(duration, formula):
    def timeout_handler(signum, frame):
        raise Exception(f"'{formula}': timed out after {duration} seconds")

    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(duration)
    yield
    signal.alarm(0)


def eval_with_timeout(formula, max_time=3):
    try:
        with timeout(max_time, formula):
            return round(eval(formula), ndigits=4)
    except Exception as e:
        signal.alarm(0)
        print(f"Warning: Failed to eval {formula}, exception: {e}")
        return None


# refer to https://github.com/openai/grade-school-math/blob/master/grade_school_math/calculator.py
def use_calculator(sample):
    if "<<" not in sample:
        return None

    parts = sample.split("<<")
    remaining = parts[-1]
    if ">>" in remaining:
        return None
    if "=" not in remaining:
        return None
    lhs = remaining.split("=")[0]
    lhs = lhs.replace(",", "")
    if any([x not in "0123456789*+-/.()" for x in lhs]):
        return None
    ans = eval_with_timeout(lhs)
    if remaining[-1] == '-' and ans is not None and ans < 0:
        ans = -ans
    return ans






