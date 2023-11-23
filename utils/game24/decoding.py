import re
import sympy
from typing import List


def extract_expression(response: str):
    return response.strip().split('\n')[-1].lower().split('the answer is ')[-1].split('=')[0].strip()


def extract_expressions(responses: List[str]):
    return [extract_expression(response) for response in responses]


# refer to https://github.com/princeton-nlp/tree-of-thought-llm/blob/master/src/tot/tasks/game24.py
def get_answer_label(expression: str, question: str):
    numbers = re.findall(r'\d+', expression)
    problem_numbers = re.findall(r'\d+', question)
    if sorted(numbers) != sorted(problem_numbers):
        return False
    try:
        # print(sympy.simplify(expression))
        return sympy.simplify(expression) == 24
    except Exception as e:
        # print(e)
        return False



