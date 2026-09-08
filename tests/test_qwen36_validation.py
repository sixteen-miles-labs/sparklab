"""The release grader must reject wrong types and unsafe generated programs."""
import importlib.util
from pathlib import Path

import pytest


path = Path(__file__).resolve().parents[1] / 'benchmarks/qwen36_marlin/validate_serving.py'
spec = importlib.util.spec_from_file_location('qwen36_validation', path)
validation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(validation)


@pytest.mark.parametrize('actual,expected', [
    ({'count':True}, {'count':1}),
    ({'a':17.0,'b':23}, {'a':17,'b':23}),
    ({'values':[False,1]}, {'values':[0,1]}),
    ({'wrong_key':False}, {'even':False}),
])
def test_json_types_and_keys_are_checked(actual, expected):
    assert not validation.typed_equal(actual, expected)


@pytest.mark.parametrize('source', [
    'def sign(value):\n    return __import__("os").getpid()',
    'def sign(value):\n    while True: pass',
    'def sign(value):\n    return sign(value)',
    'import os\ndef sign(value):\n    return 1',
    'def sign(value):\n    return value.__class__',
    'def sign(value):\n    return 1',
])
def test_code_grader_rejects_unsafe_or_wrong_programs(source):
    case = {'function':'sign', 'args':['value'], 'tests':[([-3],-1),([0],0),([4],1)]}
    assert not validation.evaluate_code(source, case)


def test_code_grader_accepts_correct_bounded_program():
    case = {'function':'sign', 'args':['value'], 'tests':[([-3],-1),([0],0),([4],1)]}
    assert validation.evaluate_code('def sign(value):\n    if value < 0: return -1\n    if value > 0: return 1\n    return 0', case)


def test_truncated_correct_answer_does_not_pass():
    case = {'kind':'exact', 'expected':'42'}
    result = {'response':{'choices':[{'message':{'content':'42'},'finish_reason':'length'}]}}
    assert not validation.score(case, result)


@pytest.mark.parametrize('field', ['reasoning_content','reasoning'])
def test_both_servers_reasoning_fields_are_scored(field):
    case = {'kind':'reasoning', 'expected':70}
    result = {'response':{'choices':[{'message':{'content':r'\boxed{70}',field:'Calculation'},'finish_reason':'stop'}]}}
    assert validation.score(case, result)
