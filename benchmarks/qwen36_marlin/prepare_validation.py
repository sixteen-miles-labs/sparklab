#!/usr/bin/env python3
"""Freeze an auditable Qwen3.6 corpus; run in the pinned image for token counts."""
import argparse
import hashlib
import json
from pathlib import Path


def build(tokenizer, aime_path):
    cases = []
    def add(name, kind, prompt, expected=None, thinking=False, budget=2048, **extra):
        body = {'messages':[{'role':'user','content':prompt}], 'temperature':0,
            'max_completion_tokens':max(budget, 8192) if thinking else budget,
            'chat_template_kwargs':{'enable_thinking':thinking}}
        if kind == 'reasoning':
            body.update(stream=True, stream_options={'include_usage':True})
        cases.append(dict(id=name, kind=kind, expected=expected, body=body, **extra))
    for i, (prompt, answer) in enumerate([
        ('What is 137 + 286?',423), ('What is 864 divided by 12?',72),
        ('What is 19 times 23?',437), ('What is 1000 minus 347?',653),
        ('What is the greatest common divisor of 84 and 126?',42),
        ('What is the remainder when 2026 is divided by 17?',3),
    ]):
        add(f'arithmetic-{i}', 'exact', prompt+' Reply with only the integer.', answer, budget=128)
    for i, (prompt, answer) in enumerate([
        ('A train travels 120 km at 60 km/h and then 180 km at 90 km/h. What is its average speed in km/h over the whole trip?',75),
        ('A box has 5 red and 7 blue balls. How many balls must be drawn without replacement to guarantee at least 3 of the same color?',5),
        ('How many three-digit positive integers have digit sum 3?',6),
        ('What is the smallest positive integer divisible by 8, 9, and 15?',360),
    ]):
        add(f'reasoning-{i}','reasoning',prompt+' Think through the problem and put your final answer in \\boxed{...}.', answer, thinking=True)
    for i, (prompt, answer) in enumerate([
        ('Extract name and count from: Maple has 7 items. Use keys name (string) and count (integer).', {'name':'Maple','count':7}),
        ('Sort these integers ascending: 9, -2, 5, 0. Use the key values with an array.', {'values':[-2,0,5,9]}),
        ('Is 37 even? Use the key even with a boolean value.', {'even':False}),
        ('For the empty list, return its count and first item. Use count and first; the first item is null when absent.', {'count':0,'first':None}),
    ]):
        add(f'json-{i}','json',prompt+' Return only valid JSON, without markdown.',answer,budget=256)
    for name, args, description, tests in [
        ('clamp',['value','lower','upper'],'Return lower if value is below lower, upper if above upper, otherwise value.',[([5,0,10],5),([-2,0,10],0),([12,0,10],10),([3.5,3.5,3.5],3.5)]),
        ('sign',['value'],'Return -1 for negative values, 0 for zero, and 1 for positive values.',[([-8],-1),([0],0),([9],1),([-0.5],-1)]),
        ('absolute_difference',['a','b'],'Return the nonnegative absolute difference between a and b.',[([2,9],7),([9,2],7),([-4,3],7),([2,2],0)]),
        ('is_leap_year',['year'],'Return a boolean using the Gregorian rule: divisible by 4, except centuries unless divisible by 400.',[([2000],True),([1900],False),([2024],True),([2023],False)]),
    ]:
        add('code-'+name,'code',f'Write a Python function {name}({", ".join(args)}). {description} Output exactly one Python code block with only the function. Use comparisons, boolean or arithmetic operators, if statements, and returns only. No imports, calls, loops, decorators, type annotations or helper functions.', thinking=True,function=name,args=args,tests=tests)
    for i,(a,b) in enumerate([(17,23),(31,12)]):
        add(f'tool-{i}','tool',f'Use the multiply tool exactly once with a={a} and b={b}. Do not calculate it yourself or answer with prose.',{'a':a,'b':b},thinking=True)
        cases[-1]['body'].update(tools=[{'type':'function','function':{'name':'multiply','description':'Multiply two integers.','parameters':{'type':'object','properties':{'a':{'type':'integer'},'b':{'type':'integer'}},'required':['a','b'],'additionalProperties':False}}}],tool_choice='auto')
    rows = [json.loads(s) for s in aime_path.read_text().splitlines() if s.strip()]
    for i,row in enumerate(rows[:5]):
        add(f'aime25-{i}','aime',row['problem']+'\nPut your final answer in \\boxed{...}.',row['answer'],thinking=True,budget=16384)
    contexts=[]
    for target in (1024,8192,16384,32768):
        secret = f'MAPLE-{target}-731940'
        prefix=f'Remember this access code exactly: {secret}.\n'
        suffix='\nWhat is the access code? Reply with only the code.'
        lo,hi=0,target
        while lo<=hi:
            mid=(lo+hi)//2
            prompt=prefix+' x'*mid+suffix
            encoded=tokenizer.apply_chat_template([{'role':'user','content':prompt}],tokenize=True,add_generation_prompt=True,enable_thinking=False)
            count=len(encoded.input_ids) if hasattr(encoded,'input_ids') else len(encoded)
            if count==target:
                break
            if count<target: lo=mid+1
            else: hi=mid-1
        else:
            raise ValueError(f'cannot construct {target} tokens')
        contexts.append({'id':f'context-{target}','kind':'exact','tokens':target,'expected':secret,'body':{'messages':[{'role':'user','content':prompt}], 'temperature':0,'max_completion_tokens':32,'chat_template_kwargs':{'enable_thinking':False}}})
    return {'description':'Fixed regression/capability cases plus first five AIME-25 problems; greedy, normal EOS. Not a representative quality benchmark.', 'aime_sha256':hashlib.sha256(aime_path.read_bytes()).hexdigest(),'cases':cases,'contexts':contexts}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',required=True)
    p.add_argument('--aime',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(a.model,local_files_only=True)
    a.output.write_text(json.dumps(build(tokenizer,a.aime),indent=2)+'\n')
