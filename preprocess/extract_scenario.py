import pdb
import time
import re
import sys
import os
import numpy as np
import pandas as pd
import json
import openai
import random
import pickle
import argparse

from tqdm import tqdm
from pathlib import Path
from transformers import GPT2TokenizerFast
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage

from prompts import extract_scenario_prompt


llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k", verbose=True)
gpt_tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")


parser = argparse.ArgumentParser()
parser.add_argument("--input-json", type=str, default=None)
parser.add_argument("--output-dir", type=str)
parser.add_argument("--debug", action='store_true')
args = parser.parse_args()



def gpt_seq_len(text):
    return len(gpt_tokenizer(text)['input_ids'])


def get_formatted_prompt(call):
    company_info = call['company_info']
    industry_starts_with_vowel = company_info['industry'] and company_info['industry'][0].lower() in ['a', 'e', 'i', 'o', 'u']
    company_appositive = f", a{'n' if industry_starts_with_vowel else ''} {company_info['industry'].lower() + ' ' if company_info['industry'] else ''}company"
    transcript = "\n".join([f"{turn['from'].upper()}: {turn['value']}" for turn in call['conversations']])
    formatted_prompt = extract_scenario_prompt.format(
        company_name=company_info['name'],
        company_appositive=company_appositive,
        company_description=company_info['description'],
        transcript=transcript
    )
    return formatted_prompt


def extract_scenario(call):
    formatted_prompt = get_formatted_prompt(call)
    print("Prompt tokens: ", gpt_seq_len(formatted_prompt))
    final_messages = [SystemMessage(content=formatted_prompt)]
    llm_response = llm(final_messages)
    raw_response = llm_response.content
    print("Response tokens: ", gpt_seq_len(raw_response))
    return formatted_prompt, raw_response



def main():
    input_filename = Path(args.input_json).stem
    with open(args.input_json, "r") as f:
        dataset = json.load(f)

    with open("/Users/andrew/dev/FastChat-Cue/preprocess/data/prospect_calls_hightouch_v1_3_extracted_scenarios_call_id.json", "r") as f:
        scenario_call_id_dict = json.load(f)

    with open("/Users/andrew/dev/FastChat-Cue/preprocess/data/prospect_calls_hightouch_v1_3_extracted_scenarios_idx.json", "r") as f:
        scenario_idx_dict = json.load(f)

    with open("/Users/andrew/dev/FastChat-Cue/preprocess/data/prospect_calls_hightouch_v1_3_extracted_scenario_prompts_idx.json", "r") as f:
        prompt_idx_dict = json.load(f)
    
    failed_idxs = []

    def save_results():
        with open(Path(args.output_dir) /  f"{input_filename}_extracted_scenarios_idx.json", "w") as f:
            json.dump(scenario_idx_dict, f)

        with open(Path(args.output_dir) / f"{input_filename}_extracted_scenarios_call_id.json", "w") as f:
            json.dump(scenario_call_id_dict, f)

        with open(Path(args.output_dir) / f"{input_filename}_extracted_scenario_prompts_idx.json", "w") as f:
            json.dump(prompt_idx_dict, f)

    for i, call in enumerate(dataset):
        if call['id'] not in scenario_call_id_dict:
            print(f"{'=' * 32} CALL {i+1} OF {len(dataset)} {'=' * 32}")
            try:
                prompt, response = extract_scenario(call)
                scenario_idx_dict[i] = response
                scenario_call_id_dict[call['id']] = response
                prompt_idx_dict[i] = prompt
            except Exception as e:
                print(e)
                failed_idxs.append(i)

            if i % 10 == 0:
                print("Saving intermediate results")
                save_results()

    print("Saving final results")
    save_results()
            

    print(f"Failed to extract scenario for {len(failed_idxs)} calls.\nFailed indexes: {failed_idxs}")



if __name__ == "__main__":
    main()