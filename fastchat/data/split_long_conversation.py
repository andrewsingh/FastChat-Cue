"""
Split long conversations based on certain max length.

Usage: python3 -m fastchat.data.split_long_conversation \
    --in sharegpt_clean.json \
    --out sharegpt_split.json \
    --model-name-or-path $<model-name>
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from typing import Dict, Sequence, Optional

import transformers
from tqdm import tqdm
import pdb


parser = argparse.ArgumentParser()
parser.add_argument("--in-file", type=str, required=True)
parser.add_argument("--out-file", type=str, default="sharegpt_split.json")
parser.add_argument("--model-name-or-path", type=str, required=True)
parser.add_argument("--max-length", type=int, default=2048)
parser.add_argument("--use-system-message", action='store_true')
args = parser.parse_args()



def make_sample(sample, start_idx, end_idx):
    assert (end_idx - start_idx) % 2 == 0
    new_sample = {
        "id": sample["id"] + "_" + str(start_idx),
    }
    for key in sample:
        if key not in new_sample:
            new_sample[key] = sample[key]

    new_sample["conversations"] = sample["conversations"][start_idx:end_idx]
    return new_sample



tokenizer = max_length = None


def split_one_sample(sample):
    max_length_v2 = args.max_length
    if args.use_system_message:
        system_message_len = len(tokenizer(sample['system_message']).input_ids)
        max_length_v2 = max_length_v2 - system_message_len

    tokenized_lens = []
    conversations = sample["conversations"]

    if conversations[0]['from'] == 'prospect':
        conversations = conversations[1:]

    if conversations[-1]['from'] == 'sales rep':
        conversations = conversations[:-1]

    if len(conversations) < 2:
        return []

    assert(conversations[0]['from'] == 'sales rep')
    assert(conversations[-1]['from'] == 'prospect')
    assert(len(conversations) % 2 == 0)
    for i in range(1, len(conversations)):
        assert(conversations[i]['from'] != conversations[i-1]['from'])

    for c in conversations:
        length = len(tokenizer(c["value"]).input_ids) + 6
        tokenized_lens.append(length)

    start_idx = 0
    cur_len = 0

    sample['conversations'] = conversations

    new_samples = []
    for i in range(0, len(conversations), 2):
        tmp_len = tokenized_lens[i] + tokenized_lens[i + 1]
        if cur_len + tmp_len > max_length_v2:
            new_samples.append(make_sample(sample, start_idx, i))
            start_idx = i
            cur_len = 0
        elif i == len(conversations) - 2:
            new_samples.append(make_sample(sample, start_idx, i + 2))

        cur_len += tmp_len

    return new_samples


def worker(input_data):
    result = []
    for sample in tqdm(input_data):
        result.extend(split_one_sample(sample))
    return result


def split_all(content, tokenizer_, max_length_):
    """
    Keep the maximum round of conversations within the max token length constraint
    """
    global tokenizer, max_length
    tokenizer = tokenizer_
    max_length = max_length_

    # content = content[begin:end]
    new_content = []

    new_content = worker(content)

    # Split content into chunks
    # chunks = [content[i : i + 1000] for i in range(0, len(content), 1000)]
    # with ProcessPoolExecutor() as executor:
    #     for result in tqdm(executor.map(worker, chunks), total=len(chunks)):
    #         new_content.extend(result)

    return new_content


def filter_invalid_roles(content):
    new_content = []
    for i, c in enumerate(content):
        roles = ["sales rep", "prospect"]
        if len(c["conversations"]) <= 1:
            print("SHORT EXAMPLE: ", len(c["conversations"]))
            continue

        assert(c["conversations"][0]['from'] == 'sales rep')
        assert(c["conversations"][-1]['from'] == 'prospect')
        assert(len(c["conversations"]) % 2 == 0)
        valid = True
        for j, s in enumerate(c["conversations"]):
            if s["from"] != roles[j % 2]:
                valid = False
                break

        assert(valid)
        new_content.append(c)

    return new_content


def main():
    content = json.load(open(args.in_file, "r"))
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        model_max_length=args.max_length,
        padding_side="right",
        use_fast=False,
    )
    new_content = split_all(content, tokenizer, args.max_length)
    new_content = filter_invalid_roles(new_content)

    print(f"#in: {len(content)}, #out: {len(new_content)}")
    json.dump(new_content, open(args.out_file, "w"), indent=2, ensure_ascii=False)



if __name__ == "__main__":
    main()
