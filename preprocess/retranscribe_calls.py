import argparse
import json
import os
import re
import requests
import torch
import locale
import time
import pdb
import shutil
import openai

from pathlib import Path
from pydub import AudioSegment
from tqdm import tqdm


locale.getpreferredencoding = lambda: "UTF-8"
device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 16
access_token = "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"

parser = argparse.ArgumentParser()
parser.add_argument("--input-json", type=str, default=None)
parser.add_argument("--data-dir", type=str)
parser.add_argument("--output-dir", type=str)
parser.add_argument("--use-api", action='store_true')
parser.add_argument("--debug", action='store_true')
args = parser.parse_args()


if args.use_api:
    import openai
else:
    import whisperx
    model = whisperx.load_model('large-v2', device=device)
    
    

def transcribe_call(call):
    # initial_prompt = "Corrected for spelling discrepancies and mistakes in grammar and syntax, and filler words removed: "
    initial_prompt = None
    call_id = call['metadata']['id']
    print(f"Transcribing call {call_id}")
    original_filename = Path(args.data_dir) / call_id / "original.mp3"
    audio = AudioSegment.from_mp3(original_filename)
    segment_dir = Path(args.data_dir) / call_id / "audio_segments_v2"
    if os.path.exists(segment_dir):
        shutil.rmtree(segment_dir)

    for i, segment in enumerate(call['segments']):
        segment_filename = Path(args.data_dir) / call_id / "audio_segments_v2" / (str(i) + '.wav')
        os.makedirs(segment_filename.parent, exist_ok=True)
        audio_segment = audio[segment['start']:segment['end']]
        audio_segment.export(segment_filename, format='wav')

    new_segments = []
    for i, segment in tqdm(enumerate(call['segments']), total=len(call['segments'])):
        segment_filename_str = str(Path(args.data_dir) / call_id / "audio_segments_v2" / (str(i) + '.wav'))
        if args.use_api:
            audio_file = open(segment_filename_str, "rb")
            segment_result = openai.Audio.transcribe("whisper-1", audio_file, initial_prompt=initial_prompt)
            segment_text = segment_result['text']
        else:
            segment_result = model.transcribe(audio=segment_filename_str, language='en', batch_size=batch_size, initial_prompt=initial_prompt)
            segment_text = " ".join([segment['text'].strip() for segment in segment_result['segments']])
        segment['text'] = segment_text
        new_segments.append(segment)

    output_filename = Path(args.data_dir) / call_id / "call_retranscribed_v3.json"
    output_filename_2 = Path(args.output_dir) / f"{call_id}.json"

    os.makedirs(output_filename.parent, exist_ok=True)
    os.makedirs(output_filename_2.parent, exist_ok=True)

    call['segments'] = new_segments

    with open(output_filename, "w") as f:
        json.dump(call, f, indent=4)

    with open(output_filename_2, "w") as f:
        json.dump(call, f, indent=4)

    print("Complete!")
    


def main():
    with open(args.input_json, "r") as f:
        calls = json.load(f)

    for i, call in enumerate(calls):
        print(f"{'=' * 32} CALL {i + 1} OF {len(calls)} {'=' * 32}")
        try:
            lines_filename_2 = Path(args.output_dir) / f"{call['metadata']['id']}.json"
            if os.path.exists(lines_filename_2):
                print("Transcript already exists, skipping call")
            else:
                t0 = time.time()
                transcribe_call(call)
                t1 = time.time()
                print(f"Time elapsed: {t1 - t0} seconds")
        except Exception as e:
            print(e)



if __name__ == "__main__":
    main()