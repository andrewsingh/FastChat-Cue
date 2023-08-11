import argparse
import json
import os
import re
import requests
import torch
import whisper
import locale
import time
import pdb

from pathlib import Path
from pydub import AudioSegment
from pyannote.audio import Pipeline
from tqdm import tqdm


locale.getpreferredencoding = lambda: "UTF-8"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
access_token = "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"

parser = argparse.ArgumentParser()
parser.add_argument("--input-json", type=str, default=None)
parser.add_argument("--data-dir", type=str)
parser.add_argument("--output-dir", type=str)
parser.add_argument("--debug", action='store_true')
parser.add_argument("--download-only", action='store_true')
args = parser.parse_args()

if not args.download_only:
    pipeline = Pipeline.from_pretrained('pyannote/speaker-diarization', use_auth_token=access_token)
    pipeline.to(device)
    model = whisper.load_model('large-v2', device=device)


def download_file(url, filename):
    """
    Download an MP3 file from a URL.

    :param url: The URL of the file to download.
    :param filename: The name of the file to save locally.
    """
    response = requests.get(url)
    response.raise_for_status()  # Raise an error for bad responses
    
    with open(filename, 'wb') as file:
        file.write(response.content)


def millisec(timeStr):
    spl = timeStr.split(":")
    s = (int)((int(spl[0]) * 60 * 60 + int(spl[1]) * 60 + float(spl[2]) )* 1000)
    return s


def group_dz(dz_filename):
    dzs = open(dz_filename).read().splitlines()

    groups = []
    g = []
    lastend = 0

    for d in dzs:   
        if g and (g[0].split()[-1] != d.split()[-1]):      #same speaker
            groups.append(g)
            g = []
    
        g.append(d)
    
        end = re.findall('[0-9]+:[0-9]+:[0-9]+\.[0-9]+', string=d)[1]
        end = millisec(end)
        if (lastend > end):       #segment engulfed by a previous segment
            groups.append(g)
            g = [] 
        else:
            lastend = end
    if g:
        groups.append(g)

    # print(*groups, sep='\n')
    return groups
    

def transcribe_call(call):
    call_id = call['metaData']['id']
    print(f"Transcribing call {call_id}")
    audio_url = call['media']['audioUrl']
    original_filename = Path(args.data_dir) / call_id / "original.mp3"
    if not os.path.exists(original_filename):
        os.makedirs(original_filename.parent, exist_ok=True)
        print("Downloading MP3")
        download_file(audio_url, original_filename)
    if args.download_only:
        return
    print("Preparing audio")
    spacermilli = 2000
    spacer = AudioSegment.silent(duration=spacermilli)
    audio = AudioSegment.from_mp3(original_filename) 
    print(f"Duration: {audio.duration_seconds} seconds")
    audio = spacer.append(audio, crossfade=0)

    input_prep_filename = Path(args.data_dir) / call_id / "input_prep.wav"
    audio.export(input_prep_filename, format='wav')

    print("Running diarization")
    pipeline_file = {'uri': 'blabla', 'audio': input_prep_filename}
    dz = pipeline(pipeline_file)  

    dz_filename = Path(args.data_dir) / call_id / "diarization.txt"
    with open(dz_filename, "w") as text_file:
        text_file.write(str(dz))

    print("Exporting audio segments")
    groups = group_dz(dz_filename)
    audio = AudioSegment.from_wav(input_prep_filename)
    gidx = -1
    for g in groups:
        start = re.findall('[0-9]+:[0-9]+:[0-9]+\.[0-9]+', string=g[0])[0]
        end = re.findall('[0-9]+:[0-9]+:[0-9]+\.[0-9]+', string=g[-1])[1]
        start = millisec(start) #- spacermilli
        end = millisec(end)  #- spacermilli
        gidx += 1
        group_filename = Path(args.data_dir) / call_id / "audio_segments" / (str(gidx) + '.wav')
        os.makedirs(group_filename.parent, exist_ok=True)
        audio[start:end].export(group_filename, format='wav')
        # print(f"group {gidx}: {start}--{end}")

    del audio, dz

    print("Transcribing segments with Whisper")
    lines = []
    for i in tqdm(range(len(groups))):
        group_filename_str = str(Path(args.data_dir) / call_id / "audio_segments" / (str(i) + '.wav'))
        result = model.transcribe(audio=group_filename_str, language='en', word_timestamps=True)
        transcript_filename = Path(args.data_dir) / call_id / "audio_transcripts" / (str(i)+'.json')
        os.makedirs(transcript_filename.parent, exist_ok=True)
        with open(transcript_filename, "w") as outfile:
            json.dump(result, outfile, indent=4)  
        
        speaker = 0 if i % 2 == 0 else 1
        lines.append({
            "speaker": speaker,
            "text": result['text'].strip()
        })
            
    lines_filename = Path(args.data_dir) / call_id / "final_transcript.json"
    lines_filename_2 = Path(args.output_dir) / f"{call_id}.json"

    os.makedirs(lines_filename.parent, exist_ok=True)
    os.makedirs(lines_filename_2.parent, exist_ok=True)

    with open(lines_filename, "w") as f:
        json.dump(lines, f, indent=4)

    with open(lines_filename_2, "w") as f:
        json.dump(lines, f, indent=4)

    print("Complete!")
    


def main():
    with open(args.input_json, "r") as f:
        calls = json.load(f)

    for i, call in enumerate(calls):
        print(f"{'=' * 32} CALL {i + 1} OF {len(calls)} {'=' * 32}")
        try:
            lines_filename_2 = Path(args.output_dir) / f"{call['metaData']['id']}.json"
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