python -m fastchat.serve.model_worker \
    --model-path /home/ubuntu/FastChat-Cue/output/checkpoint-1080-full-model \
    --model-names prospect-lm-v5 \
    --conv-template prospect_lm \
    --awq-ckpt /home/ubuntu/FastChat-Cue/output/checkpoint-1080-full-model/awq-model-w4-g128.pt \
    --awq-wbits 4 \
    --awq-groupsize 128