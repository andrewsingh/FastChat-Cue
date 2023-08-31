python -m fastchat.serve.model_worker \
    --model-path /home/ubuntu/FastChat-Cue/output/prospect_lm_v6/checkpoint-1282 \
    --model-names prospect-lm-v6 \
    --conv-template prospect_lm \
    --awq-ckpt /home/ubuntu/FastChat-Cue/output/prospect_lm_v6/checkpoint-1282/prospect-lm-v6-w4-g128-awq.pt \
    --awq-wbits 4 \
    --awq-groupsize 128