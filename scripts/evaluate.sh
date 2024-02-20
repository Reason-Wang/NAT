model_path="reasonwang/NAT-math-7b"
template="zero-shot-target-aware"
eval_task="gsm8k"

cmd="python -m data.generate \
      --model ${model_path} \
      --template ${template} \
      --temperature 0.2 \
      --task_name ${eval_task} \
      --task_split test"
echo $cmd
eval $cmd
