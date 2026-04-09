# Resume Training

Training stopped at step 18793 due to SIGHUP when the SSH session dropped (no tmux/nohup protection).
Best checkpoint: `/workspace/checkpoints/best.pt` (val_loss=3.7833 @ step 18750)

## Command

```bash
cd /root/llm-training && nohup uv run python scripts/run_training.py --train-bin /workspace/data/train.bin --val-bin /workspace/data/val.bin --device cuda --use-muon --use-amp --use-compile --early-stopping-patience 10 --checkpoint-dir /workspace/checkpoints --plot-dir /workspace/plots --resume /workspace/checkpoints/best.pt >> /root/llm-training/train.log 2>&1 & echo $!
```

`nohup ... &` detaches the process from the SSH session — it survives disconnects.
`echo $!` prints the PID so you can monitor it with `ps -p <PID>`.

## Check it's running

```bash
ps aux | grep run_training | grep -v grep
```

## Tail logs

```bash
tail -f /root/llm-training/train.log
```
