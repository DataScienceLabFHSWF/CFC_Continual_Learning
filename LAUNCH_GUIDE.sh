#!/bin/bash
# ============================================================================
# Paper Benchmarks - VPN-Safe Launch Guide
# ============================================================================

cat << 'EOF'
╔════════════════════════════════════════════════════════════════════════════╗
║                 CfC Paper Benchmarks - VPN-Safe Setup                      ║
╚════════════════════════════════════════════════════════════════════════════╝

This guide helps you run benchmarks that survive VPN disconnects.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📋 STEP 1: Launch Benchmarks (VPN-Safe)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Choose your dataset and run:

  # MNIST (fastest - recommended for overnight, ~56 hours with 4 GPUs)
  ./launch_paper_benchmarks.sh mnist 4

  # CIFAR-10 (longest - ~150 hours with 4 GPUs)
  ./launch_paper_benchmarks.sh cifar10 4

  # TEP (medium - ~60 hours with 4 GPUs)
  ./launch_paper_benchmarks.sh tep 4

  # All datasets (~266 hours with 4 GPUs)
  ./launch_paper_benchmarks.sh all 4

This creates:
  - 1 orchestrator session (paper_orchestrator)
  - Multiple experiment sessions (paper_mnist_mlp_sgd_s0, etc.)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
👀 STEP 2: Monitor Progress
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

After launching, you can monitor with:

  # Quick dashboard (refresh every 30 seconds)
  watch -n 30 ./monitor_benchmarks.sh

  # Or run once
  ./monitor_benchmarks.sh

  # Attach to orchestrator
  tmux attach -t paper_orchestrator

  # List all running experiments
  tmux ls | grep paper_

  # View specific experiment
  tmux attach -t paper_mnist_mlp_sgd_s0

  # Detach from tmux (keeps running!)
  Press: Ctrl+b, then d

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📊 STEP 3: Check WandB Dashboard
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

All experiments are logged to WandB in real-time:

  https://wandb.ai/fneubuerger/mammoth

You can:
  - View live training curves
  - Compare experiments
  - See accuracy metrics
  - Download results

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🔌 What Happens if VPN Disconnects?
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

✅ Everything keeps running!

  - Tmux sessions persist on the server
  - Experiments continue training
  - Results keep logging to WandB
  - Log files keep updating

When you reconnect:
  1. SSH back into server
  2. Run: tmux ls
  3. You'll see all your sessions still running!

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📁 Where Are Results Saved?
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Logs:       paper_results/logs/*.log
  CSV data:   paper_results/*.csv
  WandB:      https://wandb.ai/fneubuerger/mammoth

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🛑 How to Stop Everything?
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  # Kill orchestrator
  tmux kill-session -t paper_orchestrator

  # Kill all paper benchmark sessions
  tmux ls | grep paper_ | cut -d: -f1 | xargs -I {} tmux kill-session -t {}

  # Or kill specific experiment
  tmux kill-session -t paper_mnist_mlp_sgd_s0

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⏱️  Runtime Estimates (4 GPUs in parallel)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  MNIST:     ~56 hours   (90 experiments,  10 epochs each)
  CIFAR-10:  ~150 hours  (60 experiments,  50 epochs each)
  TEP:       ~60 hours   (48 experiments,  20 epochs each)
  
  Total:     ~266 hours  (11 days)

Recommended overnight run: MNIST
  - Finishes in ~2.5 days
  - 90 experiments total
  - Validates infrastructure
  - Quick results for paper

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🚀 Ready to Launch?
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Run this now:

    ./launch_paper_benchmarks.sh mnist 4

Then you can:
  - Close your laptop
  - Disconnect VPN
  - Go home for the night
  - Check WandB from anywhere

Everything will keep running! 🎉

╚════════════════════════════════════════════════════════════════════════════╝
EOF
