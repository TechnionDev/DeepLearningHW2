srun -c 2 --gres=gpu:1 --pty python -m hw2.experiments run-exp -n test -K 64 128 256 -L 1 -P 4 -H 100 --epochs 10 --bs-train 50 --batches 10 --early-stopping 5 --run-name exp1_3_L1_K64-128-256
srun -c 2 --gres=gpu:1 --pty python -m hw2.experiments run-exp -n test -K 64 128 256 -L 2 -P 4 -H 100 --epochs 10 --bs-train 50 --batches 10 --early-stopping 5 --run-name exp1_3_L2_K64-128-256
srun -c 2 --gres=gpu:1 --pty python -m hw2.experiments run-exp -n test -K 64 128 256 -L 3 -P 4 -H 100 --epochs 10 --bs-train 50 --batches 10 --early-stopping 5 --run-name exp1_3_L3_K64-128-256
srun -c 2 --gres=gpu:1 --pty python -m hw2.experiments run-exp -n test -K 64 128 256 -L 4 -P 4 -H 100 --epochs 10 --bs-train 50 --batches 10 --early-stopping 5 --run-name exp1_3_L4_K64-128-256
