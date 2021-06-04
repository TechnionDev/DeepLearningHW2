srun -c 2 --gres=gpu:1 --pty python -m hw2.experiments run-exp -n test -K 32 -L 2 -P 4 -H 100 --epochs 10 --bs-train 50 --batches 10 --early-stopping 5 --run-name exp1_1_L2_K32
srun -c 2 --gres=gpu:1 --pty python -m hw2.experiments run-exp -n test -K 32 -L 4 -P 4 -H 100 --epochs 10 --bs-train 50 --batches 10 --early-stopping 5 --run-name exp1_1_L4_K32
srun -c 2 --gres=gpu:1 --pty python -m hw2.experiments run-exp -n test -K 32 -L 8 -P 4 -H 100 --epochs 10 --bs-train 50 --batches 10 --early-stopping 5 --run-name exp1_1_L8_K32
srun -c 2 --gres=gpu:1 --pty python -m hw2.experiments run-exp -n test -K 32 -L 16 -P 4 -H 100 --epochs 10 --bs-train 50 --batches 10 --early-stopping 5 --run-name exp1_1_L16_K32
srun -c 2 --gres=gpu:1 --pty python -m hw2.experiments run-exp -n test -K 64 -L 2 -P 4 -H 100 --epochs 10 --bs-train 50 --batches 10 --early-stopping 5 --run-name exp1_1_L2_K64
srun -c 2 --gres=gpu:1 --pty python -m hw2.experiments run-exp -n test -K 64 -L 4 -P 4 -H 100 --epochs 10 --bs-train 50 --batches 10 --early-stopping 5 --run-name exp1_1_L4_K64
srun -c 2 --gres=gpu:1 --pty python -m hw2.experiments run-exp -n test -K 64 -L 8 -P 4 -H 100 --epochs 10 --bs-train 50 --batches 10 --early-stopping 5 --run-name exp1_1_L8_K64
srun -c 2 --gres=gpu:1 --pty python -m hw2.experiments run-exp -n test -K 64 -L 16 -P 4 -H 100 --epochs 10 --bs-train 50 --batches 10 --early-stopping 5 --run-name exp1_1_L16_K64
