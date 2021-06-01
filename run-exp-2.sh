srun -c 2 --gres=gpu:1 --pty python -m hw2.experiments run-exp -n test -K 32 64 128 -L 3 -P 4 -H 100 -M ycn --epochs 10 --bs-train 50 --batches 10 --early-stopping 5 --run-name exp2_L3_K32-64-128
srun -c 2 --gres=gpu:1 --pty python -m hw2.experiments run-exp -n test -K 32 64 128 -L 6 -P 4 -H 100 -M ycn --epochs 10 --bs-train 50 --batches 10 --early-stopping 5 --run-name exp2_L6_K32-64-128
srun -c 2 --gres=gpu:1 --pty python -m hw2.experiments run-exp -n test -K 32 64 128 -L 9 -P 4 -H 100 -M ycn --epochs 10 --bs-train 50 --batches 10 --early-stopping 5 --run-name exp2_L9_K32-64-128
srun -c 2 --gres=gpu:1 --pty python -m hw2.experiments run-exp -n test -K 32 64 128 -L 12 -P 4 -H 100 -M ycn --epochs 10 --bs-train 50 --batches 10 --early-stopping 5 --run-name exp2_L12_K32-64-128
