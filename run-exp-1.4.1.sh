srun -c 2 --gres=gpu:1 --pty python -m hw2.experiments run-exp -n test -K 32 -L 8 -P 8 -H 100 -M resnet --epochs 10 --bs-train 50 --batches 10 --early-stopping 5 --run-name exp1_4_L8_K32
srun -c 2 --gres=gpu:1 --pty python -m hw2.experiments run-exp -n test -K 32 -L 16 -P 8 -H 100 -M resnet --epochs 10 --bs-train 50 --batches 10 --early-stopping 5 --run-name exp1_4_L16_K32
srun -c 2 --gres=gpu:1 --pty python -m hw2.experiments run-exp -n test -K 32 -L 32 -P 8 -H 100 -M resnet --epochs 10 --bs-train 50 --batches 10 --early-stopping 5 --run-name exp1_4_L32_K32
