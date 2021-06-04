tmpl = 'srun -c 2 --gres=gpu:1 --pty python -m hw2.experiments run-exp -n test -K {K} -L {L} -P 12 -H 100 -M ycn --epochs 10 --bs-train 50 --batches 10 --early-stopping 5 --run-name exp2_L{L}_K32-64-128'

K = ['32 64 128']
L = [3,6,9,12]

cmds = []
for k in K:
	for l in L:
		cmds.append(tmpl.format(K=k, L=l))



with open('./run-exp-2.sh', 'w') as f:
	for cmd in cmds:
		f.write(f'{cmd}\n')


import os
os.system('cat ./run-exp-2.sh')
