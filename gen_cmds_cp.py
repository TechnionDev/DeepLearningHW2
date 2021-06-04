tmpl = 'srun -c 2 --gres=gpu:1 --pty python -m hw2.experiments run-exp -n test -K {K} -L {L} -P 4 -H 100 --epochs 10 --bs-train 50 --batches 10 --early-stopping 5 --run-name exp1_2_L{L}_K{K}'

L = [2,4,8]
K = [32,64,128,256]

cmds = []

for l in L:
	for k in K:
		cmds.append(tmpl.format(K=k, L=l))



with open('./run-exp-1.2.sh', 'w') as f:
	for cmd in cmds:
		f.write(f'{cmd}\n')


import os
os.system('cat ./run-exp.sh')
