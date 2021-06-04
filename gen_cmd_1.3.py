tmpl = 'srun -c 2 --gres=gpu:1 --pty python -m hw2.experiments run-exp -n test -K {K} -L {L} -P 4 -H 100 --epochs 10 --bs-train 50 --batches 10 --early-stopping 5 --run-name exp1_3_L{L}_K64-128-256'

K = "64 128 256"
L = [1,2,3,4]

cmds = []

for l in L:
	cmds.append(tmpl.format(K=K, L=l))



with open('./run-exp-1.3.sh', 'w') as f:
	for cmd in cmds:
		f.write(f'{cmd}\n')


import os
os.system('cat ./run-exp-1.3.sh')
