tmpl = 'srun -c 2 --gres=gpu:1 --pty python -m hw2.experiments run-exp -n test -K {K} -L {L} -P 8 -H 100 -M resnet --epochs 10 --bs-train 50 --batches 10 --early-stopping 5 --run-name exp1_4_L{L}_K32'

K = [32]
L = [8,16,32]

cmds = []
for k in K:
	for l in L:
		cmds.append(tmpl.format(K=k, L=l))



with open('./run-exp-1.4.1.sh', 'w') as f:
	for cmd in cmds:
		f.write(f'{cmd}\n')


import os
os.system('cat ./run-exp-1.4.1.sh')
