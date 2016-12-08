import subprocess
from subprocess import PIPE, Popen
import time

# Change process call here
commands = ["./darknet", "classifier", "predict", "cfg/faces.data", "cfg/dec4_conv5_bn.cfg", "backup/dec4_conv5_bn_5.weights"]

p = subprocess.Popen(commands, stdin=PIPE)

# Wait for darknet initial loading
time.sleep(2)

# Main Engine
while (True):
	print >> p.stdin, "data/dog.jpg"
	# Wait for detection ~0.03 on my machine
	time.sleep(0.08)
