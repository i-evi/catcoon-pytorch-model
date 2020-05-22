import os
import torch
import torch.nn as nn
from ctypes import *

bbuffer = CDLL("../bbuffer.so")
bbuffer.write_float32.argtypes = (c_float,)
bbuffer.write_float32.restypes = c_float

class classifier(nn.Module):
	pass
C = torch.load("classifier.pkl")

def dump_parameters(state_dict, name, filepath, filename):
	current_file = "%s/%s"%(filepath, filename)
	bbuffer.init(current_file.encode())
	para = state_dict[name].numpy().reshape(-1)
	for elem in para:
		bbuffer.write_float32(elem)
	bbuffer.close()

state_dict = C.state_dict()

parameters_files = [
	"conv1_w.bin",
	"conv1_b.bin",
	"conv2_w.bin",
	"conv2_b.bin",
	"fc1_w.bin",
	"fc1_b.bin",
	"fc2_w.bin",
	"fc2_b.bin",
]

parameters_filepath = "./dump"

if not os.path.exists(parameters_filepath):
	os.makedirs(parameters_filepath)  

for i, name in enumerate(state_dict):
	dump_parameters(state_dict, name,
		parameters_filepath, parameters_files[i])
