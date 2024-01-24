import torch

torch.manual_seed(42)
import random

random.seed(42)
import numpy as np

np.random.seed(0)

print("Single GPU test")

for i in range(torch.cuda.device_count()):
    # if i in [0,1,2,3]:
    #     continue
    print(i)
    print(i, torch.cuda.get_device_name(i))
    device = torch.device(f"cuda:{i}")
    activations = torch.randn(4, 4).to(device)  # usually we get our activations in a more refined way...
    labels = torch.arange(4).to(device)
    loss = torch.nn.functional.cross_entropy(activations, labels)
    average = loss / 4
    print(average.item())

    # print device memory
    # print(torch.cuda.memory_allocated(i))
    # print(torch.cuda.memory_cached(i))
    print(torch.cuda.get_device_properties(i).total_memory)
    # check device max memory possible

print("Going for multiprocessing test")
# multiprocessing test
model = torch.nn.Linear(10, 10)
testdata = torch.randn(1000, 10)
testlabels = torch.randn(1000, 10)
model = torch.nn.DataParallel(model, device_ids=list(range(torch.cuda.device_count()))).cuda()
print(model)
out = model(testdata)
print(out)
