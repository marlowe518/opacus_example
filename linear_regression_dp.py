import numpy as np
import torch
from torch.utils import data  # TODO
from d2l import torch as d2l

true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = d2l.synthetic_data(true_w, true_b, 1000)


# train_iter
def load_data(data_array, batch_size):
    # dataset = data.TensorDataset(data_array) # TODO
    dataset = data.TensorDataset(*data_array)  # TODO when use *?
    return data.DataLoader(dataset, batch_size=batch_size, shuffle=True)


batch_size = 10
data_iter = load_data((features, labels), batch_size)

from opacus.data_loader import DPDataLoader
dp_data_loader = DPDataLoader.from_data_loader(data_iter, distributed=False)

# params
lr = 0.01
num_epochs = 20

# model
from torch import nn
from opacus import GradSampleModule
from opacus.privacy_engine import forbid_accumulation_hook

net = nn.Sequential(nn.Linear(2, 1, bias=True))
net = GradSampleModule(net)
net.register_forward_pre_hook(forbid_accumulation_hook)

# initialize
net[0].weight.data.normal_(0, 0.01)  # replace
net[0].bias.data.fill_(0)

# optimizer
optimizer = torch.optim.SGD(params=net.parameters(), lr=lr)
from opacus.optimizers import DPOptimizer
sample_rate = 1 / len(dp_data_loader) # what's the difference to dataset
expected_batch_size = int(len(dp_data_loader.dataset) * sample_rate)
optimizer = DPOptimizer(
    optimizer=optimizer,
    noise_multiplier=1.0,
    max_grad_norm=1.0,
    expected_batch_size=expected_batch_size,
)
from opacus.accountants import RDPAccountant
accountant = RDPAccountant
optimizer.attach_step_hook(accountant.get_optimizer_hook_fn(sample_rate=sample_rate))


# loss
loss = nn.MSELoss()

# train
for epoch in range(num_epochs):
    for X, y in data_iter:
        # loss = net(X) # TODO
        l = loss(net(X), y)
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
    # l = loss(net(features), labels)  # TODO all data
    print(f"epoch:{epoch}, loss:{l}")

w = net[0].weight.data
b = net[0].bias.data
print(f"w:{w}, b:{b}")

# test
