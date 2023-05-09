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

# params
lr = 0.01
num_epochs = 20

# model
from torch import nn

net = nn.Sequential(nn.Linear(2, 1))

# initialize
net[0].weight.data.normal_(0, 0.01)  # replace
net[0].bias.data.fill_(0)

from opacus.validators import ModuleValidator

errors = ModuleValidator.validate(net, strict=False)
print(errors[-5:])
net = ModuleValidator.fix(net)
ModuleValidator.validate(net, strict=False)


# optimizer
optimizer = torch.optim.SGD(params=net.parameters(), lr=lr)

from opacus import PrivacyEngine

privacy_engine = PrivacyEngine(accountant="rdp")
net, optimizer, data_iter = privacy_engine.make_private_with_epsilon(
    module=net,
    optimizer=optimizer,
    data_loader=data_iter,
    epochs=20,
    target_epsilon=10,
    target_delta=0.001,
    max_grad_norm=5.0,
)

# loss
loss = nn.MSELoss()

# train
for epoch in range(num_epochs):
    for X, y in data_iter:
        # loss = net(X) # TODO
        optimizer.zero_grad()
        l = loss(net(X), y)
        # optimizer.zero_grad()
        l.backward()
        optimizer.step()
    # l = loss(net(features), labels)  # TODO all data
    print(f"epoch:{epoch}, loss:{l}")
    epsilon = privacy_engine.get_epsilon(0.00001)
    print(f"epsilon:{epsilon}, delta:{0.00001}")

w = net[0].weight.data
b = net[0].bias.data
print(f"w:{w}, b:{b}")

# test
