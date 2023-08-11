import torch
from torch import nn
from torchvision import models
import torch.optim as optim

import deepdrr
from deepdrr.projector import Projector # initializes PyCUDA

# Before creating a Projector, run backprop to initialize PyTorch
criterion = nn.CrossEntropyLoss()
model = models.resnet50() # Your model here
model.cuda()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
optimizer.zero_grad()
x = torch.ones((32, 3, 224, 224), dtype=torch.float32).cuda() # Your image size
y = torch.ones(32, dtype=torch.int64).cuda()
y_pred = model(x)
loss = criterion(y_pred, y)
loss.backward()
optimizer.step()
# log.info(f"Ran dummy batch to initialize torch.")

volume = ...
carm = ...
with Projector(volume, carm=carm):
  image = projector()
  image = image.unsqueeze(0) # add batch dim
  y_pred = model(image)
  ...

print('ok')