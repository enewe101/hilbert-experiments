import torch
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Using device: {}'.format(DEVICE.type))