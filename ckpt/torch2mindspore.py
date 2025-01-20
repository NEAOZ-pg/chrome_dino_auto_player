import os
import sys

ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_PATH)

import numpy as np
import cv2
import torch
from models.mindspore.Dino_CNN_LSTM_FC import Dino_CNN_LSTM_FC as Mindspore_Dino_CNN_LSTM_FC
from mindspore import load_checkpoint, load_param_into_net
import mindspore

mindspore.context.set_context(device_target="CPU")

pt_name = r"LY_epoch_1867_loss_0.002432126324856654.pt"

class Dino_CNN_LSTM_FC(torch.nn.Module):
    def __init__(self, num_classes=2, hidden_size=512, num_layers=1):
        super(Dino_CNN_LSTM_FC, self).__init__()

        self.cnn = torch.nn.Sequential(
            torch.nn.Conv2d(1, 2, kernel_size=3, padding=1),  # 1 channel grayscale -> 4 feature maps
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2),  # Reduce spatial dimensions by 2
            torch.nn.Conv2d(2, 4, kernel_size=3, padding=1),  # 4 feature maps -> 16 feature maps
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2)  # Further reduce spatial dimensions
        )

        self.cnn_output_size = 4 * 32 * 80
        # LSTM layers
        self.lstm = torch.nn.LSTM(self.cnn_output_size, hidden_size=hidden_size, 
                                  num_layers=num_layers, batch_first=True)

        # Fully connected layer for classification
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, hidden_size // 4),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size // 4, hidden_size // 16),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size // 16, num_classes)
        )

    def forward(self, x, h_0, c_0):
        cnn_out = self.cnn(x)  # Convolutional layer
        cnn_out = cnn_out.view(1, 1, self.cnn_output_size)  # Reshape output
        lstm_out, (h_0, c_0) = self.lstm(cnn_out, (h_0, c_0))  # LSTM
        out = self.fc(lstm_out)  # Fully connected layer
        return out, h_0, c_0

# load torch model
torch_model = Dino_CNN_LSTM_FC()
torch_model.load_state_dict(torch.load(r'ckpt/torch/'+pt_name, weights_only=True, map_location=torch.device('cpu')))
weights = torch_model.state_dict()
torch_model = torch_model.to(device="cpu")

# load mindspore model
mindspore_model = Mindspore_Dino_CNN_LSTM_FC()

# load mindspore weight
def load_pytorch_weights_to_mindspore(pytorch_weights, mindspore_model):
    mindspore_weights = mindspore_model.parameters_dict()
    
    for name, param in pytorch_weights.items():
        if name in mindspore_weights:
            ms_param = mindspore_weights[name]
            ms_param.set_data(mindspore.Tensor(param.numpy()))  # 将 PyTorch 权重转换为 MindSpore Tensor

    return mindspore_model

mindspore_model = load_pytorch_weights_to_mindspore(weights, mindspore_model)

# eval

# init
x = np.zeros((1, 1, 128, 320))

x = cv2.imread('ckpt/Figure_0.png', cv2.IMREAD_GRAYSCALE)
_, x = cv2.threshold(x, 127, 255, cv2.THRESH_BINARY)
x = x.reshape(1, 1, 128, 320).astype(np.float32)

h_0 = np.zeros((1, 1, 512))
c_0 = np.zeros((1, 1, 512))

x_mindspore = mindspore.Tensor(x, dtype=mindspore.float32)
h_0_mindspore = mindspore.Tensor(h_0, dtype=mindspore.float32)
c_0_mindspore = mindspore.Tensor(c_0, dtype=mindspore.float32)

x_torch = torch.tensor(x, dtype=torch.float32).to(device="cpu")
h_0_torch = torch.tensor(h_0, dtype=torch.float32).to(device="cpu")
c_0_torch = torch.tensor(c_0, dtype=torch.float32).to(device="cpu")

import time
mindspore_start = time.time()
output_mindspore, h_0_out_mindspore, c_0_out_mindspore = mindspore_model(x_mindspore, h_0_mindspore, c_0_mindspore)
mindspore_stop = time.time()
output_torch, h_0_torch, c_0_torch = torch_model(x_torch, h_0_torch, c_0_torch)
torch_stop = time.time()

print("--------time eval----------")
print("mindspore: \n", mindspore_stop - mindspore_start)
print("torch:\n", torch_stop - mindspore_stop)

print("--------data verify---------")
print("mindspore: \n", output_mindspore)
print("torch:\n", output_torch)

# save as mindspore format
checkpoint_path = "/ckpt/mindspore/" + pt_name
mindspore.save_checkpoint(mindspore_model, checkpoint_path)

