import torch
import torch.nn as nn
import torch.onnx


# Define or load your model (use the model class you provided)
class Dino_CNN_LSTM_FC(nn.Module):
    def __init__(self, num_classes=2, hidden_size=512, num_layers=1):
        super(Dino_CNN_LSTM_FC, self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 2, kernel_size=3, padding=1),  # 1 channel grayscale -> 4 feature maps
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # Reduce spatial dimensions by 2
            nn.Conv2d(2, 4, kernel_size=3, padding=1),  # 4 feature maps -> 16 feature maps
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)  # Further reduce spatial dimensions
        )

        self.cnn_output_size = 4 * 32 * 80
        # LSTM layers
        self.lstm = nn.LSTM(self.cnn_output_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)

        # Fully connected layer for classification
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),
            nn.ReLU(),
            nn.Linear(hidden_size // 4, hidden_size // 16),
            nn.ReLU(),
            nn.Linear(hidden_size // 16, num_classes)
        )

    def forward(self, x, h_0, c_0):
        cnn_out = self.cnn(x)  # Convolutional layer
        cnn_out = cnn_out.view(1, 1, self.cnn_output_size)  # Reshape output
        lstm_out, (h_0, c_0) = self.lstm(cnn_out, (h_0, c_0))  # LSTM
        out = self.fc(lstm_out)  # Fully connected layer
        return out, h_0, c_0

# Load the model (replace this with your actual model loading code)
model = Dino_CNN_LSTM_FC()
ptname = "epoch_1575_loss_3.503831360153097e-10.pt"
model.load_state_dict(torch.load("ckpt/torch/" + ptname + ".pt", weights_only=True, map_location=torch.device('cpu')))
model.eval()  # Set model to evaluation mode

# Prepare dummy inputs
x = torch.randn(1, 1, 128, 320)
h_0 = torch.zeros(1, 1, 512)  # LSTM initial hidden state
c_0 = torch.zeros(1, 1, 512)  # LSTM initial cell state

# Export the model to ONNX
onnx_path = "ckpt/onnx/" + ptname + ".onnx"
torch.onnx.export(
    model, 
    (x, h_0, c_0),  # Input tuple, corresponding to (x, h_0, c_0)
    onnx_path,
    input_names=["x", "h_0", "c_0"],  # Name of the input layers
    output_names=["out", "h_0_out", "c_0_out"],  # Name of the output layers
    opset_version=12,  # ONNX opset version, you can try 11 or 13 depending on your setup
    dynamic_axes={}
)

print(f"Model successfully exported to {onnx_path}")
