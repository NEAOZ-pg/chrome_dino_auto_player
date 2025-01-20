import torch
import torch.nn as nn

class Dino_CNN_LSTM_FC(nn.Module):
    def __init__(self, num_classes=3, hidden_size=512, num_layers=1):
        super(Dino_CNN_LSTM_FC, self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 2, kernel_size=3, padding=1),  # 1 channel grayscale -> 4 feature maps
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # Reduce spatial dimensions by 2
            nn.Conv2d(2, 4, kernel_size=3, padding=1),  # 4 feature maps -> 16 feature maps
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)  # Further reduce spatial dimensions
        )
        cnn_output_size = 4 * 32 * 80
        # LSTM layers
        self.lstm = nn.LSTM(cnn_output_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        # Fully connected layer for classification
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),
            nn.ReLU(),
            nn.Linear(hidden_size // 4, hidden_size // 16),
            nn.ReLU(),
            nn.Linear(hidden_size // 16, num_classes)
        )


    def forward(self, x):
        # input torch.size (batch, sequence_length, 128, 320) 
        batch_size, sequence_length, height, width = x.size()
        # Reshape to combine batch and sequence length for CNN input
        x = x.view(batch_size * sequence_length, 1, height, width)  # Shape: (batch_size * sequence_length, 1, 128, 320)
        # Apply CNN in parallel to all images
        cnn_out = self.cnn(x)  # Shape: (batch_size * sequence_length, 4, 32, 80)
        # Flatten CNN output
        cnn_out = cnn_out.view(batch_size, sequence_length, -1)  # Shape: (batch_size, sequence_length, 16*32*80)
        # LSTM processing
        lstm_out, (h_0, c_0) = self.lstm(cnn_out)
        # Fully connected layer to map LSTM output to label
        out = self.fc(lstm_out)  # shape: (batch_size, output_size)
        return out

    def real_time_forward(self, x, h_0, c_0):
        # torch.size    (1, 1, 128, 320)
        cnn_out = self.cnn(x)
        # torch.size    (1, 4, 32, 80)
        cnn_out = cnn_out.view(1, 1, -1)
        lstm_out, (h_0, c_0) = self.lstm(cnn_out, (h_0, c_0))
        out = self.fc(lstm_out)
        return out, h_0, c_0
