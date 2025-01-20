import mindspore

class Dino_CNN_LSTM_FC(mindspore.nn.Cell):
    def __init__(self, num_classes=2, hidden_size=512, num_layers=1):
        super(Dino_CNN_LSTM_FC, self).__init__()

        # 定义卷积层
        self.cnn = mindspore.nn.SequentialCell(
            mindspore.nn.Conv2d(1, 2, kernel_size=3, padding=1 , pad_mode='pad', has_bias=True),  # Using pad_mode='same' to retain size
            mindspore.nn.ReLU(),
            mindspore.nn.MaxPool2d(kernel_size=2, stride=2, pad_mode='valid'),  # MaxPool with stride=2 to reduce dimensions
            mindspore.nn.Conv2d(2, 4, kernel_size=3, padding=1, pad_mode='pad', has_bias=True),  # Retain size with pad_mode='same'
            mindspore.nn.ReLU(),
            mindspore.nn.MaxPool2d(kernel_size=2, stride=2, pad_mode='valid')  # Another MaxPool layer
        )

        self.cnn_output_size = 4 * 32 * 80  # CNN输出的维度

        # 定义LSTM层
        self.lstm = mindspore.nn.LSTM(input_size=self.cnn_output_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)

        # 定义全连接层
        self.fc = mindspore.nn.SequentialCell(
            mindspore.nn.Dense(hidden_size, hidden_size // 4),
            mindspore.nn.ReLU(),
            mindspore.nn.Dense(hidden_size // 4, hidden_size // 16),
            mindspore.nn.ReLU(),
            mindspore.nn.Dense(hidden_size // 16, num_classes)
        )

    def construct(self, x, h_0, c_0):
        cnn_out = self.cnn(x)  # 通过卷积层
        cnn_out = cnn_out.view((1, 1, self.cnn_output_size))  # 展平输出为合适的形状
        lstm_out, (h_0, c_0) = self.lstm(cnn_out, (h_0, c_0))  # 通过LSTM层
        out = self.fc(lstm_out)  # 通过全连接层进行分类
        return out, h_0, c_0
    
