import os
import sys

sys.path.append(os.path.dirname((os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from models.torch.Dino_CNN_LSTM_FC import Dino_CNN_LSTM_FC
from utils.Dino_Dataset_Series import Dino_Dataset_Series
import tqdm


if __name__ == "__main__":
    # replace the path below
    model_path = "/Users/neaoz/Desktop/chrome_dino/train_test/log/epoch_255_loss_0.11676250845193863.pt"
    dino_dataset = Dino_Dataset_Series(r"/Users/neaoz/Desktop/chrome_dino/data/dataset/csv_path.csv")

    device = torch.device("mps")
    # device = torch.device("cuda" if torch.cuda.is_available() else if"cpu")
    print(f"Using device: {device}")
    
    dino_dataloader = DataLoader(dataset=dino_dataset, batch_size=1, shuffle=True)

    hidden_size = 512
    model = Dino_CNN_LSTM_FC(num_classes=3, hidden_size=hidden_size)
    model.load_state_dict(torch.load(model_path))
    model.to(device).eval()

    criterion = torch.nn.CrossEntropyLoss()

    print(f'#------------------Testing-----------------#')
    epoch_correct = 0
    total_num = 0
    epoch_loss = 0.0
    pbar = tqdm.tqdm(dino_dataloader)
    for dino_image, label in pbar:
        dino_image = dino_image.to(device)
        label = label.to(device)

        pred = model(dino_image)
        loss = criterion(pred.view(-1, pred.size(-1)), label.view(-1))

        epoch_correct += torch.eq(pred.argmax(dim=-1), label).sum().item()
        epoch_loss += loss.item()

        total_num += len(label.view(-1))
        pbar.set_description('Test Loss = {:f}'.format(loss.item()))

    print("correct num = ", epoch_correct)
    print("correct rate = ", epoch_correct / total_num)
