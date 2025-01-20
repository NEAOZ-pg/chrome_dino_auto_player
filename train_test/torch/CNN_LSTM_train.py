import os
import sys

sys.path.append(os.path.dirname((os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from models.torch.Dino_CNN_LSTM_FC import Dino_CNN_LSTM_FC
from utils.Dino_Dataset_Series import Dino_Dataset_Series
import tqdm
import pandas as pd

if __name__ == "__main__":
    SAVE_PATH = r"log"
    #replace the path below
    dino_dataset = Dino_Dataset_Series(r"/Users/neaoz/Desktop/chrome_dino/data/dataset/csv_path.csv")

    device = torch.device("mps")
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
        
    dino_dataloader = DataLoader(dataset=dino_dataset, batch_size=1, shuffle=True)

    hidden_size = 512
    model = Dino_CNN_LSTM_FC(num_classes=2, hidden_size=hidden_size, num_layers=1).to(device).train()
    class_weights = torch.tensor([1/32, 1], dtype=torch.float).to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=1e-8, max_lr=1e-3, step_size_up=len(dino_dataset)*10, step_size_down=len(dino_dataset)*5, 
                                                  mode='triangular', scale_mode='cycle', cycle_momentum=False)

    loss_csv = []
    os.makedirs(SAVE_PATH, exist_ok=True)

    for epoch in range(2048):
        print(f'#---------Training epoch: {epoch}-----------#')
        epoch_correct = 0
        epoch_loss = 0.0
        pbar = tqdm.tqdm(dino_dataloader)
        for dino_image, label in pbar:
            dino_image = dino_image.to(device)
            label = label.to(device)

            optimizer.zero_grad()
            pred = model(dino_image)
            loss = criterion(pred.view(-1, pred.size(-1)), label.view(-1))
            loss.backward()
            optimizer.step()
            scheduler.step()

            epoch_correct += torch.eq(pred.argmax(dim=-1), label).sum().item()
            epoch_loss += loss.item()

            pbar.set_description('Train Loss = {:f}'.format(loss.item()))

            '''
            if label == 1:
                predicted_class = torch.argmax(output, dim=1)
                print(f"Predicted class: {predicted_class.item()},  Actual class: {label.item()}, output: {output}")
            '''

        avg_loss = epoch_loss / len(dino_dataloader)
        loss_csv.append(avg_loss)
        print(f"Epoch {epoch}, Loss: {avg_loss:.6f}")

        if not (epoch + 1) % 4:
            model_path = f'{SAVE_PATH}/epoch_{epoch}_loss_{avg_loss}.pt'
            torch.save(model.state_dict(), model_path)

    print()
    data = {
        r"train_loss":loss_csv
    }
    df = pd.DataFrame(data)
    df.to_csv(f"{SAVE_PATH}/training_loss.csv", index_label="epoch")
    min_value = min(loss_csv)
    print("min label = ", loss_csv.index(min_value))
