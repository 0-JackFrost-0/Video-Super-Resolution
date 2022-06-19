# Train the model here

from model import Net
import torch
from torch import nn
from tqdm import tqdm
import datasets

CUDA = torch.cuda.is_available()
device = torch.device('cuda' if CUDA else 'cpu')


data = './HDF5 Format/91-image_x2.h5'
lr = 0.01
nEpochs = 3
batch_size = 5
criterion = nn.MSELoss()
model = Net(num_channels=1, base_filter=64).to(device)
optimizer = torch.optim.Adam(model.parameters())
train_dataset = datasets.TrainDataset(data)
training_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


def save_model():
    model_out_path = "model_path.pth"
    torch.save(model, model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))


def train():
    train_loss = 0
    with tqdm(total=len(training_loader)) as p_bar:
        for batch_num, (data, target) in enumerate(training_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            loss = criterion(model(data), target)
            train_loss += loss.item()
            loss.backward()
            optimizer.step()
            p_bar.update(1)
        # progress_bar(batch_num, len(training_loader), 'Loss: %.4f' % (train_loss / (batch_num + 1)))

    print("    Average Loss: {:.4f}".format(train_loss / len(training_loader)))


for epoch in range(1, nEpochs + 1):
    print("\n===> Epoch {} starts:".format(epoch))
    train()
    # scheduler.step(epoch)
    if epoch == nEpochs:
        save_model()
