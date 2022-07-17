import argparse
from model import Net
import torch
import torch.backends.cudnn as cudnn
from torch import nn
from tqdm import tqdm
from datasets import EvalDataset, TrainDataset
import copy
from preprocess import AverageMeter
from metrics import calc_psnr

CUDA = torch.cuda.is_available()
device = torch.device('cuda' if CUDA else 'cpu')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-file', type=str, required=True)
    parser.add_argument('--eval-file', type=str, required=True)
    parser.add_argument('--outputs-dir', type=str)
    parser.add_argument('--scale', type=int, default=3)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batch-size', type=int, default=5)
    parser.add_argument('--num-epochs', type=int, default=500)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--seed', type=int, default=69)
    args = parser.parse_args()

    cudnn.benchmark = True

    torch.manual_seed(args.seed)
    lr = args.lr
    nEpochs = args.num_epochs
    batch_size = args.batch_size
    criterion = nn.MSELoss()
    model = Net(num_channels=1, base_filter=64).to(device)
    optimizer = torch.optim.Adam(model.parameters())
    train_dataset = TrainDataset(args.train_file)
    training_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=args.num_workers)
    eval_dataset = EvalDataset(args.eval_file)
    eval_loader = torch.utils.data.DataLoader(dataset=eval_dataset, batch_size=1)

    best_psnr = 0.0
    best_epoch = 0
    best_weights = copy.deepcopy(model.state_dict())

    def save_model():
        model_out_path = args.outputs_dir + '/model_x'+str(args.scale)+'.pth'
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

    model.eval()
    epoch_psnr = AverageMeter()

    for data in eval_loader:
        inputs, labels = data

        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            preds = model(inputs).clamp(0.0, 1.0)

        epoch_psnr.update(calc_psnr(preds, labels), len(inputs))

    print('eval psnr: {:.2f}'.format(epoch_psnr.avg))

    if epoch_psnr.avg > best_psnr:
        best_epoch = epoch
        best_psnr = epoch_psnr.avg
        best_weights = copy.deepcopy(model.state_dict())

    print('best epoch: {}, psnr: {:.2f}'.format(best_epoch, best_psnr))
