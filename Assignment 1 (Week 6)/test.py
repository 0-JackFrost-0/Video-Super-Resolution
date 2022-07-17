# Use this to test your model. 
# Use appropriate command line arguments and conditions

import argparse
import torch
from metrics import calc_psnr
from model import Net
from tqdm import tqdm
from datasets import EvalDataset
import matplotlib.pyplot as plt

CUDA = torch.cuda.is_available()
device = torch.device('cuda' if CUDA else 'cpu')

batch_size = 4
test_dataset = EvalDataset('./testdata/Set5_x2.h5')
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
avg_psnr = 0

model = torch.load('./model_path.pth')
model.eval()

with torch.no_grad():
    # with tqdm(total=len(test_loader)) as p_bar:
    #     for batch_num, (data, target) in enumerate(test_loader):
    #         pass
    data, target = torch.from_numpy(test_dataset.__getitem__(1)[0]), torch.from_numpy(test_dataset.__getitem__(1)[1])
    print(data.shape, target.shape)
    prediction = model(data)
    # print('PSNR of LR with HR:', calc_psnr(data, target))
    print('PSNR is coming out to be: ', calc_psnr(target, prediction))
    avg_psnr += calc_psnr(prediction, target)
    # p_bar.update(1)

    fig = plt.figure()
    fig.add_subplot(2, 2, 1)
    plt.imshow(torch.squeeze(data))
    plt.title('LR')
    plt.axis('off')
    fig.add_subplot(2, 2, 2)
    plt.imshow(torch.squeeze(target))
    plt.title('HR')
    plt.axis('off')
    fig.add_subplot(2, 2, 3)
    plt.imshow(torch.squeeze(prediction))
    plt.title('Predicted')
    plt.axis('off')

    plt.show()
    print("PSNR: {:.4f} dB".format(avg_psnr))

