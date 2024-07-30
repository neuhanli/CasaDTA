
import sys, os
from random import shuffle
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.models as models

from create_data import loadTest, loadTrain, Data, collate_fn
from myDataset import MyDataset
from functionCi import *
from model import CasaNet
# training function at each epoch

def train(model, device, train_loader, optimizer, epoch):
    print('Training on {} samples...'.format(len(train_loader.dataset)))
    device = torch.device("cuda:0")
    model.train()
    loss = 0
    for batch_idx, data in enumerate(train_loader):
        x_smiles = data[0].to(device)

        y = data[1].to(device)  # 将元组的第二个元素转换为张量并移动到指定设备
        target_Embedding = data[2].to(device)


        optimizer.zero_grad()
        model.to(device)
        output,_= model(x_smiles, target_Embedding)

        loss = loss_fn(output, y.view(-1, 1).float().to(device))
        loss.backward()
        optimizer.step()
        torch.cuda.empty_cache()
        if batch_idx % LOG_INTERVAL == 0:
            print('Train epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch,
                                                                           batch_idx * len(x_smiles),
                                                                           len(train_loader.dataset),
                                                                           100. * batch_idx / len(train_loader),
                                                                           loss.item()))
    return loss.item()


def predicting(model, device, loader):
    model.eval()
    loss = 0
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    print('Make prediction for {} samples...'.format(len(loader.dataset)))
    with torch.no_grad():
        for data in loader:
            x_smiles = data[0].to(device)


            y = data[1].to(device)  # 将元组的第二个元素转换为张量并移动到指定设备
            target_Embedding = data[2].to(device)


            model.to(device)
            output,_ = model(x_smiles, target_Embedding)
            total_preds = torch.cat((total_preds, output.cpu()), 0)

            total_labels = torch.cat((total_labels, y.view(-1, 1).cpu()), 0)
            loss = loss_fn(output, y.view(-1, 1).float().to(device))
        print('Test epoch: Loss: {:.6f}'.format(loss.item()))
    return total_labels.numpy().flatten(),total_preds.numpy().flatten(),loss.item()





torch.backends.cudnn.enabled = False

datasets = [['davis','kiba'][int(sys.argv[1])]]

cuda_name = "cuda:0"
if len(sys.argv)>2:
    cuda_name = "cuda:" + str(int(sys.argv[2]))
print('cuda_name:', cuda_name)
print(torch.__version__)
print(torch.cuda.is_available())

TRAIN_BATCH_SIZE =256
TEST_BATCH_SIZE = 256
LR = 0.0001
LOG_INTERVAL = 20
NUM_EPOCHS = 1000

print('Learning rate: ', LR)
print('Epochs: ', NUM_EPOCHS)

# Main program: iterate over different datasets
for dataset in datasets:
    train_data = loadTrain(dataset)
    test_data = loadTest(dataset)

    # make data PyTorch mini-batch processing ready
    train_loader = DataLoader(train_data, batch_size=TRAIN_BATCH_SIZE, shuffle=True,collate_fn=collate_fn)
    test_loader = DataLoader(test_data, batch_size=TEST_BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

    checkpoint_path = 'checkpoint_davis.pt'
    startEpoch = 0

    # training the model
    device = torch.device(cuda_name)
    model = CasaNet().to(device)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        startEpoch = checkpoint['epoch']

    best_mse = 1000
    best_ci = 0
    best_rm2 = 0
    best_epoch = -1
    model_file_name = 'model_' + '_' + dataset +  '.pth'
    result_file_name = 'result_' + '_' + dataset +  '.csv'
    for epoch in range(startEpoch,NUM_EPOCHS):
        t_loss = train(model, device, train_loader, optimizer, epoch+1)
        G,P,p_loss = predicting(model, device, test_loader)
        ret = [rmse(G,P),mse(G,P),pearson(G,P),spearman(G,P),ci(G,P),get_rm2(G,P),t_loss,p_loss]
        torch.save({
            'epoch': epoch+1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        },checkpoint_path)
        # ret = [rmse(G,P),mse(G,P),pearson(G,P),spearman(G,P),ci(G,P),t_loss,p_loss]
        if ret[1]<best_mse:
            torch.save(model.state_dict(), model_file_name)
            with open(result_file_name,'w') as f:
                f.write(','.join(map(str,ret)))
            best_epoch = epoch+1
            best_mse = ret[1]
            best_ci = ret[4]
            best_rm2 = ret[5]

            print('ci improved at epoch ', best_epoch, '; best_mse,best_rm2,best_ci:', best_mse,best_rm2,best_ci,dataset)
        else:
            print(ret[1],'No improvement since epoch ', best_epoch, '; best_mse,best_rm2,best_ci:', best_mse,best_rm2,best_ci,dataset)

