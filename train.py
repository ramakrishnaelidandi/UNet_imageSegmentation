import torch 
from model import UNET
from dataset import Carvana_dataset
from tqdm import tqdm 
from torch.cuda.amp import autocast, grad_scaler

device = 'cuda' if torch.cuda.is_available() else 'cpu'
Learning_rate = 

def train(dataloader, model, optimizer, loss_fun):
    scaler  = grad_scaler()
    loop = tqdm(dataloader)

    for idx, (data,target) in enumerate(loop):
        data = data.to(device= device)
        target = target.tofloat().unsqueeze(1).to(device=device)

        #forward
        with autocast():
            logits = model(data)
            loss = loss_fun(logits, target)

        #backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        #update tqdm
        loop.set_postfix(loss = loss.item())


def main():
    
    model =  UNET(in_channels=3,out_channels= 1).to(device)
    optimizer = torch.optim.Adam(model.parameters(),lr = Learning_rate)


