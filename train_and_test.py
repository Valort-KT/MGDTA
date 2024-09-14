import torch
import copy
import logging
from scipy import stats
import pandas as pd
from utils.metrics import get_cindex, get_rm2
from tqdm import tqdm
import os

def test(data_loader, model, loss_fn,result_path,device,scatter = None,dataset_name = None,model_name=None,logger=None):
    with torch.no_grad():
        model.eval()
        y_true = []
        y_pred = []
        score_list = []
        label_list = []
        smiles_list = []
        fasta_list = []
        ve_list  = []

        running_loss = 0.0
        for sample in data_loader:

            smiles, fasta, label,cg,smi,pro = sample
            cg = cg.to(device)

            smiles = smiles.to(device)
            fasta = fasta.to(device)
            label = label.to(device)

            score,ve = model(smiles,fasta,cg)
            score = score.view(-1)


            loss = loss_fn(score, label)
            running_loss += loss.item()
            y_pred += score.detach().cpu().tolist()
            y_true += label.detach().cpu().tolist()
            ve_list.append(ve.detach().cpu())
            score_list.append(score)
            label_list.append(label)
            smiles_list.append(smiles)
            fasta_list.append(fasta)
        ci = get_cindex(y_true, y_pred)
        rm2 = get_rm2(y_true, y_pred)
        Spearman = stats.spearmanr(y_true,y_pred)[0]

        ve_tensor = torch.cat(tuple(ve_list),dim=0)

            
        with open( f"{result_path}/prediction.txt", 'a') as f:
            f.write("ci: " + str(ci) + " rm2: " + str(rm2) + "Spearman:" + str(Spearman)+  '\n')

    return running_loss/len(data_loader), ci, rm2,Spearman,ve_tensor


def train(model, train_loader, val_loader,test_loader, writer, NAME,result_path,device,logger=None, lr=0.0001, epoch=1000):

    opt = torch.optim.Adam(model.parameters(), lr = lr)
    loss_fn = torch.nn.MSELoss()
    model_best = copy.deepcopy(model)
    min_loss = 1000
    max_rm2 = 0
    max_ci = 0

    for epo in range(epoch):
        model.train()
        running_loss = 0.0
        

        for data in tqdm(train_loader):

            smiles, fasta, label,cg = data
            cg = cg.to(device)

            smiles = smiles.to(device)
            fasta = fasta.to(device)
            label = label.to(device)
            
            score = model(smiles,fasta,cg).view(-1)

            loss = loss_fn(score, label)
            
            opt.zero_grad()
            loss.backward()
            opt.step()

            running_loss += loss.item()

        logger.info(f'Training at Epoch {epo + 1} with loss {running_loss/len(train_loader):.4f}')
        
        val_loss, val_ci, val_rm2 = test(val_loader, model, loss_fn,result_path,device)

        logger.info(f'Validation at Epoch {epo+1} with loss {val_loss:.4f}, ci {val_ci}, rm2 {val_rm2}')
        if val_loss < min_loss:
            min_loss = val_loss
            loss_best = copy.deepcopy(model)
            loss_model = {
                "epoch":epo,
                "loss":val_loss,
                "ci":val_ci,
                "rm2":val_rm2,
                "state":loss_best.state_dict(),
            }
        if max_rm2 < val_rm2:
            max_rm2 = val_rm2
            rm2_best = copy.deepcopy(model)
            rm2_model = {
                "epoch":epo,
                "loss":val_loss,
                "ci":val_ci,
                "rm2":val_rm2,
                "state":rm2_best.state_dict(),
            }
        if max_ci < val_ci:
            max_ci = val_ci
            ci_best = copy.deepcopy(model)
            ci_model = {
                "epoch":epo,
                "loss":val_loss,
                "ci":val_ci,
                "rm2":val_rm2,
                "state":ci_best.state_dict(),
            }
        

    test_loss, test_ci, test_rm2 = test(test_loader, loss_best, loss_fn,result_path,device)
    logger.info(f'Loss_best   model param: Test loss {test_loss:.4f}, ci {test_ci}, rm2 {test_rm2}')
    test_loss, test_ci, test_rm2 = test(test_loader, rm2_best, loss_fn,result_path,device)
    logger.info(f'Loss_rm2   model param: Test loss {test_loss:.4f}, ci {test_ci}, rm2 {test_rm2}')
    test_loss, test_ci, test_rm2 = test(test_loader, ci_best, loss_fn,result_path,device)
    logger.info(f'Loss_ci   model param: Test loss {test_loss:.4f}, ci {test_ci}, rm2 {test_rm2}')
    torch.save(loss_model, f'{result_path}/loss_{NAME}.pth')
    torch.save(rm2_model, f'{result_path}/rm2_{NAME}.pth')
    torch.save(ci_model, f'{result_path}/ci_{NAME}.pth')
