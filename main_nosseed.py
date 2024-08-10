import torch
import logging
import pandas as pd 
# from torch.utils.data import DataLoader, random_split
from torch.utils.data import random_split
from joblib import Parallel, delayed

import time

from torch.utils.tensorboard import SummaryWriter
from dgl import load_graphs
import torch_geometric
from torch_geometric.loader import DataLoader
from logger import create_logger


# from model import Classifier, SMILESModel, FASTAModel
from models.model2 import Classifier, SMILESModel, FASTAModel2
from models.model2_g import Prot3DGraphModel,Drug3DModel
from process_data import DTAData, CHARISOSMILEN, CHARPROTLEN, MACCSLEN
from train_and_test import train
import os,sys
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
code_path = os.path.dirname(os.path.realpath(__file__))
print('code_path:', code_path)
result_path = code_path + '/result/'
if not os.path.exists(result_path):
    os.makedirs(result_path)


MODEL_NAME = "FEDTA"
BATCH_SIZE = 256
DATASET = "Bind"                                  

# logging.basicConfig(filename=f'{result_path}/{MODEL_NAME}.log', level=logging.DEBUG)
writer = SummaryWriter()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    if DATASET == 'davis':
        data_path = f"/data/hkt/work_DAT/work_graphgps/mam_dtitr/data/dtdav/4"
        train_data = pd.read_csv(f"{data_path}/train.csv")
        valid_data = pd.read_csv(f"{data_path}/valid.csv")
        test_data = valid_data
        max_smiles_len = 85
        max_fasta_len = 1000
    if DATASET == 'kiba':
        data_path = f"/data/hkt/work_DAT/work_graphgps/mam_dtitr_qs/data/dtkiba/4"
        train_data = pd.read_csv(f"{data_path}/train.csv")
        valid_data = pd.read_csv(f"{data_path}/valid.csv")
        test_data = valid_data
        max_smiles_len = 100
        max_fasta_len = 1000
    if DATASET == 'Bind':
        data_path = f"/data/hkt/work_DAT/work_graphgps/mam_dtitr_qs/data/binddb_kd/4"
        train_data = pd.read_csv(f"{data_path}/train.csv")
        valid_data = pd.read_csv(f"{data_path}/valid.csv")
        test_data = valid_data
        max_smiles_len = 100
        max_fasta_len = 1000
    output_dir = result_path
    date_now=time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    output_dir = os.path.join(output_dir,date_now)+f"{DATASET}_{BATCH_SIZE}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    global logger
    




    # fasta_train = list(df_train["PROTEIN_SEQUENCE"])[:500]
    # fasta_test = list(df_test["PROTEIN_SEQUENCE"])[:500]
    # smiles_train = list(df_train["COMPOUND_SMILES"])[:500]
    # smiles_test = list(df_test["COMPOUND_SMILES"])[:500]
    # label_train = list(df_train['REG_LABEL'])[:500]
    # label_test = list(df_test['REG_LABEL'])[:500]

    # train_valid_set = DTAData(smiles_train, fasta_train, label_train, compound_graph1,protein_graph1,device, max_smiles_len, max_fasta_len)
    # test_set = DTAData(smiles_test, fasta_test, label_test,compound_graph2,protein_graph2, device, max_smiles_len, max_fasta_len)


    date_now=time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    logger,out_dir = create_logger(output_dir=output_dir,date_now = date_now, name=f"{DATASET}_{BATCH_SIZE}")
    logger.info(f"{date_now}_{DATASET}_mam_qs_mam3")

    train_set = DTAData(train_data,device, max_smiles_len, max_fasta_len)
    valid_set = DTAData(valid_data,device, max_smiles_len, max_fasta_len)
    test_set = DTAData(test_data,device, max_smiles_len, max_fasta_len)
    # train_size = int(len(train_valid_set) * 0.75)
    # valid_size = len(train_valid_set) - train_size
    # train_set, valid_set = random_split(train_valid_set, [train_size, valid_size])
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,num_workers=8)
    valid_loader = DataLoader(valid_set, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

    smiles_model = SMILESModel(char_set_len = MACCSLEN)
    fasta_model = FASTAModel2(char_set_len=CHARPROTLEN+1)
    prot_model = Prot3DGraphModel(d_vocab=41, d_edge=5, hidden_dim=166,
                                    in_feat_dropout=0.0,  pos_enc_dim=8)
    mol_model = Drug3DModel(d_vocab=44, d_edge=10, hidden_dim=166,
                            in_feat_dropout=0.0,  pos_enc_dim=8)
    model = Classifier(smiles_model, fasta_model,mol_model,prot_model,device)
    model = model.to(device)
    # print(model)
    logger.info(model)
    models_path = f"{result_path}/tv1{DATASET}_{BATCH_SIZE}"
    if not os.path.exists(models_path):
        os.mkdir(models_path)

    train(model, train_loader, valid_loader, test_loader, writer, MODEL_NAME,models_path,device,logger)

    del model



def process_graph(compound_graph, protein_graph):
    # 处理化合物图
    c = compound_graph.ndata
    cx = torch.as_tensor(c["atom"], dtype=torch.float32, device="cpu")
    cpos = c['lap_pos_enc']
    ce = torch.as_tensor(compound_graph.edata['bond'], dtype=torch.float32, device="cpu")
    ces, cet = compound_graph.edges()

    csrc_nodes = ces.clone().detach()
    cdst_nodes = cet.clone().detach()
    # csrc_nodes = ces.detach()
    # cdst_nodes = cet.detach()
    cedge_index = torch.stack([csrc_nodes, cdst_nodes])
    compound_tg = torch_geometric.data.Data(x=cx, edge_index=cedge_index, edge_attr=ce, lap_pos=cpos)

    # 处理蛋白质图
    p = protein_graph.ndata
    px = torch.as_tensor(p["feats"], dtype=torch.float32, device="cpu")
    ppos = p['lap_pos_enc']
    pe = torch.as_tensor(protein_graph.edata['feats'], dtype=torch.float32, device="cpu")
    pes, pet = protein_graph.edges()

    psrc_nodes = pes.clone().detach()
    pdst_nodes = pet.clone().detach()
    # psrc_nodes = pes.detach()
    # pdst_nodes = pet.detach()
    pedge_index = torch.stack([psrc_nodes, pdst_nodes])
    protein_tg = torch_geometric.data.Data(x=px, edge_index=pedge_index, edge_attr=pe, lap_pos=ppos)

    return compound_tg, protein_tg


def process_graphs_parallel(compound_graphs, protein_graphs):
    # with Parallel(n_jobs=-1,verbose=1) as parallel:
        # 并行处理每个图
    results = Parallel(n_jobs=-1,verbose=1)(delayed(process_graph)(compound_graph, protein_graph) for compound_graph, protein_graph in zip(compound_graphs, protein_graphs))

    # 获取并行处理结果
    compound_graphs_parallel, protein_graphs_parallel = zip(*results)

    return compound_graphs_parallel, protein_graphs_parallel



if __name__ == "__main__":
    main()