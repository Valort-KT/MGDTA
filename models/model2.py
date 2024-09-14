import torch
from torch import nn
import torch.nn.functional as F
from mamba_ssm.models.mixer_seq_simple import MixerModel

class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert(self.head_dim * heads == embed_size), "Embed size needs to be div by heads"

        self.values = nn.Linear(self.head_dim, self.head_dim, bias = False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias = False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias = False)
        self.fc_out = nn.Linear(heads*self.head_dim, embed_size)

    def forward(self, values, keys, queries, mask):
        N = queries.shape[0]

        value_len, key_len, query_len = values.shape[1], keys.shape[1], queries.shape[1]

        #Split embedding into self.heads pieces
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = queries.reshape(N, query_len, self.heads, self.head_dim)

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)

        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])

        attention = torch.softmax(energy/ (self.embed_size ** (1/2)), dim=3)

        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(N, query_len, self.heads*self.head_dim)
        
        out = self.fc_out(out)
        return out

class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion*embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion*embed_size, embed_size)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask):
        attention = self.attention(value, key, query, mask)

        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward+x))
        return out

class Encoder(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        embed_size,
        num_layers,
        heads,
        device,
        forward_expansion,
        dropout,
        max_length,
        trg_vocab_size,
    ):
        super(Encoder, self).__init__()
        self.embed_size = embed_size
        self.device = device
        self.word_embedding = nn.Embedding(src_vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)

        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    embed_size,
                    heads,
                    dropout=dropout,
                    forward_expansion=forward_expansion
                )
            for _ in range(num_layers)]
        )
        self.dropout = nn.Dropout(dropout)
        self.fc_out = nn.Linear(embed_size, trg_vocab_size)
    
    def forward(self, x, mask):
        N, seq_length = x.shape
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)

        out = self.dropout(self.word_embedding(x) + self.position_embedding(positions))

        for layer in self.layers:
            out = layer(out, out, out, mask)
            
        out = self.fc_out(out)
        return out
  
class Transformer(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        trg_vocab_size,
        src_pad_idx=0,
        trg_pad_idx=0,
        embed_size = 256,
        num_layers = 1,
        forward_expansion = 2,
        heads = 8,
        dropout = 0.1,
        device = "cuda",
        max_length = 166,
    ):
        super(Transformer, self).__init__()
    
        self.encoder = Encoder(
            src_vocab_size,
            embed_size,
            num_layers,
            heads,
            device,
            forward_expansion,
            dropout,
            max_length,
            trg_vocab_size
        )



        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device

    def make_src_mask(self, src):
            src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
            return src_mask.to(self.device)

    def make_trg_mask(self, trg):
            N, trg_len = trg.shape
            trg_mask = torch.tril(torch.ones((trg_len,trg_len))).expand(
                N, 1, trg_len, trg_len
            )
            return trg_mask.to(self.device)
        
    def forward(self, src):
            src_mask = self.make_src_mask(src)
            out = self.encoder(src, src_mask)
            return out


class Cross_Transformer(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(Cross_Transformer, self).__init__()
        
        self.attention = SelfAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion*embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion*embed_size, embed_size)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask):
        attention = self.attention(value, key, query, mask)
        attention = F.tanh(attention)
        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        # out = self.dropout(self.norm2(forward+x))
        out = forward+x
        return out
    
        


class SMILESModel(nn.Module):
    def __init__(self, char_set_len):
        super().__init__()
        self.transform = Transformer(char_set_len, char_set_len)

    def forward(self, smiles):
        v = self.transform(smiles)
        v, _  = torch.max(v, -1)
        return v

class FASTAModel2(nn.Module):
    def __init__(self, char_set_len):
        super().__init__()
        self.mam = MixerModel(d_model=166,n_layer=3,vocab_size=char_set_len,rms_norm=True)

    
    def forward(self, fasta):
        # keras is channal last, different with pytorch
        v = self.mam(fasta)
        ve, inid  = torch.max(v, dim=1)
        return ve,v

class Classifier(nn.Sequential):

    def __init__(self, smiles_model, fasta_model,mol_model,prot_model=None,device=None):

        super().__init__()
        self.smiles_model = smiles_model
        self.fasta_model = fasta_model
        self.mol_model = mol_model

        self.dcross_trans = Cross_Transformer(166, 1, 0.1, 2)

        self.fc1 = nn.Linear(332,1024)
        self.dropout1 = nn.Dropout(0.1)
        self.fc2 = nn.Linear(1024,1024)
        self.dropout2 = nn.Dropout(0.1)
        self.fc3 = nn.Linear(1024,512)
        self.fc4 = nn.Linear(512,1)
        self.fc4.weight.data.normal_()
        self.fc4.bias.data.normal_()

    def forward(self, smiles, fasta,cg):
        v_smiles = self.smiles_model(smiles)
        v_fasta,ve = self.fasta_model(fasta)
        v_mol = self.mol_model(cg)

        v_smiles = torch.unsqueeze(v_smiles,dim=1)
        v_mol = torch.unsqueeze(v_mol,dim=1)

        v_ms = self.dcross_trans(v_mol,v_mol,v_smiles,mask = None)
        v_ms = torch.squeeze(v_ms,dim=1)

        v = torch.cat((v_ms,v_fasta),-1)
        v = F.leaky_relu(self.fc1(v))
        v = self.dropout1(v)
        v = F.leaky_relu(self.fc2(v))
        v = self.dropout2(v)
        v = self.fc3(v)
        v = self.fc4(v)
        return v,ve
