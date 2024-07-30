import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (self.head_dim * heads == embed_size), "Embed size needs to be div by heads"

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, values, keys, queries, mask):
        N = queries.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], queries.shape[1]

        # Split embedding into self.heads pieces
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = queries.reshape(N, query_len, self.heads, self.head_dim)

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)

        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])

        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)

        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(N, query_len, self.heads * self.head_dim)

        out = self.fc_out(out)
        return out, attention

 # 位置编码
def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2*(i // 2))/ np.float32(d_model))
    return pos * angle_rates
def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                           np.arange(d_model)[np.newaxis,:],
                           d_model)
    sines = np.sin(angle_rads[:, 0::2])
    cones = np.cos(angle_rads[:, 1::2])
    pos_encoding = np.concatenate([sines, cones], axis=-1)
    pos_encoding = pos_encoding[np.newaxis, ...]
    return torch.tensor(pos_encoding, dtype=torch.float32)

class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask):
        out, attention = self.attention(value, key, query, mask)

        x = self.dropout(self.norm1(out + query))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out, attention


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
        # self.word_embedding = nn.Embedding(src_vocab_size, embed_size)
        # self.position_embedding = nn.Embedding(max_length, embed_size)
        self.position_embedding = positional_encoding(max_length, 128)
        self.position_embedding_drug = positional_encoding(100, 128)

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
        self.fc_out = nn.Linear(embed_size, 128)

    def forward(self, x_drug_v, x_drug_k, x_protein_q, mask):
        device = torch.device("cuda:0")
        N, seq_length_drug, _= x_drug_v.shape
        M, seq_length_protein, _ = x_protein_q.shape

        padded_tensor_drug = self.position_embedding_drug[:, :seq_length_drug, :]
        padded_tensor_protein = self.position_embedding[:, :seq_length_protein, :]

        padded_tensor_drug = padded_tensor_drug.to(device)
        padded_tensor_protein = padded_tensor_protein.to(device)
        # positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)

        # out_v = self.dropout(x_drug_v + self.position_embedding(positions))
        # out_k = self.dropout(x_protein_k + self.position_embedding(positions))
        # out_q = self.dropout(x_protein_q+ self.position_embedding(positions))

        out_v = self.dropout(x_drug_v + padded_tensor_drug)
        out_k = self.dropout(x_drug_k + padded_tensor_drug)
        out_q = self.dropout(x_protein_q + padded_tensor_protein)
        out = []
        attention = []
        for layer in self.layers:
            out, attention = layer(out_v, out_k, out_q, mask)
        # out = self.fc_out(out)
        return out, attention


class DecoderBlock(nn.Module):
    def __init__(self, embed_size, heads, forward_expansion, dropout, device):
        super(DecoderBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.norm = nn.LayerNorm(embed_size)
        self.transformer_block = TransformerBlock(
            embed_size, heads, dropout, forward_expansion
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, value, key, src_mask, trg_mask):
        attention = self.attention(x, x, x, trg_mask)
        query = self.dropout(self.norm(attention + x))
        out = self.transformer_block(value, key, query, src_mask)
        return out


class Decoder(nn.Module):
    def __init__(
            self,
            trg_vocab_size,
            embed_size,
            num_layers,
            heads,
            forward_expansion,
            dropout,
            device,
            max_length,

    ):
        super(Decoder, self).__init__()
        self.device = device
        self.word_embedding = nn.Embedding(trg_vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)

        self.layers = nn.ModuleList(
            [
                DecoderBlock(embed_size, heads, forward_expansion, dropout, device)
                for _ in range(num_layers)
            ]
        )

        self.fc_out = nn.Linear(embed_size, trg_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out, src_mask, trg_mask):
        N, seq_length = x.shape
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)
        x = self.dropout(self.word_embedding(x) + self.position_embedding(positions))

        for layer in self.layers:
            x = layer(x, enc_out, enc_out, src_mask, trg_mask)

        out = self.fc_out(x)
        return out


class Transformer(nn.Module):
    def __init__(
            self,
            src_vocab_size,
            trg_vocab_size,
            src_pad_idx=0,
            trg_pad_idx=0,
            embed_size=128,
            num_layers=3,
            forward_expansion=4,
            heads=1,
            dropout=0.1,
            device="cuda",
            max_length=1000,
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

        self.decoder = Decoder(
            trg_vocab_size,
            embed_size,
            num_layers,
            heads,
            forward_expansion,
            dropout,
            device,
            max_length
        )

        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device

    def make_src_mask(self, src):
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask.to(self.device)

    def make_trg_mask(self, trg):
        N, trg_len = trg.shape
        trg_mask = torch.tril(torch.ones((trg_len, trg_len))).expand(
            N, 1, trg_len, trg_len
        )
        return trg_mask.to(self.device)

    def forward(self, x_v, x_k, x_q):
        src_mask = self.make_src_mask(x_q)
        out, attention = self.encoder(x_v, x_k, x_q, src_mask)
        return out, attention


def get_EF(input_size, dim, method="learnable", head_dim=None, bias=True):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    """
    Retuns the E or F matrix, initialized via xavier initialization.
    This is the recommended way to do it according to the authors of the paper.
    Includes a method for convolution, as well as a method for no additional params.
    """
    assert method == "learnable" or method == "convolution" or method == "no_params", "The method flag needs to be either 'learnable', 'convolution', or 'no_params'!"
    if method == "convolution":
        conv = nn.Conv1d(head_dim, head_dim, kernel_size=int(input_size / dim), stride=int(input_size / dim))
        return conv
    if method == "no_params":
        mat = torch.zeros((input_size, dim))
        torch.nn.init.normal_(mat, mean=0.0, std=1 / dim)
        return mat
    lin = nn.Linear(input_size, dim, bias).to(device)
    torch.nn.init.xavier_normal_(lin.weight).to(device)
    return lin


class SMILES_FASTAModel_xtd(nn.Module):
    def __init__(self, char_set_len_xtd, char_set_len_protein):
        super().__init__()
        self.embedding= nn.Embedding(90, 128)
        # self.embed_drug = nn.Embedding(char_set_len_xtd, 128)

        # self.embed_protein = nn.Embedding(char_set_len_protein, 128)
        self.transform = Transformer(char_set_len_xtd, char_set_len_protein)

    def forward(self, xd, protein):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # xt = self.embed_protein(protein)
        # xtd = torch.cat((xt, xd), 1)
        # E = get_EF(1100, 256)
        # fc_layer = nn.Linear(1280, 128).to(device)
        # fasta_reshaped = xtd.view(xtd.size(0), xtd.size(1), -1).to(device)
        # fasta_flat = fasta_reshaped.view(-1, fasta_reshaped.size(-1)).to(device)
        # fasta_reduced = fc_layer(fasta_flat).to(device)
        # fasta = fasta_reduced.view(xtd.size(0), xtd.size(1), -1).to(device)
        # query,k,v




        out, attention = self.transform(xd, xd, protein)

        # [256,100,128]
        # v, _  = torch.max(out, -1)
        return out,attention