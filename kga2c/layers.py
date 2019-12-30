import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

import networkx as nx
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)

    def forward(self, input):#, hidden):
        embedded = self.embedding(input)#.unsqueeze(0)
        hidden = self.initHidden(input.size(0))
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self, batch_size):
        return torch.zeros(1, batch_size, self.hidden_size).cuda()


class PackedEncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(PackedEncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden=None):
        embedded = self.embedding(input).permute(1,0,2) # T x Batch x EmbDim
        if hidden is None:
            hidden = self.initHidden(input.size(0))

        # Pack the padded batch of sequences
        lengths = torch.tensor([torch.nonzero(n)[-1] + 1 for n in input], dtype=torch.long).cuda()#, device=device)

        packed = nn.utils.rnn.pack_padded_sequence(embedded, lengths, enforce_sorted=False)
        output, hidden = self.gru(packed, hidden)
        # Unpack the padded sequence
        output, _ = nn.utils.rnn.pad_packed_sequence(output)
        # Return only the last timestep of output for each sequence
        idx = (lengths-1).view(-1,1).expand(len(lengths), output.size(2)).unsqueeze(0)
        output = output.gather(0, idx).squeeze(0)
        return output, hidden

    def initHidden(self, batch_size):
        return torch.zeros(1, batch_size, self.hidden_size).cuda()


class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(output_size, hidden_size)
        #self.combine = nn.Linear(hidden_size*2, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)#, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input, hidden):
        input = input.unsqueeze_(0).expand(100, -1, -1)#.transpose(0, 1).contiguous()
        output, hidden = self.gru(input, hidden)
        output = self.out(output[0])
        return output, hidden

    def initHidden(self, batch):
        return torch.zeros(1, batch, self.hidden_size).cuda()#).cuda()


class UnregDropout(nn.Module):
    def __init__(self, p: float = 0.5):
        super(UnregDropout, self).__init__()
        if p < 0 or p > 1:
            raise ValueError("dropout probability has to be between 0 and 1, " "but got {}".format(p))
        self.p = p
        self.training = True
        #print(self.p)

    def forward(self, X):
        if self.training:
            binomial = torch.distributions.binomial.Binomial(probs=1-self.p)
            sample = binomial.sample(X.size()).cuda()
            return X * sample
        return X


class DecoderRNN2(nn.Module):
    def __init__(self, hidden_size, output_size, embeddings, graph_dropout):
        super(DecoderRNN2, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = embeddings#nn.Embedding(output_size, hidden_size)
        self.combine = nn.Linear(hidden_size, hidden_size)#int(hidden_size / 2))
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.graph_dropout = UnregDropout(p=graph_dropout)#, training=False)#, inplace=True)
        self.graph_dropout_perc = graph_dropout

    def forward(self, input, hidden, encoder_output, graph_mask=None):
        output = self.embedding(input).unsqueeze(0)
        encoder_output = encoder_output.unsqueeze(0)
        output = torch.cat((output, encoder_output), dim=-1)
        output = self.combine(output)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.out(output[0])
        ret_output = output.clone().detach()

        if self.graph_dropout_perc != 1:
            with torch.no_grad():
                norm = ret_output.norm(p=2, dim=1, keepdim=True)
                ret_output = ret_output.div(norm)
                # graph_mask = self.graph_dropout(((~graph_mask).float()))#, p=0.5, training=False)
                # graph_mask = ~(graph_mask.bool())
                ret_output[~graph_mask] = float('-inf')

        return output, ret_output, hidden

    def initHidden(self, device):
        return torch.zeros(1, 1, self.hidden_size).cuda()


class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, max_length, dropout_p=0.2):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self, device):
        return torch.zeros(1, 1, self.hidden_size)#).cuda()


class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=False):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(nn.init.xavier_uniform_(torch.Tensor(in_features, out_features).type(torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor), gain=np.sqrt(2.0)), requires_grad=True)
        self.a = nn.Parameter(nn.init.xavier_uniform_(torch.Tensor(2*out_features, 1).type(torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor), gain=np.sqrt(2.0)), requires_grad=True)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, input, adj):
        h = torch.mm(input, self.W)
        N = h.size()[0]
        a_input = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=1).view(N, -1, 2 * self.out_features)

        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))

        zero_vec = torch.zeros_like(e)
        zero_vec = zero_vec.fill_(9e-15)
        attention = torch.where(adj > 0, e, zero_vec)

        attention = F.softmax(attention, dim=1)

        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, h)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class EncoderLSTM(nn.Module):

    def __init__(self, vocab_size, embedding_size, hidden_size, padding_idx,
                            dropout_ratio, embeddings, bidirectional=False, num_layers=1):
        super(EncoderLSTM, self).__init__()
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.drop = nn.Dropout(p=dropout_ratio)
        self.num_directions = 2 if bidirectional else 1
        self.num_layers = num_layers
        self.embedding = embeddings#nn.Embedding(vocab_size, embedding_size, padding_idx)
        self.gru = nn.LSTM(embedding_size, hidden_size, self.num_layers,
                          batch_first=True, dropout=dropout_ratio,
                          bidirectional=bidirectional)
        self.encoder2decoder = nn.Linear(hidden_size * self.num_directions,
            hidden_size * self.num_directions
        )

    def init_state(self, inputs):
        batch_size = inputs.size(0)
        h0 = Variable(torch.zeros(
            self.num_layers * self.num_directions,
            batch_size,
            self.hidden_size
        ), requires_grad=False)
        c0 = Variable(torch.zeros(
            self.num_layers * self.num_directions,
            batch_size,
            self.hidden_size
        ), requires_grad=False)
        return h0.cuda(), c0.cuda()

    def forward(self, inputs, lengths=0):
        embeds = self.embedding(inputs)   # (batch, seq_len, embedding_size)
        embeds = self.drop(embeds)
        h0, c0 = self.init_state(inputs)

        enc_h, (enc_h_t, enc_c_t) = self.gru(embeds, (h0, c0))

        if self.num_directions == 2:
            h_t = torch.cat((enc_h_t[-1], enc_h_t[-2]), 1)
            c_t = torch.cat((enc_c_t[-1], enc_c_t[-2]), 1)
        else:
            h_t = enc_h_t[-1]
            c_t = enc_c_t[-1] # (batch, hidden_size)

        decoder_init = nn.Tanh()(self.encoder2decoder(h_t))

        ctx = self.drop(enc_h)

        #print("lstm", ctx.size(), c_t.size())

        return ctx,decoder_init,c_t  # (batch, seq_len, hidden_size*num_directions)
                                 # (batch, hidden_size)
