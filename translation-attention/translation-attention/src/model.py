import torch
from torch import nn
import config


class Attention(nn.Module):
    def forward(self, decoder_hidden, encoder_outputs):
        # decoder_hidden.shape: [1, batch_size, decoder_hidden_size]
        # encoder_outputs.shape: [batch_size, seq_len, decoder_hidden_size]

        attention_score = torch.bmm(decoder_hidden.transpose(0, 1), encoder_outputs.transpose(1, 2))  #decoder_hidden(Q)  K(encoder_outputs)
        # attention_score.shape: [batch_size, 1, seq_len]
        attention_weight = torch.softmax(attention_score, dim=-1)
        context_vector = torch.bmm(attention_weight, encoder_outputs)  # V（encoder_outputs）
        # context_vector.shape: [batch_size, 1, decoder_hidden_size]
        return context_vector


class TranslationEncoder(nn.Module):
    def __init__(self, vocab_size, padding_index):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size,
                                      embedding_dim=config.EMBEDDING_DIM,
                                      padding_idx=padding_index)

        self.gru = nn.GRU(input_size=config.EMBEDDING_DIM,
                          hidden_size=config.ENCODER_HIDDEN_SIZE,
                          batch_first=True,
                          num_layers=config.ENCODER_LAYERS,
                          bidirectional=True)

    def forward(self, x):
        # x.shape: [batch_size, seq_len]
        embed = self.embedding(x)
        # embed.shape: [batch_size, seq_len, embedding_dim]

        output, hidden = self.gru(embed)
        # output.shape: [batch_size, seq_len, hidden_size * 2]
        # hidden.shape: [num_layers * num_directions, batch_size, hidden_size]
        last_hidden_forward = hidden[-2]
        last_hidden_backward = hidden[-1]
        context_vector = torch.cat([last_hidden_forward, last_hidden_backward], dim=-1)
        # last_hidden.shape: [batch_size, hidden_size * 2]
        return output, context_vector


class TranslationDecoder(nn.Module):
    def __init__(self, vocab_size, padding_index):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size,
                                      embedding_dim=config.EMBEDDING_DIM,
                                      padding_idx=padding_index)
        self.gru = nn.GRU(input_size=config.EMBEDDING_DIM,
                          hidden_size=config.DECODER_HIDDEN_SIZE,
                          batch_first=True)

        self.attention = Attention()

        self.linear = nn.Linear(in_features=2 * config.DECODER_HIDDEN_SIZE,
                                out_features=vocab_size)

    def forward(self, x, hidden_0, encoder_outputs):
        # x.shape: [batch_size, 1]
        embed = self.embedding(x)
        # embed.shape: [batch_size, 1, embedding_dim]

        output, hidden_n = self.gru(embed, hidden_0)
        # output.shape: [batch_size, 1, hidden_size]
        # hidden_n.shape: [1, batch_size, hidden_size]

        # 注意力机制
        # encoder_outputs和 hidden_n 进行计算，得到一个上下文向量
        context_vector = self.attention(hidden_n, encoder_outputs)
        # context_vector.shape: [batch_size, 1, hidden_size]

        # 拼接上下文向量和输出向量
        combined = torch.cat((output, context_vector), dim=2)
        # combined.shape: [batch_size, 1, hidden_size*2]

        output = self.linear(combined)
        # output.shape: [batch_size, 1, vocab_size]
        return output, hidden_n
