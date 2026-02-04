import time
from itertools import chain

import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from tokenizer import ChineseTokenizer, EnglishTokenizer
import config
from model import TranslationDecoder, TranslationEncoder
from dataset import get_dataloader


def train_one_epoch(dataloader, encoder, decoder, optimizer, loss_function, device):
    encoder.train()
    decoder.train()
    epoch_total_loss = 0
    for inputs, targets in tqdm(dataloader, desc='训练'):
        inputs = inputs.to(device)
        # inputs.shape: [batch_size, seq_len]
        targets = targets.to(device)
        # targets.shape: [batch_size, seq_len]

        optimizer.zero_grad()

        # 编码
        encoder_outputs, context_vector = encoder(inputs)
        # context_vector.shape: [batch_size, decoder_hidden_size]

        # 解码
        decoder_input = targets[:, 0:1]
        # decoder_input.shape: [batch_size, 1]
        decoder_hidden = context_vector.unsqueeze(0)
        # decoder_hidden.shape: [1, batch_size, decoder_hidden_size]

        decoder_outputs = []
        # decoder_outputs.shape: [[batch_size, 1, vocab_size]] 长度:seq_len-1

        for t in range(1, targets.shape[1]):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden, encoder_outputs)
            # decoder_output.shape: [batch_size, 1, vocab_size]
            decoder_outputs.append(decoder_output)
            decoder_input = targets[:, t:t + 1]

        # 预测结果
        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        # decoder_outputs.shape: [batch_size, seq_len-1, vocab_size]
        decoder_outputs = decoder_outputs.reshape(-1, decoder_outputs.shape[-1])
        # decoder_outputs.shape: [batch_size * (seq_len-1), vocab_size]

        # 期望值
        decoder_targets = targets[:, 1:]
        # decoder_targets.shape: [batch_size, seq_len-1]
        decoder_targets = decoder_targets.reshape(-1)
        # decoder_targets.shape: [batch_size * (seq_len-1)]

        # 计算损失
        loss = loss_function(decoder_outputs, decoder_targets)

        loss.backward()
        optimizer.step()
        epoch_total_loss += loss.item()
    return epoch_total_loss / len(dataloader)


def train():
    # 选择设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Tokenizer
    zh_tokenizer = ChineseTokenizer.from_vocab(config.PROCESSED_DIR / 'zh_vocab.txt')
    en_tokenizer = EnglishTokenizer.from_vocab(config.PROCESSED_DIR / 'en_vocab.txt')

    # 模型
    encoder = TranslationEncoder(vocab_size=zh_tokenizer.vocab_size,
                                 padding_index=zh_tokenizer.pad_token_id).to(device)
    decoder = TranslationDecoder(vocab_size=en_tokenizer.vocab_size,
                                 padding_index=en_tokenizer.pad_token_id).to(device)

    # 加载数据
    dataloader = get_dataloader()

    # 损失函数
    loss_function = torch.nn.CrossEntropyLoss(ignore_index=en_tokenizer.pad_token_id)

    # 优化器
    optimizer = torch.optim.Adam(params=chain(encoder.parameters(), decoder.parameters()), lr=config.LEARNING_RATE)

    # tensorboard
    writer = SummaryWriter(log_dir=config.LOGS_DIR / time.strftime('%Y-%m-%d_%H-%M-%S'))

    best_loss = float('inf')
    for epoch in range(1, 1 + config.EPOCHS):
        print(f'========== Epoch: {epoch} ==========')
        avg_loss = train_one_epoch(dataloader, encoder, decoder, optimizer, loss_function, device)
        print(f'Loss: {avg_loss:.4f}')

        writer.add_scalar('Loss', avg_loss, epoch)

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(encoder.state_dict(), config.MODELS_DIR / 'encoder.pt')
            torch.save(decoder.state_dict(), config.MODELS_DIR / 'decoder.pt')
            print('模型保存成功')
        else:
            print('模型无需保存')


if __name__ == '__main__':
    train()
