import sentencepiece as spm
import math
import torch.nn.init as init
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from collections import Counter
from sklearn.model_selection import train_test_split
import random
import pandas as pd
from nltk.translate.bleu_score import sentence_bleu
from torch.nn.utils.rnn import pad_sequence
import jieba
import json
import nltk
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()


class TranslationDataset(Dataset):
    def __init__(self, source_sentences, target_sentences):
        self.source_sentences = source_sentences
        self.target_sentences = target_sentences

    def __len__(self):
        return len(self.source_sentences)

    def __getitem__(self, idx):
        return torch.tensor(self.source_sentences[idx], dtype=torch.long), torch.tensor(self.target_sentences[idx], dtype=torch.long)

# pad sequences
def collate_fn(batch):
    src_batch, trg_batch = zip(*batch)
    src_batch = pad_sequence(src_batch, padding_value=zh_sp.piece_to_id('<pad>'), batch_first=True)
    trg_batch = pad_sequence(trg_batch, padding_value=en_sp.piece_to_id('<pad>'), batch_first=True)
    return src_batch, trg_batch


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Add batch dimension
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)].detach()


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        assert self.head_dim * num_heads == d_model, "d_model must be divisible by num_heads"

        self.query_linear = nn.Linear(d_model, d_model)
        self.key_linear = nn.Linear(d_model, d_model)
        self.value_linear = nn.Linear(d_model, d_model)
        self.fc_out = nn.Linear(d_model, d_model)

    def forward(self, query, key, value, mask=None):
        batch_size = query.shape[0]

        query = self.query_linear(query).reshape(batch_size, -1, self.num_heads, self.head_dim)
        key = self.key_linear(key).reshape(batch_size, -1, self.num_heads, self.head_dim)
        value = self.value_linear(value).reshape(batch_size, -1, self.num_heads, self.head_dim)

        query = query.transpose(1, 2)  # (batch_size, num_heads, seq_len, head_dim)
        key = key.transpose(1, 2)      # (batch_size, num_heads, seq_len, head_dim)
        value = value.transpose(1, 2)  # (batch_size, num_heads, seq_len, head_dim)

        energy = torch.matmul(query, key.transpose(-2, -1))  # (batch_size, num_heads, seq_len, seq_len)
        if mask is not None:
            energy = energy.to(torch.float32)
            energy = energy.masked_fill(mask == 0, float('-inf'))

        attention = torch.softmax(energy / math.sqrt(self.head_dim), dim=-1)
        out = torch.matmul(attention, value)  # (batch_size, num_heads, seq_len, head_dim)

        out = out.transpose(1, 2).reshape(batch_size, -1, self.d_model)
        out = self.fc_out(out)  # (batch_size, seq_len, d_model)

        return out

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, ff_hidden_dim, dropout):
        super(EncoderLayer, self).__init__()
        self.multihead_attention = MultiHeadAttention(d_model, num_heads)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ff_hidden_dim),
            nn.ReLU(),
            nn.Linear(ff_hidden_dim, d_model)
        )
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask):
        attention_out = self.multihead_attention(src, src, src, src_mask)
        src = self.layer_norm1(src + self.dropout(attention_out))

        ffn_out = self.ffn(src)
        src = self.layer_norm2(src + self.dropout(ffn_out))

        return src


class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, ff_hidden_dim, dropout):
        super(DecoderLayer, self).__init__()
        self.multihead_attention1 = MultiHeadAttention(d_model, num_heads)
        self.multihead_attention2 = MultiHeadAttention(d_model, num_heads)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ff_hidden_dim),
            nn.ReLU(),
            nn.Linear(ff_hidden_dim, d_model)
        )
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.layer_norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, trg, enc_output, trg_mask, enc_mask):
        attention_out1 = self.multihead_attention1(trg, trg, trg, trg_mask)
        trg = self.layer_norm1(trg + self.dropout(attention_out1))

        attention_out2 = self.multihead_attention2(trg, enc_output, enc_output, enc_mask)
        trg = self.layer_norm2(trg + self.dropout(attention_out2))

        ffn_out = self.ffn(trg)
        trg = self.layer_norm3(trg + self.dropout(ffn_out))

        return trg


class Transformer(nn.Module):
    def __init__(self, src_vocab_size, trg_vocab_size, d_model, num_heads, num_layers, ff_hidden_dim, dropout, max_len=5000):
        super(Transformer, self).__init__()
        self.src_embedding = nn.Embedding(src_vocab_size, d_model, padding_idx=zh_sp.piece_to_id('<pad>'))
        self.trg_embedding = nn.Embedding(trg_vocab_size, d_model, padding_idx=en_sp.piece_to_id('<pad>'))
        self.positional_encoding = PositionalEncoding(d_model, max_len)

        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, ff_hidden_dim, dropout) for _ in range(num_layers)])
        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, num_heads, ff_hidden_dim, dropout) for _ in range(num_layers)])

        self.fc_out = nn.Linear(d_model, trg_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def make_src_mask(self, src):
        # src: (batch_size, src_len)
        src_mask = (src != zh_sp.piece_to_id('<pad>')).unsqueeze(1).unsqueeze(2)  # (batch_size, 1, 1, src_len)
        return src_mask  # 1表示保留，0表示遮蔽

    def make_trg_mask(self, trg):
        # trg: (batch_size, trg_len)
        trg_pad_mask = (trg != en_sp.piece_to_id('<pad>')).unsqueeze(1).unsqueeze(2)  # (batch_size, 1, 1, trg_len)
        trg_len = trg.size(1)
        trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len), device=trg.device)).bool()  # (trg_len, trg_len)
        trg_mask = trg_pad_mask & trg_sub_mask  # (batch_size, 1, trg_len, trg_len)
        return trg_mask

    def forward(self, src, trg):
        src_mask = self.make_src_mask(src)  # (batch_size, 1, 1, src_len)
        trg_mask = self.make_trg_mask(trg)  # (batch_size, 1, trg_len, trg_len)

        src = self.src_embedding(src) * math.sqrt(self.src_embedding.embedding_dim)
        trg = self.trg_embedding(trg) * math.sqrt(self.trg_embedding.embedding_dim)

        src = self.positional_encoding(src)
        trg = self.positional_encoding(trg)

        for layer in self.encoder_layers:
            src = layer(src, src_mask)

        for layer in self.decoder_layers:
            trg = layer(trg, src, trg_mask, src_mask)

        output = self.fc_out(trg)
        return output


def train_model(model, train_dataloader, val_dataloader, optimizer, criterion, num_epochs, patience=5):
    # Early stopping parameters
    best_val_loss = float('inf')
    patience_counter = 0
    model.train()
    for epoch in range(num_epochs):
        model.train()
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}")
        epoch_loss = 0

        for src, trg in progress_bar:
            src, trg = src.to(device), trg.to(device)
            optimizer.zero_grad()

            
            trg_input = trg[:, :-1]
            trg_output = trg[:, 1:]

            with autocast():
                output = model(src, trg_input)  # output: (batch_size, trg_len -1, trg_vocab_size)
                output_dim = output.shape[-1]
                output = output.reshape(-1, output_dim)  # ((batch_size * (trg_len -1)), trg_vocab_size)
                trg_output = trg_output.reshape(-1)      # (batch_size * (trg_len -1))

                loss = criterion(output, trg_output)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

        print(f'Epoch {epoch + 1}, Loss: {epoch_loss / len(train_dataloader)}')

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for src, trg in val_dataloader:
                src, trg = src.to(device), trg.to(device)
                trg_input = trg[:, :-1]
                trg_output = trg[:, 1:]

                output = model(src, trg_input)
                output_dim = output.shape[-1]
                output = output.reshape(-1, output_dim)
                trg_output = trg_output.reshape(-1)
                loss = criterion(output, trg_output)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_dataloader)
        print(f'Validation Loss: {avg_val_loss}')

        # Early stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0

            torch.save(model.state_dict(), f'model_best.pth')
            print(f"Saved best model at epoch {epoch + 1} with validation loss: {best_val_loss}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch + 1}. Validation loss did not improve.")
                break

        #torch.save(model.state_dict(), f'model_epoch_{epoch + 1}.pth')


def translate_sentence(model, sentence, max_len=50):
    model.eval()
    src_tensor = torch.tensor([zh_sp.piece_to_id('<s>')] + zh_sp.encode_as_ids(sentence) + [zh_sp.piece_to_id('</s>')]).unsqueeze(0).to(device)

    trg_indexes = [en_sp.piece_to_id('<s>')]

    for _ in range(max_len):
        trg_tensor = torch.tensor(trg_indexes, dtype=torch.long).unsqueeze(0).to(device)  # (1, len(trg))
        with torch.no_grad():
            output = model(src_tensor, trg_tensor)  # (1, len(trg), trg_vocab_size)
            pred_token = output.argmax(-1)[:, -1].item()

        trg_indexes.append(pred_token)

        if pred_token == en_sp.piece_to_id('</s>'):
            break

    trg_tokens = en_sp.decode_ids(trg_indexes[1:-1])
    return trg_tokens


def load_model(model_path, src_vocab_size, trg_vocab_size, d_model, num_heads, num_layers, ff_hidden_dim, dropout, device):
    model = Transformer(src_vocab_size, trg_vocab_size, d_model, num_heads, num_layers, ff_hidden_dim, dropout).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


data = []
with open('middle.json', "r", encoding="utf-8") as file:
    for line in file:
        line = line.strip()
        if line:
            data.append(json.loads(line.strip()))
df = pd.DataFrame(data)

source_sentences = [sentence for sentence in df['chinese'].tolist()]
target_sentences = [sentence for sentence in df['english'].tolist()]

spm.SentencePieceTrainer.train(input='en.txt', model_prefix='en_spm', vocab_size=50000, character_coverage=0.9995, user_defined_symbols='<pad>')
en_sp = spm.SentencePieceProcessor(model_file='en_spm.model')
spm.SentencePieceTrainer.train(input='zh.txt', model_prefix='zh_spm', vocab_size=50000, character_coverage=0.9995, user_defined_symbols='<pad>')
zh_sp = spm.SentencePieceProcessor(model_file='zh_spm.model')

source_tokenized = [ [zh_sp.piece_to_id('<s>')] + zh_sp.encode_as_ids(sentence) + [zh_sp.piece_to_id('</s>')] for sentence in source_sentences ]
target_tokenized = [ [en_sp.piece_to_id('<s>')] + en_sp.encode_as_ids(sentence) + [en_sp.piece_to_id('</s>')] for sentence in target_sentences ]

train_src, val_src, train_trg, val_trg = train_test_split(source_tokenized, target_tokenized, test_size=0.1, random_state=42)
train_dataset = TranslationDataset(train_src, train_trg)
val_dataset = TranslationDataset(val_src, val_trg)

train_dataloader = DataLoader(train_dataset, batch_size=128, collate_fn=collate_fn, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=128, collate_fn=collate_fn)

en_vocab_size = en_sp.piece_size()
zh_vocab_size = zh_sp.piece_size()

src_vocab_size = zh_vocab_size
trg_vocab_size = en_vocab_size
d_model = 256
num_heads = 8
num_layers = 3
ff_hidden_dim = 512
dropout = 0.2

model = Transformer(src_vocab_size, trg_vocab_size, d_model, num_heads, num_layers, ff_hidden_dim, dropout).to(device)

optimizer = optim.Adam(model.parameters(), lr=0.0001)
criterion = nn.CrossEntropyLoss(ignore_index=zh_sp.piece_to_id('<pad>'))

train_model(model, train_dataloader, val_dataloader, optimizer, criterion, num_epochs=50)

saved_model_path = 'model_best.pth'
model = load_model(saved_model_path, src_vocab_size, trg_vocab_size, d_model, num_heads, num_layers, ff_hidden_dim, dropout, device)

sentences = [
    "我需要你的帮助。",
    "他敲了敲教室的门。",
    "如果有问题的话，请在明天中午给我打电话。",
    "但真正的重建或恢复工作却一拖再拖，也许长达数年之久。"
]

for sentence in sentences:
    translation = translate_sentence(model, sentence)
    print(f'原句: {sentence}')
    print(f'翻译: {translation}\n')
