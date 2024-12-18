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
    src_batch = pad_sequence(src_batch, padding_value=sp.piece_to_id('<pad>'), batch_first=True)  
    trg_batch = pad_sequence(trg_batch, padding_value=sp.piece_to_id('<pad>'), batch_first=True)  
    return src_batch, trg_batch

class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hidden_dim, n_layers, dropout):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim, padding_idx=0)
        self.lstm = nn.LSTM(emb_dim, hidden_dim, n_layers, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)

        self.fc_hidden = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc_cell = nn.Linear(hidden_dim * 2, hidden_dim)
        self.n_layers = n_layers
        self.init_weights()

    def init_weights(self):
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:  
                init.xavier_uniform_(param)  
            elif 'weight_hh' in name:  
                init.kaiming_uniform_(param, a=math.sqrt(5)) 
            elif 'bias' in name:
                init.zeros_(param)  

        init.xavier_uniform_(self.fc_hidden.weight)
        init.zeros_(self.fc_hidden.bias)
        init.xavier_uniform_(self.fc_cell.weight)
        init.zeros_(self.fc_cell.bias)

    def forward(self, src):
        embedded = self.dropout(self.embedding(src))
        outputs, (hidden, cell) = self.lstm(embedded)


        hidden = torch.cat((hidden[::2], hidden[1::2]), dim=2)  # 维度: (n_layers, batch_size, hidden_dim*2)
        cell = torch.cat((cell[::2], cell[1::2]), dim=2)  # 维度: (n_layers, batch_size, hidden_dim*2)
        
        hidden = torch.tanh(self.fc_hidden(hidden))  # (n_layers, batch_size, hidden_dim)
        cell = torch.tanh(self.fc_cell(cell))  # (n_layers, batch_size, hidden_dim)
        

        return hidden, cell

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hidden_dim, n_layers, dropout):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(output_dim, emb_dim, padding_idx=0)
        self.lstm = nn.LSTM(emb_dim, hidden_dim, n_layers, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.init_weights()

    def init_weights(self):
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                init.kaiming_uniform_(param, a=math.sqrt(5))
            elif 'bias' in name:
                init.zeros_(param)
        init.xavier_uniform_(self.fc_out.weight)
        init.zeros_(self.fc_out.bias)

    def forward(self, trg, hidden, cell):
        embedded = self.dropout(self.embedding(trg.unsqueeze(1)))
        output, (hidden, cell) = self.lstm(embedded, (hidden, cell))
        prediction = self.fc_out(output.squeeze(1))
        return prediction, hidden, cell

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size = trg.shape[0]
        trg_len = trg.shape[1]
        trg_vocab_size = self.decoder.fc_out.out_features

        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(trg.device)
        hidden, cell = self.encoder(src)
        input = trg[:, 0]

        for t in range(1, trg_len):
            output, hidden, cell = self.decoder(input, hidden, cell)
            outputs[:, t, :] = output
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            input = trg[:, t] if teacher_force else output.argmax(1)

        return outputs

def train_model(model, train_dataloader, val_dataloader, optimizer, criterion, num_epochs, patience=5):
    # Early stopping parameters
    best_val_loss = float('inf')  
    patience_counter = 0  
    model.train()  
    for epoch in range(num_epochs):
        model.train()
        teacher_forcing_ratio = max(0.9 * (0.85 ** (epoch)), 0.1)
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}")
        epoch_loss = 0
        
        for src, trg in progress_bar:
            src, trg = src.to(device), trg.to(device)
            optimizer.zero_grad()
            
            with autocast():
                output = model(src, trg, teacher_forcing_ratio=teacher_forcing_ratio)
                output_dim = output.shape[-1]
                output = output[:, 1:].contiguous().view(-1, output_dim)
                trg = trg[:, 1:].contiguous().view(-1)
                
                loss = criterion(output, trg)

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
                output = model(src, trg, teacher_forcing_ratio=0.0)
                output_dim = output.shape[-1]
                output = output[:, 1:].contiguous().view(-1, output_dim)
                trg = trg[:, 1:].contiguous().view(-1)
                loss = criterion(output, trg)
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

        torch.save(model.state_dict(), f'/home/ubuntu/models/model_epoch_{epoch + 1}.pth')

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


def translate_sentence(model, sentence, sp, max_len=50):
    model.eval()
    src_tensor = torch.tensor([[1]+sp.encode_as_ids(sentence)+[2]].to(device))

    with torch.no_grad():
        hidden, cell = model.encoder(src_tensor)

    trg_indexes = [sp.piece_to_id('<s>')]

    for _ in range(max_len):
        trg_tensor = torch.tensor([trg_indexes[-1]], dtype=torch.long).to(device)

        with torch.no_grad():
            output, hidden, cell = model.decoder(trg_tensor, hidden, cell)
            pred_token = output.argmax(1).item()

        trg_indexes.append(pred_token)

        if pred_token == sp.piece_to_id('</s>'):
            break

    trg_tokens = sp.decode_ids(trg_indexes[1:-1])
    return trg_tokens



def load_model(model_path, input_dim, output_dim, emb_dim, hidden_dim, n_layers, dropout, device):
    encoder = Encoder(input_dim, emb_dim, hidden_dim, n_layers, dropout)
    decoder = Decoder(output_dim, emb_dim, hidden_dim, n_layers, dropout)
    model = Seq2Seq(encoder, decoder).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model




saved_model_path = '/home/ubuntu/model_best.pth'
model = load_model(saved_model_path, input_dim, output_dim, emb_dim, hidden_dim, n_layers, dropout, device)


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
