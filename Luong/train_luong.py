import torch
import torch.nn as nn
import torch.optim as optim
import time
import math
import random

from utils import prepareData, TranslationDataset, PAD_token, SOS_token, EOS_token
from models import EncoderRNN, LuongAttnDecoderRNN

# --- Config ---
HIDDEN_SIZE = 256
BATCH_SIZE = 32
LEARNING_RATE = 0.001
DROPOUT = 0.1
EPOCHS = 10 
MAX_LENGTH = 15
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# --- Helper Functions ---
def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

def train_epoch(dataloader, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion):
    total_loss = 0
    
    for i, data in enumerate(dataloader):
        input_tensor, target_tensor = data
        input_tensor = input_tensor.to(DEVICE)
        target_tensor = target_tensor.to(DEVICE)

        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        encoder_outputs, encoder_hidden = encoder(input_tensor)
        
        # Always use teacher forcing for training to ensure shape matching
        decoder_outputs, _, _ = decoder(encoder_outputs, encoder_hidden, target_tensor)

        output_dim = decoder_outputs.shape[-1]
        seq_len = min(decoder_outputs.size(1), target_tensor.size(1))
        
        decoder_outputs = decoder_outputs[:, :seq_len, :]
        target_tensor_slice = target_tensor[:, :seq_len]
        
        loss = criterion(decoder_outputs.reshape(-1, output_dim), target_tensor_slice.reshape(-1))
        
        loss.backward()

        encoder_optimizer.step()
        decoder_optimizer.step()

        total_loss += loss.item()
        
        if (i + 1) % 100 == 0:
             print(f"Batch {i+1}/{len(dataloader)} - Loss: {loss.item():.4f}")

    return total_loss / len(dataloader)

def train(train_dataloader, encoder, decoder, n_epochs, learning_rate=0.001, print_every=100):
    start = time.time()
    print_loss_total = 0 

    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_token)

    for epoch in range(1, n_epochs + 1):
        loss = train_epoch(train_dataloader, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)
        print_loss_total += loss

        if epoch % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, epoch / n_epochs),
                                        epoch, epoch / n_epochs * 100, print_loss_avg))

if __name__ == '__main__':
    # 1. Load Data (pointing to data in nlpproject3)
    input_lang, output_lang, pairs = prepareData('eng', 'spa', '/Users/tayfuncebeci/Desktop/nlpproject3/data/spa.txt')
    
    from torch.nn.utils.rnn import pad_sequence
    def collate_batch(batch):
        input_tensors, target_tensors = zip(*batch)
        input_padded = pad_sequence(input_tensors, batch_first=True, padding_value=PAD_token)
        target_padded = pad_sequence(target_tensors, batch_first=True, padding_value=PAD_token)
        return input_padded, target_padded

    train_data = TranslationDataset(input_lang, output_lang, pairs)
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch)

    # 2. Initialize Models (Using LuongAttnDecoderRNN)
    encoder = EncoderRNN(input_lang.n_words, HIDDEN_SIZE, DROPOUT).to(DEVICE)
    decoder = LuongAttnDecoderRNN('dot', HIDDEN_SIZE, output_lang.n_words, DROPOUT, MAX_LENGTH).to(DEVICE)

    # 3. Train
    print("Starting Training (Luong Attention)...")
    train(train_dataloader, encoder, decoder, EPOCHS, LEARNING_RATE, print_every=1)
    
    # 4. Save Models
    torch.save(encoder.state_dict(), '/Users/tayfuncebeci/Desktop/nlpproject3/encoder_luong.pth')
    torch.save(decoder.state_dict(), '/Users/tayfuncebeci/Desktop/nlpproject3/decoder_luong.pth')
    print("Models saved in nlpproject3.")
