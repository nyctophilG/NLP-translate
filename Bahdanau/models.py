import torch
import torch.nn as nn
import torch.nn.functional as F

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_p=0.1):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, input):
        embedded = self.dropout(self.embedding(input))
        output, hidden = self.gru(embedded)
        return output, hidden

class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size):
        super(BahdanauAttention, self).__init__()
        self.Wa = nn.Linear(hidden_size, hidden_size)
        self.Ua = nn.Linear(hidden_size, hidden_size)
        self.Va = nn.Linear(hidden_size, 1)

    def forward(self, query, keys):
        # query: [batch_size, 1, hidden_size] (Decoder hidden state)
        # keys: [batch_size, seq_len, hidden_size] (Encoder outputs)
        
        scores = self.Va(torch.tanh(self.Wa(query) + self.Ua(keys)))
        # scores: [batch_size, seq_len, 1]
        
        scores = scores.squeeze(2).unsqueeze(1)
        # scores: [batch_size, 1, seq_len]

        weights = F.softmax(scores, dim=-1)
        # weights: [batch_size, 1, seq_len]
        
        context = torch.bmm(weights, keys)
        # context: [batch_size, 1, hidden_size]

        return context, weights

class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=15):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attention = BahdanauAttention(self.hidden_size)
        self.gru = nn.GRU(self.hidden_size * 2, self.hidden_size, batch_first=True)
        self.out = nn.Linear(self.hidden_size, self.output_size)
        self.dropout = nn.Dropout(self.dropout_p)

    def forward(self, encoder_outputs, encoder_hidden, target_tensor=None):
        batch_size = encoder_outputs.size(0)
        decoder_input = torch.empty(batch_size, 1, dtype=torch.long, device=encoder_outputs.device).fill_(1) # SOS_token = 1
        decoder_hidden = encoder_hidden
        decoder_outputs = []
        attentions = []

        # Determine if we run in training mode (teacher forcing) or inference
        # Simple loop for max_length
        loop_length = target_tensor.size(1) if target_tensor is not None else self.max_length

        for i in range(loop_length):
            decoder_output, decoder_hidden, attn_weights = self.forward_step(
                decoder_input, decoder_hidden, encoder_outputs
            )
            decoder_outputs.append(decoder_output)
            attentions.append(attn_weights)

            if target_tensor is not None:
                # Teacher forcing: Feed the target as the next input
                decoder_input = target_tensor[:, i].unsqueeze(1) # Teacher forcing
            else:
                # Without teacher forcing: use its own predictions as the next input
                _, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze(-1).detach()  # detach from history as input

        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        attentions = torch.cat(attentions, dim=1)
        return decoder_outputs, decoder_hidden, attentions

    def forward_step(self, input, hidden, encoder_outputs):
        embedded = self.dropout(self.embedding(input))

        # Calculate attention context
        # hidden is [1, batch, hidden], permute to [batch, 1, hidden] for attention calculation
        query = hidden.permute(1, 0, 2)
        context, attn_weights = self.attention(query, encoder_outputs)
        
        input_gru = torch.cat((embedded, context), dim=2)
        output, hidden = self.gru(input_gru, hidden)
        output = self.out(output)

        return output, hidden, attn_weights

class LuongAttention(nn.Module):
    def __init__(self, method, hidden_size):
        super(LuongAttention, self).__init__()
        self.method = method
        if self.method not in ['dot', 'general', 'concat']:
            raise ValueError(self.method, "is not an appropriate attention method.")
        self.hidden_size = hidden_size
        
        if self.method == 'general':
            self.attn = nn.Linear(self.hidden_size, hidden_size)
        elif self.method == 'concat':
            self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
            self.v = nn.Parameter(torch.FloatTensor(hidden_size))

    def dot_score(self, hidden, encoder_output):
        return torch.sum(hidden * encoder_output, dim=2)

    def general_score(self, hidden, encoder_output):
        energy = self.attn(encoder_output)
        return torch.sum(hidden * energy, dim=2)

    def concat_score(self, hidden, encoder_output):
        energy = self.attn(torch.cat((hidden.expand(encoder_output.size(0), -1, -1), encoder_output), 2)).tanh()
        return torch.sum(self.v * energy, dim=2)

    def forward(self, hidden, encoder_outputs):
        # hidden: [1, batch, hidden_size]
        # encoder_outputs: [batch, seq_len, hidden_size]
        
        # Calculate attention weights (energies) based on the given method
        if self.method == 'general':
            attn_energies = self.general_score(hidden.permute(1, 0, 2), encoder_outputs)
        elif self.method == 'concat':
            # Not fully implemented for batched properly here for brevity, focusing on 'dot' or 'general' usually
            pass 
        elif self.method == 'dot':
             attn_energies = self.dot_score(hidden.permute(1, 0, 2), encoder_outputs)
        
        # attn_energies: [batch, seq_len]
        return F.softmax(attn_energies, dim=1).unsqueeze(1)

class LuongAttnDecoderRNN(nn.Module):
    def __init__(self, attn_model, hidden_size, output_size, dropout_p=0.1, max_length=15):
        super(LuongAttnDecoderRNN, self).__init__()
        self.attn_model = attn_model
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.attention = LuongAttention(attn_model, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.concat = nn.Linear(hidden_size * 2, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, encoder_outputs, encoder_hidden, target_tensor=None):
        batch_size = encoder_outputs.size(0)
        decoder_input = torch.empty(batch_size, 1, dtype=torch.long, device=encoder_outputs.device).fill_(1) # SOS
        decoder_hidden = encoder_hidden
        decoder_outputs = []
        attentions = []

        loop_length = target_tensor.size(1) if target_tensor is not None else self.max_length

        for i in range(loop_length):
            decoder_output, decoder_hidden, attn_weights = self.forward_step(
                decoder_input, decoder_hidden, encoder_outputs
            )
            decoder_outputs.append(decoder_output)
            attentions.append(attn_weights)
            
            if target_tensor is not None:
                decoder_input = target_tensor[:, i].unsqueeze(1)
            else:
                _, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze(-1).detach()

        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        attentions = torch.cat(attentions, dim=1)
        return decoder_outputs, decoder_hidden, attentions

    def forward_step(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input)
        embedded = self.dropout(embedded)

        # Forward through GRU first (Luong differs from Bahdanau here)
        rnn_output, hidden = self.gru(embedded, hidden)
        
        # Calculate attention weights from current GRU output and encoder outputs
        attn_weights = self.attention(hidden, encoder_outputs)
        
        # Multiply attention weights to encoder outputs to get new "weighted sum" context vector
        context = attn_weights.bmm(encoder_outputs)
        
        # Concatenate weighted context vector and GRU output using Luong eq. 5
        rnn_output = rnn_output.squeeze(1)
        context = context.squeeze(1)
        concat_input = torch.cat((rnn_output, context), 1)
        concat_output = torch.tanh(self.concat(concat_input))
        
        # Predict next word using Luong eq. 6
        output = self.out(concat_output)
        output = output.unsqueeze(1) # Restore sequence dim

        return output, hidden, attn_weights
