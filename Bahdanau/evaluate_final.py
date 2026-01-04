import torch
import random
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from utils import prepareData, normalizeString, SOS_token, EOS_token
from models import EncoderRNN, AttnDecoderRNN
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# --- Config ---
HIDDEN_SIZE = 256
DROPOUT = 0.1
MAX_LENGTH = 15
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate(encoder, decoder, sentence, input_lang, output_lang):
    with torch.no_grad():
        input_tensor = tensorFromSentence(input_lang, sentence).to(DEVICE)
        
        encoder_outputs, encoder_hidden = encoder(input_tensor.unsqueeze(0)) # Add batch dim
        decoder_outputs, decoder_hidden, decoder_attn = decoder(encoder_outputs, encoder_hidden, None) # No teacher forcing

        _, topi = decoder_outputs.topk(1)
        decoded_ids = topi.squeeze()

        decoded_words = []
        for idx in decoded_ids:
            if idx.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            decoded_words.append(output_lang.index2word[idx.item()])
        
        return decoded_words, decoder_attn

def tensorFromSentence(lang, sentence):
    indexes = [lang.word2index[word] for word in sentence.split(' ')]
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long)

def showAttention(input_sentence, output_words, attentions, save_path='attention.png'):
    # Set up figure with colorbar
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    # attentions: [batch=1, output_len, input_len] -> squeeze batch
    # Also trim to actual lengths
    cax = ax.matshow(attentions.squeeze(0).cpu().numpy(), cmap='bone')
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels([''] + input_sentence.split(' ') + ['<EOS>'], rotation=90)
    ax.set_yticklabels([''] + output_words)

    # Show label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.savefig(save_path)
    print(f"Attention heatmap saved to {save_path}")
    plt.close()

def evaluateAndShowAttention(encoder, decoder, input_sentence, input_lang, output_lang, save_path):
    output_words, attentions = evaluate(encoder, decoder, input_sentence, input_lang, output_lang)
    print('input =', input_sentence)
    print('output =', ' '.join(output_words))
    showAttention(input_sentence, output_words, attentions, save_path)

def evaluateRandomly(encoder, decoder, pairs, input_lang, output_lang, n=20):
    print(f"--- Generating {n} Random Examples ---")
    results = []
    for i in range(n):
        pair = random.choice(pairs)
        output_words, _ = evaluate(encoder, decoder, pair[0], input_lang, output_lang)
        output_sentence = ' '.join(output_words)
        
        print(f"Example {i+1}:")
        print(f"Source    : {pair[0]}")
        print(f"Reference : {pair[1]}")
        print(f"Prediction: {output_sentence}")
        print("-" * 30)
        
        results.append((pair[0], pair[1], output_sentence))
    return results

def calculate_bleu(encoder, decoder, pairs, input_lang, output_lang, n_samples=1000):
    print(f"Calculating BLEU score on {n_samples} samples...")
    smoothie = SmoothingFunction().method1
    total_score = 0
    test_pairs = random.sample(pairs, n_samples)
    
    for pair in test_pairs:
        input_sentence = pair[0]
        target_sentence = pair[1]
        output_words, _ = evaluate(encoder, decoder, input_sentence, input_lang, output_lang)
        if '<EOS>' in output_words:
            output_words = output_words[:-1]
        ref = [target_sentence.split()]
        hyp = output_words
        score = sentence_bleu(ref, hyp, smoothing_function=smoothie)
        total_score += score

    avg_score = total_score / n_samples
    print(f"Average BLEU Score: {avg_score:.4f}")
    return avg_score

if __name__ == '__main__':
    # 1. Load Data
    input_lang, output_lang, pairs = prepareData('eng', 'spa', '/Users/tayfuncebeci/Desktop/nlpproject2/data/spa.txt')

    # 2. Load Models
    print("Loading Bahdanau Model...")
    encoder = EncoderRNN(input_lang.n_words, HIDDEN_SIZE, DROPOUT).to(DEVICE)
    decoder = AttnDecoderRNN(HIDDEN_SIZE, output_lang.n_words, DROPOUT, MAX_LENGTH).to(DEVICE)

    encoder.load_state_dict(torch.load('/Users/tayfuncebeci/Desktop/nlpproject2/encoder_bahdanau.pth', map_location=DEVICE))
    decoder.load_state_dict(torch.load('/Users/tayfuncebeci/Desktop/nlpproject2/decoder_bahdanau.pth', map_location=DEVICE))
    
    encoder.eval()
    decoder.eval()

    # 3. Generate 20 Examples
    evaluateRandomly(encoder, decoder, pairs, input_lang, output_lang, n=20)
    
    # 4. Calculate BLEU
    bleu = calculate_bleu(encoder, decoder, pairs, input_lang, output_lang, n_samples=500)

    # 5. Generate Heatmap for a specific sentence
    # Picking a somewhat long sentence for good visualization
    sample_sentence = random.choice(pairs)[0]
    evaluateAndShowAttention(encoder, decoder, sample_sentence, input_lang, output_lang, save_path='/Users/tayfuncebeci/Desktop/nlpproject2/heatmap_example.png')
