import torch
import random
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
        decoder_outputs, _, _ = decoder(encoder_outputs, encoder_hidden, None) # No teacher forcing

        _, topi = decoder_outputs.topk(1)
        decoded_ids = topi.squeeze()

        decoded_words = []
        for idx in decoded_ids:
            if idx.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            decoded_words.append(output_lang.index2word[idx.item()])
        
        return decoded_words

def tensorFromSentence(lang, sentence):
    indexes = [lang.word2index[word] for word in sentence.split(' ')]
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long)

def evaluateRandomly(encoder, decoder, pairs, input_lang, output_lang, n=5):
    print("--- Random Examples ---")
    for i in range(n):
        pair = random.choice(pairs)
        print('>', pair[0])
        print('=', pair[1])
        output_words = evaluate(encoder, decoder, pair[0], input_lang, output_lang)
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        print('')

def calculate_bleu(encoder, decoder, pairs, input_lang, output_lang, n_samples=1000):
    print(f"Calculating BLEU score on {n_samples} samples...")
    # Use a smoothing function for short sentences to avoid 0 scores
    smoothie = SmoothingFunction().method1
    total_score = 0
    
    # Select random samples for testing
    test_pairs = random.sample(pairs, n_samples)
    
    for pair in test_pairs:
        input_sentence = pair[0]
        target_sentence = pair[1]
        
        output_words = evaluate(encoder, decoder, input_sentence, input_lang, output_lang)
        
        # Remove <EOS> for score calculation
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
    print("Loading models...")
    encoder = EncoderRNN(input_lang.n_words, HIDDEN_SIZE, DROPOUT).to(DEVICE)
    decoder = AttnDecoderRNN(HIDDEN_SIZE, output_lang.n_words, DROPOUT, MAX_LENGTH).to(DEVICE)

    encoder.load_state_dict(torch.load('/Users/tayfuncebeci/Desktop/nlpproject2/encoder_bahdanau.pth', map_location=DEVICE))
    decoder.load_state_dict(torch.load('/Users/tayfuncebeci/Desktop/nlpproject2/decoder_bahdanau.pth', map_location=DEVICE))
    
    encoder.eval()
    decoder.eval()

    # 3. Evaluate
    evaluateRandomly(encoder, decoder, pairs, input_lang, output_lang)
    calculate_bleu(encoder, decoder, pairs, input_lang, output_lang, n_samples=500)
