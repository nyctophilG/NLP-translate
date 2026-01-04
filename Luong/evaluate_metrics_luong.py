import torch
import random
import nltk
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_https_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from utils import prepareData, normalizeString, SOS_token, EOS_token
from models import EncoderRNN, LuongAttnDecoderRNN
from bert_score import score as bert_score_func

# --- Config ---
HIDDEN_SIZE = 256
DROPOUT = 0.1
MAX_LENGTH = 15
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate(encoder, decoder, sentence, input_lang, output_lang):
    with torch.no_grad():
        input_tensor = tensorFromSentence(input_lang, sentence).to(DEVICE)
        encoder_outputs, encoder_hidden = encoder(input_tensor.unsqueeze(0))
        decoder_outputs, _, _ = decoder(encoder_outputs, encoder_hidden, None)

        _, topi = decoder_outputs.topk(1)
        decoded_ids = topi.squeeze()

        decoded_words = []
        if decoded_ids.dim() == 0:
            decoded_ids = decoded_ids.unsqueeze(0)
            
        for idx in decoded_ids:
            if idx.item() == EOS_token:
                break
            decoded_words.append(output_lang.index2word[idx.item()])
        
        return decoded_words

def tensorFromSentence(lang, sentence):
    indexes = [lang.word2index[word] for word in sentence.split(' ')]
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long)

def calculate_all_metrics(encoder, decoder, pairs, input_lang, output_lang, n_samples=100):
    print(f"Generating translations for {n_samples} samples...")
    sources, references, hypotheses = [], [], []
    test_pairs = random.sample(pairs, n_samples)

    for pair in test_pairs:
        output_words = evaluate(encoder, decoder, pair[0], input_lang, output_lang)
        hypotheses.append(' '.join(output_words))
        references.append(pair[1])
        sources.append(pair[0])

    # 1. BLEU
    smoothie = SmoothingFunction().method1
    bleu_scores = [sentence_bleu([ref.split()], hyp.split(), smoothing_function=smoothie) for ref, hyp in zip(references, hypotheses)]
    avg_bleu = sum(bleu_scores) / len(bleu_scores)

    # 2. METEOR
    meteor_scores = [meteor_score([ref.split()], hyp.split()) for ref, hyp in zip(references, hypotheses)]
    avg_meteor = sum(meteor_scores) / len(meteor_scores)

    # 3. BERTScore
    P, R, F1 = bert_score_func(hypotheses, references, lang="es", verbose=False)
    avg_bertscore = F1.mean().item()

    print("\n" + "="*30)
    print("LUONG MODEL METRICS")
    print("="*30)
    print(f"BLEU: {avg_bleu:.4f}")
    print(f"METEOR: {avg_meteor:.4f}")
    print(f"BERTScore: {avg_bertscore:.4f}")
    print("="*30)
    
    # Print 10 examples
    print("\n--- 10 Random Examples ---")
    for i in range(10):
        print(f"({i+1}) Source: {sources[i]}")
        print(f"    Target: {references[i]}")
        print(f"    Pred  : {hypotheses[i]}")

if __name__ == '__main__':
    input_lang, output_lang, pairs = prepareData('eng', 'spa', '/Users/tayfuncebeci/Desktop/nlpproject3/data/spa.txt')
    encoder = EncoderRNN(input_lang.n_words, HIDDEN_SIZE, DROPOUT).to(DEVICE)
    decoder = LuongAttnDecoderRNN('dot', HIDDEN_SIZE, output_lang.n_words, DROPOUT, MAX_LENGTH).to(DEVICE)
    encoder.load_state_dict(torch.load('/Users/tayfuncebeci/Desktop/nlpproject3/encoder_luong.pth', map_location=DEVICE))
    decoder.load_state_dict(torch.load('/Users/tayfuncebeci/Desktop/nlpproject3/decoder_luong.pth', map_location=DEVICE))
    encoder.eval()
    decoder.eval()
    calculate_all_metrics(encoder, decoder, pairs, input_lang, output_lang, n_samples=100)
