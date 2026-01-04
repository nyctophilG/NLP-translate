import torch
import random
import nltk
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from utils import prepareData, normalizeString, SOS_token, EOS_token
from models import EncoderRNN, AttnDecoderRNN
import evaluate # HuggingFace evaluate
from bert_score import score as bert_score_func
# Try importing COMET, handle if not installed/configured
try:
    from comet import download_model, load_from_checkpoint
    COMET_AVAILABLE = True
except ImportError:
    COMET_AVAILABLE = False
    print("Warning: 'unbabel-comet' not found. COMET score will be skipped.")

# --- Config ---
HIDDEN_SIZE = 256
DROPOUT = 0.1
MAX_LENGTH = 15
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Download NLTK data for Meteor
nltk.download('wordnet')
nltk.download('omw-1.4')

def evaluate(encoder, decoder, sentence, input_lang, output_lang):
    with torch.no_grad():
        input_tensor = tensorFromSentence(input_lang, sentence).to(DEVICE)
        encoder_outputs, encoder_hidden = encoder(input_tensor.unsqueeze(0))
        decoder_outputs, _, _ = decoder(encoder_outputs, encoder_hidden, None)

        _, topi = decoder_outputs.topk(1)
        decoded_ids = topi.squeeze()

        decoded_words = []
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
    print(f"Generating translations for {n_samples} samples to calculate metrics...")
    print("This might take a while (downloading metric models)...")

    sources = []
    references = []
    hypotheses = []

    # Select random samples
    test_pairs = random.sample(pairs, n_samples)

    for pair in test_pairs:
        input_sentence = pair[0]
        target_sentence = pair[1]
        
        output_words = evaluate(encoder, decoder, input_sentence, input_lang, output_lang)
        output_sentence = ' '.join(output_words)

        sources.append(input_sentence)
        references.append(target_sentence)
        hypotheses.append(output_sentence)

    # 1. BLEU (NLTK)
    print("\n--- Calculating BLEU ---")
    smoothie = SmoothingFunction().method1
    bleu_scores = []
    for ref, hyp in zip(references, hypotheses):
        # NLTK expects tokenized lists
        bleu_scores.append(sentence_bleu([ref.split()], hyp.split(), smoothing_function=smoothie))
    avg_bleu = sum(bleu_scores) / len(bleu_scores)
    print(f"BLEU Score: {avg_bleu:.4f}")

    # 2. METEOR (NLTK)
    print("\n--- Calculating METEOR ---")
    meteor_scores = []
    for ref, hyp in zip(references, hypotheses):
        # NLTK meteor expects list of references (tokenized) and hypothesis (tokenized)
        meteor_scores.append(meteor_score([ref.split()], hyp.split()))
    avg_meteor = sum(meteor_scores) / len(meteor_scores)
    print(f"METEOR Score: {avg_meteor:.4f}")

    # 3. BERTScore
    print("\n--- Calculating BERTScore (downloading model if first time) ---")
    # lang="es" because target is Spanish
    P, R, F1 = bert_score_func(hypotheses, references, lang="es", verbose=True)
    avg_bertscore = F1.mean().item()
    print(f"BERTScore F1: {avg_bertscore:.4f}")

    # 4. COMET
    # COMET requires Source, Hypothesis, and Reference
    # It uses a pretrained model like 'Unbabel/wmt22-comet-da'
    if COMET_AVAILABLE:
        print("\n--- Calculating COMET (Downloading model ~2GB if first time) ---")
        try:
            # We use the evaluate library wrapper or direct usage
            # Let's use direct usage for better control as evaluate wrapper can be tricky with paths
            # Load default model
            model_path = download_model("Unbabel/wmt22-comet-da")
            model = load_from_checkpoint(model_path)
            
            data = []
            for src, hyp, ref in zip(sources, hypotheses, references):
                data.append({"src": src, "mt": hyp, "ref": ref})
            
            model_output = model.predict(data, batch_size=8, gpus=0) # gpus=0 for CPU
            avg_comet = model_output.system_score
            print(f"COMET Score: {avg_comet:.4f}")
        except Exception as e:
            print(f"COMET Calculation failed: {e}")
            avg_comet = "N/A"
    else:
        print("\n--- COMET Skipped (Library not installed) ---")
        avg_comet = "N/A"

    # Summary
    print("\n" + "="*30)
    print("FINAL METRICS REPORT")
    print("="*30)
    print(f"Model: GRU + Bahdanau")
    print(f"Samples: {n_samples}")
    print(f"BLEU: {avg_bleu:.4f}")
    print(f"METEOR: {avg_meteor:.4f}")
    print(f"BERTScore: {avg_bertscore:.4f}")
    print(f"COMET: {avg_comet}")
    print("="*30)

if __name__ == '__main__':
    # 1. Load Data
    input_lang, output_lang, pairs = prepareData('eng', 'spa', '/Users/tayfuncebeci/Desktop/nlpproject2/data/spa.txt')

    # 2. Load Models
    print("Loading Model...")
    encoder = EncoderRNN(input_lang.n_words, HIDDEN_SIZE, DROPOUT).to(DEVICE)
    decoder = AttnDecoderRNN(HIDDEN_SIZE, output_lang.n_words, DROPOUT, MAX_LENGTH).to(DEVICE)

    encoder.load_state_dict(torch.load('/Users/tayfuncebeci/Desktop/nlpproject2/encoder_bahdanau.pth', map_location=DEVICE))
    decoder.load_state_dict(torch.load('/Users/tayfuncebeci/Desktop/nlpproject2/decoder_bahdanau.pth', map_location=DEVICE))
    
    encoder.eval()
    decoder.eval()

    # 3. Calculate
    calculate_all_metrics(encoder, decoder, pairs, input_lang, output_lang, n_samples=100)
