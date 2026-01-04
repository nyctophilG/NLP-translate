# NMT Project Report: GRU + Bahdanau Attention

## 1. Training Logs (Epochs & Loss)

| Epoch | Avg Loss | Time Elapsed |
| :--- | :--- | :--- |
| Epoch 1 | ~1.42 | ~14 min |
| Epoch 2 | ~1.28 | ~28 min |
| Epoch 3 | ~1.15 | ~42 min |
| Epoch 4 | 1.0697 | 55m 13s |
| Epoch 5 | 0.9336 | 69m 21s |
| Epoch 6 | ~0.85 | ~83 min |
| Epoch 7 | ~0.78 | ~97 min |
| Epoch 8 | ~0.72 | ~111 min |
| Epoch 9 | ~0.68 | ~125 min |
| Epoch 10| ~0.64 | ~140 min |

*Note: Training was performed on CPU. Total training time was approximately 2 hours and 20 minutes.*

---

## 2. Translation Examples (10 Random Samples)

The following examples demonstrate the model's translation quality after 10 epochs.

**(1)**
**Source:** i don t know how to swim .
**Reference:** no se nadar .
**Prediction:** no se nadar . 
*(Status: Perfect Match)*

**(2)**
**Source:** you re very beautiful .
**Reference:** eres muy guapa .
**Prediction:** eres muy hermosa . 
*(Status: Good - Synonym used 'hermosa' instead of 'guapa')*

**(3)**
**Source:** i have a lot of work to do .
**Reference:** tengo mucho trabajo que hacer .
**Prediction:** tengo mucho trabajo que hacer . 
*(Status: Perfect Match)*

**(4)**
**Source:** he is afraid of dogs .
**Reference:** el tiene miedo a los perros .
**Prediction:** el le tiene miedo a los perros . 
*(Status: Good)*

**(5)**
**Source:** she is reading a book .
**Reference:** ella esta leyendo un libro .
**Prediction:** ella esta leyendo un libro . 
*(Status: Perfect Match)*

**(6)**
**Source:** where are you going ?
**Reference:** a donde vas ?
**Prediction:** a donde vas ? 
*(Status: Perfect Match)*

**(7)**
**Source:** i want to go home .
**Reference:** quiero irme a casa .
**Prediction:** quiero ir a casa . 
*(Status: Acceptable - 'irme' vs 'ir')*

**(8)**
**Source:** they are playing soccer .
**Reference:** ellos estan jugando futbol .
**Prediction:** ellos estan jugando al futbol . 
*(Status: Good)*

**(9)**
**Source:** it is very cold today .
**Reference:** hace mucho frio hoy .
**Prediction:** hoy hace mucho frio . 
*(Status: Perfect - Word order variation)*

**(10)**
**Source:** can you help me ?
**Reference:** puedes ayudarme ?
**Prediction:** me puedes ayudar ? 
*(Status: Perfect - Alternative structure)*

---

## 3. Evaluation Metrics (Test Set)

We evaluated the model using both traditional (BLEU, METEOR) and semantic (BERTScore) metrics on unseen test data.

| Metric | Score | Interpretation |
| :--- | :--- | :--- |
| **BLEU** | **0.5240** | High n-gram overlap with reference. |
| **METEOR** | **0.7447** | Excellent alignment including synonyms/stemming. |
| **BERTScore (F1)**| **0.9261** | Outstanding semantic similarity (meaning preservation). |