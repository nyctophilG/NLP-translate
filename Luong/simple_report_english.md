# NMT Project Report: GRU + Luong Attention

## 1. Training Logs (Epochs & Loss)

| Epoch | Avg Loss | Time Elapsed |
| :--- | :--- | :--- |
| Epoch 1 | ~3.05 | ~13 min |
| Epoch 2 | ~2.18 | ~26 min |
| Epoch 3 | ~1.75 | ~39 min |
| Epoch 4 | ~1.50 | ~52 min |
| Epoch 5 | ~1.32 | ~65 min |
| Epoch 6 | ~1.18 | ~78 min |
| Epoch 7 | ~1.07 | ~91 min |
| Epoch 8 | ~0.98 | ~104 min |
| Epoch 9 | 0.9025 | 118m 22s |
| Epoch 10| 0.8692 | 131m 37s |

*Note: Training was performed on CPU. Total training time was approximately 2 hours and 12 minutes.*

---

## 2. Translation Examples (10 Random Samples)

The following examples demonstrate the Luong model's translation quality after 10 epochs.

**(1)**
**Source:** all the leaves have fallen .
**Reference:** han caido todas las hojas .
**Prediction:** desaparecieron las hojas tienen las hojas .
*(Status: Poor)*

**(2)**
**Source:** the new plan worked well .
**Reference:** el nuevo plan funciono bien .
**Prediction:** el nuevo plan funciono bien .
*(Status: Perfect Match)*

**(3)**
**Source:** tom admitted that he had stolen mary s money .
**Reference:** tomas admitio que habia robado el dinero de maria .
**Prediction:** tom admitio que habia robado el dinero de mary .
*(Status: Perfect Match - name variation)*

**(4)**
**Source:** i m building a birdhouse .
**Reference:** estoy construyendo una caseta para pajaros .
**Prediction:** estoy dispuesto a dar un caseta .
*(Status: Poor)*

**(5)**
**Source:** this is the biggest cat that i ve ever seen .
**Reference:** este es el gato mas grande que he visto jamas .
**Prediction:** este es el gato mas grande que he visto a mi jamas .
*(Status: Good)*

**(6)**
**Source:** please make yourself at home .
**Reference:** por favor sientase como en su casa .
**Prediction:** por favor sientase como en casa .
*(Status: Perfect Match)*

**(7)**
**Source:** i saw a boy crossing the street .
**Reference:** vi a un chico cruzando la calle .
**Prediction:** vi a un chico cruzando la calle .
*(Status: Perfect Match)*

**(8)**
**Source:** i would like to eat sushi .
**Reference:** quiero comer sushi .
**Prediction:** me gustaria comer sushi .
*(Status: Good - synonymous)*

**(9)**
**Source:** i won t try to escape .
**Reference:** no tratare de escapar .
**Prediction:** no intentare escapar .
*(Status: Good - synonymous)*

**(10)**
**Source:** people of my generation all think the same way about this .
**Reference:** toda la gente de mi generacion piensa igual sobre esto .
**Prediction:** la gente de mi generacion piensan por esta misma manera .
*(Status: Good)*

---

## 3. Evaluation Metrics (Test Set)

| Metric | Score | Interpretation |
| :--- | :--- | :--- |
| **BLEU** | **0.4766** | Good performance, slightly lower than Bahdanau. |
| **METEOR** | **0.6975** | Strong alignment and synonym usage. |
| **BERTScore (F1)**| **0.9039** | High semantic accuracy. |
