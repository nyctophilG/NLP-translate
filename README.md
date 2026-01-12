# NLP-translate
an NLP project that translates english to spanish using 6 models with 120K datas db https://www.kaggle.com/datasets/tejasurya/eng-spanish

Overview

This project implements an attention-based Seq2Seq model to translate English source text into Spanish. It serves as the final capstone for the semester's NLP coursework, demonstrating proficiency in handling distinct linguistic morphologies.

Technical Specifications

    Architecture: Bi-directional GRU Encoder with an Additive Attention Decoder.

    Vocabulary: Built using spacy (en_core_web_sm and es_core_news_sm) with a frequency threshold of 2.

    Embeddings: Pre-trained FastText vectors (English and Spanish) to handle semantic nuances.

    Model Variants & Experimental Setup

To analyze the efficacy of different sequence processing techniques, we implemented and evaluated 6 distinct architectures:

    Baseline RNN: A simple unidirectional Recurrent Neural Network to establish a performance floor.

    Standard LSTM: Long Short-Term Memory network to handle vanishing gradients in longer sentences.

    Gated Recurrent Unit (GRU): A more efficient gated mechanism to compare training speed against LSTM.

    Bi-Directional GRU + Bahdanau Attention: Adds additive attention to handle alignment and context from both directions.

    ConvS2S (Convolutional Seq2Seq): Utilizes 1D convolutions for parallelized training, following Facebook AI Research papers.

    Transformer: The "Attention Is All You Need" architecture, relying entirely on self-attention mechanisms for global dependency modeling.
