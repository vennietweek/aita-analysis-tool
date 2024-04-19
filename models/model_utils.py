import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
from sklearn.model_selection import KFold
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Embedding, LSTM, GRU, SimpleRNN, Conv1D, GlobalMaxPooling1D, MultiHeadAttention, Dropout, Flatten, Bidirectional
from tensorflow.keras.regularizers import l2
from transformers import BertTokenizer, BertModel
from keras import backend as K
from model_architectures import *

# Model utilities

def create_model(model_name, input_shape, vocab_size, embedding_dim):

    if model_name == 'RNN':
        model = RNNModel(input_shape=input_shape, vocab_size=vocab_size, embedding_dim=embedding_dim)
    elif model_name == 'RNN_Attention':
        model = RNNAttentionModel(input_shape=input_shape, vocab_size=vocab_size, embedding_dim=embedding_dim)
    elif model_name == 'CNN':
        model = CNNModel(input_shape=input_shape, vocab_size=vocab_size, embedding_dim=embedding_dim)
    elif model_name == 'CNN_Attention':
        model = CNNAttentionModel(input_shape=input_shape, vocab_size=vocab_size, embedding_dim=embedding_dim)
    elif model_name == 'Transformer':
        model = TransformerModel(input_shape=input_shape, vocab_size=vocab_size, embedding_dim=embedding_dim)
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def evaluate(y_true, y_pred_probs, threshold=0.5):
    y_pred_binary = (y_pred_probs >= threshold).astype(int)
    
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred_binary),
        'precision': precision_score(y_true, y_pred_binary),
        'recall': recall_score(y_true, y_pred_binary),
        'f1_score': f1_score(y_true, y_pred_binary),
        'roc_auc': roc_auc_score(y_true, y_pred_probs),
        'confusion_matrix': confusion_matrix(y_true, y_pred_binary)
    }
    
    return metrics

def print_metrics(metrics):
    print("\n=== Evaluation Metrics ===")
    for metric, value in metrics.items():
        if metric != 'confusion_matrix': 
            print(f"{metric.capitalize()}: {value:.4f}")

def plot_graphs(y_true, y_pred_probs, metrics):
    # ROC Curve
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_probs)
    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {metrics["roc_auc"]:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.show()
    
    # Confusion Matrix
    plt.figure(figsize=(5, 4))
    sns.heatmap(metrics['confusion_matrix'], annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.title('Confusion Matrix')
    plt.show()

def k_fold_cross_validation(model_name, X, y, input_shape, vocab_size, embedding_size, num_folds=3, num_epochs=10, batch_size=128):
    kfold = KFold(n_splits=num_folds, shuffle=True, random_state=42)
    
    # Initialize dictionaries to store scores for each metric across all folds
    aggregated_scores = {
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1_score': [],
        'roc_auc': []
    }
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(X)):
        print(f"\nTraining on fold {fold+1}/{num_folds}...")
        
        # Split data
        X_train_fold, X_val_fold = X[train_idx], X[val_idx]
        y_train_fold, y_val_fold = y[train_idx], y[val_idx]
        
        tf.keras.backend.clear_session()  
        model = create_model(model_name, input_shape, vocab_size, embedding_size)
        
        # Train the model
        model.fit(X_train_fold, y_train_fold, epochs=num_epochs, batch_size=batch_size,
                  validation_data=(X_val_fold, y_val_fold), verbose=1)
        
        # Generate predictions and evaluate
        y_pred_probs = model.predict(X_val_fold).ravel()
        fold_metrics = evaluate(y_val_fold, y_pred_probs)
        
        # Append the fold's metrics to the aggregated_scores
        for metric in aggregated_scores.keys():
            aggregated_scores[metric].append(fold_metrics[metric])
    
    # After all folds are processed, print aggregated scores
    print(f"Model name: {model_name}")
    print_aggregated_scores(aggregated_scores)

def print_aggregated_scores(aggregated_scores):
    print("\n=== Aggregate Scores Across All Folds ===")
    for metric, scores in aggregated_scores.items():
        average_score = np.mean(scores)
        print(f"{metric.capitalize()}: {average_score:.4f} (± {np.std(scores):.4f})")

def train_and_test(model_name, X_train, y_train, X_test, y_test, input_shape, vocab_size, embedding_dim, num_epochs=10, batch_size=128):
    # Clearing the TensorFlow session 
    tf.keras.backend.clear_session()

    # Creating the model using the provided model creation function
    model = create_model(model_name, input_shape, vocab_size, embedding_dim)

    # Training the model
    model.fit(X_train, y_train, epochs=num_epochs, batch_size=batch_size, verbose=1, validation_split=0.2)

    # Generate predictions
    y_pred_probs = model.predict(X_test).ravel()

    # Evaluate to get metrics
    metrics = evaluate(y_test, y_pred_probs)

    # Print the metrics
    print_metrics(metrics)

    # Plot the graphs
    plot_graphs(y_test, y_pred_probs, metrics)
    
def get_optimal_params(texts):

    # Initialize the tokenizer with a very high num_words to include all words
    tokenizer = Tokenizer(num_words=100000)
    tokenizer.fit_on_texts(texts)
    word_index = tokenizer.word_index
    sequences = tokenizer.texts_to_sequences(texts)
    sequence_lengths = [len(seq) for seq in sequences]
    
    # Calculate and print statistics about sequence lengths
    mean_length = np.mean(sequence_lengths)
    max_length = np.max(sequence_lengths)
    max_len = int(np.percentile(sequence_lengths, 95))
    vocab_size = len(word_index) + 1  # Including 0 index for padding
    
    print(f'Found {len(word_index)} unique tokens.')
    print(f"Mean sequence length: {mean_length}")
    print(f"Maximum sequence length: {max_length}")
    print(f"Chosen max_len based on 95th percentile: {max_len}")

    # Plot the distribution of sequence lengths
    plt.figure(figsize=(4, 3))
    sns.histplot(sequence_lengths, bins=50, kde=True)
    plt.title("Distribution of Sequence Lengths")
    plt.xlabel("Sequence Length")
    plt.ylabel("Frequency")
    plt.axvline(x=max_len, color='red', linestyle='--', label=f'95th Percentile: {max_len}')
    plt.legend()
    plt.show()
    
    return max_len, vocab_size

# Preprocessing for RNN using Keras tokeniser
def preprocess_sequences(train_texts, test_texts, num_words, maxlen=650):

    # Initialize and fit the tokenizer
    tokenizer = Tokenizer(num_words=num_words)
    tokenizer.fit_on_texts(train_texts)
    
    # Prepare training data
    train_sequences = tokenizer.texts_to_sequences(train_texts)
    train_sequences = pad_sequences(train_sequences, maxlen=maxlen)

    # Prepare testing data
    test_sequences = tokenizer.texts_to_sequences(test_texts)
    test_sequences = pad_sequences(test_sequences, maxlen=maxlen)
    
    return train_sequences, test_sequences

# Find optimal maximum sequence length for BERT

def get_optimal_params_bert(train_texts):
    # Load BERT tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Preliminary tokenization to find optimal max_len
    token_lens = []
    for txt in train_texts: 
        tokens = tokenizer.encode(txt)
        token_lens.append(len(tokens))

    # Analyze the distribution and choose max len based on a value that covers 95% of the texts
    print(f"Maximum sequence length: {max(token_lens)}")
    print(f"Average sequence length: {np.mean(token_lens)}")
    max_len_bert = np.quantile(token_lens, 0.95)
    print(f"Chosen max_len: {max_len_bert}")

    # Plot the distribution
    sns.histplot(token_lens, bins=50)
    plt.xlabel('Token count')
    plt.ylabel('Number of texts')
    plt.show()

    return max_len_bert

# Generate BERT embeddings
def preprocess_bert_embeddings(data, max_len):
    # Initialize the tokenizer and model
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')

    # Tokenize the text and convert to required format for BERT
    encoded_batch = tokenizer.batch_encode_plus(
        data.tolist(),
        add_special_tokens=True,
        max_length=max_len,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )

    input_ids = encoded_batch['input_ids']
    attention_mask = encoded_batch['attention_mask']

    model.eval()

    # Obtain embeddings
    with torch.no_grad():  # Disable gradient calculations for inference
        outputs = model(input_ids, attention_mask=attention_mask)
    embeddings = outputs.last_hidden_state
    
    # Convert embeddings to numpy array
    embeddings = embeddings.cpu().numpy() 

    return embeddings