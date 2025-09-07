# -*- coding: utf-8 -*-
"""
Deep Learning Models for Sentiment Analysis - MITNOP Project
Created for: Minja Knežić IN 25/2022

This module implements deep learning models (RNN, LSTM, GRU, Transformer) 
for sentiment analysis using the preprocessed data from classical ML models.

Based on research specification:
- RNN, LSTM, GRU models for sequence modeling
- Simple Transformer implementation
- Comparison with classical models
- Performance evaluation and analysis
"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Deep Learning Libraries
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Dense, Embedding, LSTM, GRU, SimpleRNN, Dropout, 
    GlobalMaxPooling1D, GlobalAveragePooling1D, Bidirectional,
    MultiHeadAttention, LayerNormalization, Input, Add
)
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

# Scikit-learn for evaluation
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_auc_score, roc_curve, precision_recall_curve
)

# Set random seeds for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)

# Configuration
DATA_PATH = "IMDB Dataset.csv"
OUTPUT_DIR = "./sentiment_deep_models_output"
CLASSICAL_MODELS_DIR = "./sentiment_student1_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Model hyperparameters
MAX_FEATURES = 10000  # Vocabulary size
MAX_LEN = 100         # Maximum sequence length
EMBEDDING_DIM = 128   # Embedding dimension
BATCH_SIZE = 64
EPOCHS = 5
VALIDATION_SPLIT = 0.2

class DeepSentimentAnalyzer:
    """
    Deep Learning Sentiment Analyzer with multiple architectures
    """
    
    def __init__(self, max_features=MAX_FEATURES, max_len=MAX_LEN, embedding_dim=EMBEDDING_DIM):
        self.max_features = max_features
        self.max_len = max_len
        self.embedding_dim = embedding_dim
        self.tokenizer = None
        self.models = {}
        self.histories = {}
        
    def load_and_preprocess_data(self):
        """Load and preprocess data using colleague's preprocessing"""
        print("Loading dataset...")
        df = pd.read_csv(DATA_PATH)
        
        # Use the same preprocessing as classical models
        from mitnop import preprocess_series
        
        print("Preprocessing text...")
        df['clean_review'] = preprocess_series(df['review'])
        
        # Convert labels
        y = df['sentiment'].map({'positive': 1, 'negative': 0}).values
        texts = df['clean_review'].values
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            texts, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
        )
        
        print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")
        
        # Tokenization
        print("Tokenizing texts...")
        self.tokenizer = Tokenizer(num_words=self.max_features, oov_token="<OOV>")
        self.tokenizer.fit_on_texts(X_train)
        
        # Convert to sequences
        X_train_seq = self.tokenizer.texts_to_sequences(X_train)
        X_test_seq = self.tokenizer.texts_to_sequences(X_test)
        
        # Pad sequences
        X_train_pad = pad_sequences(X_train_seq, maxlen=self.max_len)
        X_test_pad = pad_sequences(X_test_seq, maxlen=self.max_len)
        
        print(f"Vocabulary size: {len(self.tokenizer.word_index)}")
        print(f"Sequence shape: {X_train_pad.shape}")
        
        # Store for later use
        self.X_train, self.X_test = X_train_pad, X_test_pad
        self.y_train, self.y_test = y_train, y_test
        self.raw_texts_train, self.raw_texts_test = X_train, X_test
        
        return X_train_pad, X_test_pad, y_train, y_test
    
    def build_simple_rnn(self):
        """Build Simple RNN model"""
        model = Sequential([
            Embedding(self.max_features, self.embedding_dim, input_length=self.max_len),
            SimpleRNN(64, dropout=0.3, recurrent_dropout=0.3),
            Dense(32, activation='relu', kernel_regularizer=l2(0.01)),
            Dropout(0.5),
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def build_lstm(self, bidirectional=False):
        """Build LSTM model"""
        model = Sequential()
        model.add(Embedding(self.max_features, self.embedding_dim, input_length=self.max_len))
        
        if bidirectional:
            model.add(Bidirectional(LSTM(64, dropout=0.3, recurrent_dropout=0.3)))
        else:
            model.add(LSTM(64, dropout=0.3, recurrent_dropout=0.3))
            
        model.add(Dense(32, activation='relu', kernel_regularizer=l2(0.01)))
        model.add(Dropout(0.5))
        model.add(Dense(1, activation='sigmoid'))
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def build_gru(self, bidirectional=False):
        """Build GRU model"""
        model = Sequential()
        model.add(Embedding(self.max_features, self.embedding_dim, input_length=self.max_len))
        
        if bidirectional:
            model.add(Bidirectional(GRU(64, dropout=0.3, recurrent_dropout=0.3)))
        else:
            model.add(GRU(64, dropout=0.3, recurrent_dropout=0.3))
            
        model.add(Dense(32, activation='relu', kernel_regularizer=l2(0.01)))
        model.add(Dropout(0.5))
        model.add(Dense(1, activation='sigmoid'))
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def build_transformer_encoder(self):
        """Build simple Transformer encoder model"""
        # Input
        inputs = Input(shape=(self.max_len,))
        
        # Embedding
        embedding_layer = Embedding(self.max_features, self.embedding_dim)(inputs)
        
        # Multi-head attention
        attention_output = MultiHeadAttention(
            num_heads=8, key_dim=self.embedding_dim//8
        )(embedding_layer, embedding_layer)
        
        # Add & Norm
        attention_output = LayerNormalization()(Add()([embedding_layer, attention_output]))
        
        # Feed forward
        ff_output = Dense(128, activation='relu')(attention_output)
        ff_output = Dense(self.embedding_dim)(ff_output)
        
        # Add & Norm
        transformer_output = LayerNormalization()(Add()([attention_output, ff_output]))
        
        # Global pooling and classification
        pooled = GlobalAveragePooling1D()(transformer_output)
        dropout = Dropout(0.3)(pooled)
        dense = Dense(64, activation='relu', kernel_regularizer=l2(0.01))(dropout)
        dropout2 = Dropout(0.5)(dense)
        outputs = Dense(1, activation='sigmoid')(dropout2)
        
        model = Model(inputs=inputs, outputs=outputs)
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train_model(self, model, model_name, X_train, y_train):
        """Train a model with callbacks"""
        print(f"\nTraining {model_name}...")
        
        # Callbacks
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6),
            ModelCheckpoint(
                os.path.join(OUTPUT_DIR, f'{model_name}_best.h5'),
                monitor='val_accuracy',
                save_best_only=True,
                mode='max'
            )
        ]
        
        # Train
        history = model.fit(
            X_train, y_train,
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            validation_split=VALIDATION_SPLIT,
            callbacks=callbacks,
            verbose=1
        )
        
        self.models[model_name] = model
        self.histories[model_name] = history
        
        return model, history
    
    def evaluate_model(self, model, model_name, X_test, y_test):
        """Comprehensive model evaluation"""
        print(f"\n=== Evaluating {model_name} ===")
        
        # Predictions
        y_pred_proba = model.predict(X_test, verbose=0)
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)
        
        print(f"Accuracy: {accuracy:.4f}")
        print(f"ROC AUC: {auc:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['Negative', 'Positive']))
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Negative', 'Positive'],
                   yticklabels=['Negative', 'Positive'])
        plt.title(f'Confusion Matrix - {model_name}')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f'confusion_matrix_{model_name}.png'))
        plt.show()
        
        # ROC Curve
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        plt.figure(figsize=(6, 4))
        plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc:.4f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {model_name}')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f'roc_curve_{model_name}.png'))
        plt.show()
        
        return {
            'model_name': model_name,
            'accuracy': accuracy,
            'auc': auc,
            'predictions': y_pred_proba.flatten(),
            'true_labels': y_test
        }
    
    def plot_training_history(self, model_name):
        """Plot training history"""
        if model_name not in self.histories:
            print(f"No history found for {model_name}")
            return
            
        history = self.histories[model_name]
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Accuracy
        axes[0].plot(history.history['accuracy'], label='Training Accuracy')
        axes[0].plot(history.history['val_accuracy'], label='Validation Accuracy')
        axes[0].set_title(f'{model_name} - Accuracy')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Accuracy')
        axes[0].legend()
        axes[0].grid(True)
        
        # Loss
        axes[1].plot(history.history['loss'], label='Training Loss')
        axes[1].plot(history.history['val_loss'], label='Validation Loss')
        axes[1].set_title(f'{model_name} - Loss')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f'training_history_{model_name}.png'))
        plt.show()
    
    def run_complete_analysis(self):
        """Run complete deep learning analysis"""
        print("=" * 60)
        print("DEEP LEARNING SENTIMENT ANALYSIS")
        print("=" * 60)
        
        # Load and preprocess data
        X_train, X_test, y_train, y_test = self.load_and_preprocess_data()
        
        # Define models to train
        model_configs = [
            ('SimpleRNN', self.build_simple_rnn),
            ('LSTM', lambda: self.build_lstm(bidirectional=False)),
            ('BiLSTM', lambda: self.build_lstm(bidirectional=True)),
            ('GRU', lambda: self.build_gru(bidirectional=False)),
            ('BiGRU', lambda: self.build_gru(bidirectional=True)),
            ('Transformer', self.build_transformer_encoder)
        ]
        
        # Train and evaluate all models
        results = []
        
        for model_name, model_builder in model_configs:
            print(f"\n{'-'*40}")
            print(f"Processing {model_name}")
            print(f"{'-'*40}")
            
            # Build model
            model = model_builder()
            print(f"Model {model_name} architecture:")
            model.summary()
            
            # Train model
            trained_model, history = self.train_model(model, model_name, X_train, y_train)
            
            # Plot training history
            self.plot_training_history(model_name)
            
            # Evaluate model
            result = self.evaluate_model(trained_model, model_name, X_test, y_test)
            results.append(result)
            
            # Save model
            model.save(os.path.join(OUTPUT_DIR, f'{model_name}_model.h5'))
            print(f"Model {model_name} saved successfully!")
        
        # Compare all models
        self.compare_models(results)
        
        # Load and compare with classical models
        self.compare_with_classical_models(results)
        
        return results
    
    def compare_models(self, results):
        """Compare deep learning models"""
        print("\n" + "="*60)
        print("DEEP LEARNING MODELS COMPARISON")
        print("="*60)
        
        # Create comparison DataFrame
        comparison_df = pd.DataFrame([
            {
                'Model': result['model_name'],
                'Accuracy': result['accuracy'],
                'ROC AUC': result['auc']
            }
            for result in results
        ])
        
        comparison_df = comparison_df.sort_values('ROC AUC', ascending=False)
        print(comparison_df.to_string(index=False))
        
        # Save results
        comparison_df.to_csv(os.path.join(OUTPUT_DIR, 'deep_models_comparison.csv'), index=False)
        
        # Plot comparison
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Accuracy comparison
        axes[0].bar(comparison_df['Model'], comparison_df['Accuracy'], color='skyblue')
        axes[0].set_title('Model Accuracy Comparison')
        axes[0].set_ylabel('Accuracy')
        axes[0].set_ylim(0.8, 1.0)
        axes[0].tick_params(axis='x', rotation=45)
        
        # AUC comparison
        axes[1].bar(comparison_df['Model'], comparison_df['ROC AUC'], color='lightcoral')
        axes[1].set_title('Model ROC AUC Comparison')
        axes[1].set_ylabel('ROC AUC')
        axes[1].set_ylim(0.8, 1.0)
        axes[1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'deep_models_comparison.png'))
        plt.show()
    
    def compare_with_classical_models(self, deep_results):
        """Compare deep learning models with classical ML models"""
        print("\n" + "="*60)
        print("DEEP VS CLASSICAL MODELS COMPARISON")
        print("="*60)
        
        # Load classical model results
        try:
            classical_results = pd.read_csv(os.path.join(CLASSICAL_MODELS_DIR, 'model_summary.csv'))
            print("Classical Model Results:")
            print(classical_results.to_string(index=False))
            
            # Combine results
            deep_df = pd.DataFrame([
                {
                    'model': result['model_name'],
                    'accuracy': result['accuracy'],
                    'roc_auc': result['auc'],
                    'type': 'Deep Learning'
                }
                for result in deep_results
            ])
            
            classical_df = classical_results.copy()
            classical_df['type'] = 'Classical ML'
            
            # Combine and plot
            combined_df = pd.concat([
                classical_df[['model', 'accuracy', 'roc_auc', 'type']],
                deep_df
            ], ignore_index=True)
            
            # Plot comparison
            fig, axes = plt.subplots(1, 2, figsize=(16, 6))
            
            # Accuracy comparison
            classical_acc = classical_df.groupby('type')['accuracy'].mean().values[0]
            deep_acc = deep_df.groupby('type')['accuracy'].mean().values[0]
            
            axes[0].bar(['Classical ML', 'Deep Learning'], [classical_acc, deep_acc], 
                       color=['lightblue', 'lightgreen'])
            axes[0].set_title('Average Accuracy: Classical vs Deep Learning')
            axes[0].set_ylabel('Accuracy')
            axes[0].set_ylim(0.8, 1.0)
            
            # Individual model comparison
            sns.boxplot(data=combined_df, x='type', y='roc_auc', ax=axes[1])
            axes[1].set_title('ROC AUC Distribution: Classical vs Deep Learning')
            axes[1].set_ylabel('ROC AUC')
            
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR, 'classical_vs_deep_comparison.png'))
            plt.show()
            
            # Save combined results
            combined_df.to_csv(os.path.join(OUTPUT_DIR, 'all_models_comparison.csv'), index=False)
            
        except FileNotFoundError:
            print("Classical model results not found. Skipping comparison.")
    
    def analyze_model_errors(self, model_name='BiLSTM'):
        """Analyze errors of the best performing model"""
        if model_name not in self.models:
            print(f"Model {model_name} not found!")
            return
            
        print(f"\n=== Error Analysis for {model_name} ===")
        
        model = self.models[model_name]
        
        # Get predictions
        y_pred_proba = model.predict(self.X_test, verbose=0)
        y_pred = (y_pred_proba > 0.5).astype(int).flatten()
        
        # Find misclassified examples
        misclassified_mask = y_pred != self.y_test
        misclassified_indices = np.where(misclassified_mask)[0]
        
        print(f"Total misclassified: {len(misclassified_indices)} out of {len(self.y_test)}")
        
        # Analyze misclassified examples
        misclassified_data = []
        for idx in misclassified_indices[:100]:  # Analyze first 100
            original_idx = idx
            text = self.raw_texts_test[original_idx]
            true_label = self.y_test[original_idx]
            pred_label = y_pred[original_idx]
            confidence = y_pred_proba[original_idx][0]
            
            misclassified_data.append({
                'text': text[:200] + '...' if len(text) > 200 else text,
                'true_sentiment': 'Positive' if true_label == 1 else 'Negative',
                'predicted_sentiment': 'Positive' if pred_label == 1 else 'Negative',
                'confidence': confidence,
                'text_length': len(text.split())
            })
        
        # Save misclassified examples
        misclassified_df = pd.DataFrame(misclassified_data)
        misclassified_df.to_csv(os.path.join(OUTPUT_DIR, f'misclassified_{model_name}.csv'), index=False)
        
        print("\nSample misclassified examples:")
        print(misclassified_df.head(5).to_string(index=False))


def main():
    """Main function to run the complete analysis"""
    print("Starting Deep Learning Sentiment Analysis...")
    print("Author: Minja Knežić IN 25/2022")
    print("Project: MITNOP - Sentiment Analysis with Classical and Deep Models")
    
    # Initialize analyzer
    analyzer = DeepSentimentAnalyzer()
    
    # Run complete analysis
    results = analyzer.run_complete_analysis()
    
    # Additional analysis
    print("\n" + "="*60)
    print("ADDITIONAL ANALYSIS")
    print("="*60)
    
    # Error analysis for best model
    analyzer.analyze_model_errors('BiLSTM')
    
    print(f"\nAnalysis complete! Results saved in: {OUTPUT_DIR}")
    print("Files created:")
    for file in os.listdir(OUTPUT_DIR):
        print(f"  - {file}")


if __name__ == "__main__":
    main()
