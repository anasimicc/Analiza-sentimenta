import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Import configurations
from model_configs import DEFAULT_CONFIG
from text_preprocessing import preprocess_series

# Deep Learning Libraries
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Dense, Embedding, LSTM, GRU, SimpleRNN, Dropout, 
    GlobalMaxPooling1D, GlobalAveragePooling1D, Bidirectional,
    MultiHeadAttention, LayerNormalization, Input, Add, Layer
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


class DeepSentimentAnalyzer:
    """
    Deep Learning Sentiment Analyzer with multiple architectures
    """
    
    def __init__(self, config=None, experiment_name="default"):
        # Load configuration
        self.config = config if config is not None else DEFAULT_CONFIG.copy()
        self.experiment_name = experiment_name
        
        # Create experiment-specific output directory
        base_output_dir = self.config['OUTPUT_DIR']
        self.output_dir = os.path.join(base_output_dir, experiment_name)
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Model parameters
        self.max_features = self.config['MAX_FEATURES']
        self.max_len = self.config['MAX_LEN']
        self.embedding_dim = self.config['EMBEDDING_DIM']
        
        # Initialize
        self.tokenizer = None
        self.models = {}
        self.histories = {}
        
        # GloVe embeddings support
        self.use_glove = self.config.get('USE_GLOVE', False)
        self.glove_path = self.config.get('GLOVE_PATH', None)
        self.embedding_matrix = None
        
        # Debug GloVe configuration
        print(f"GloVe Configuration:")
        print(f"USE_GLOVE: {self.use_glove}")
        print(f"GLOVE_PATH: {self.glove_path}")
        if self.use_glove and self.glove_path:
            print(f"File exists: {os.path.exists(self.glove_path)}")
        
        # Set random seeds
        np.random.seed(self.config['RANDOM_STATE'])
        tf.random.set_seed(self.config['RANDOM_STATE'])
        
        # Save configuration for tracking
        self.save_config()
    
    def save_config(self):
        """Save experiment configuration to file for tracking"""
        import json
        from datetime import datetime
        
        # Create config info with metadata
        config_info = {
            'experiment_name': self.experiment_name,
            'timestamp': datetime.now().isoformat(),
            'configuration': self.config.copy()
        }
        
        # Save as JSON
        config_file = os.path.join(self.output_dir, 'experiment_config.json')
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config_info, f, indent=2)
        
        # Also save as readable text
        config_txt_file = os.path.join(self.output_dir, 'experiment_config.txt')
        with open(config_txt_file, 'w', encoding='utf-8') as f:
            f.write(f"EXPERIMENT CONFIGURATION\n")
            f.write(f"========================\n\n")
            f.write(f"Experiment Name: {self.experiment_name}\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"PARAMETERS:\n")
            f.write(f"-----------\n")
            for key, value in self.config.items():
                f.write(f"{key:20}: {value}\n")
            f.write(f"\nDERIVED SETTINGS:\n")
            f.write(f"-----------------\n")
            f.write(f"Output Directory    : {self.output_dir}\n")
            f.write(f"Max Features        : {self.max_features}\n")
            f.write(f"Max Sequence Length : {self.max_len}\n")
            f.write(f"Embedding Dimension : {self.embedding_dim}\n")
        
        print(f"Configuration saved to:")
        print(f"- {config_file}")
        print(f"- {config_txt_file}")
    
    def load_glove_embeddings(self, glove_path):
        """Load GloVe embeddings from file"""
        print(f"Loading GloVe embeddings from {glove_path}...")
        
        # Check if file exists
        if not os.path.exists(glove_path):
            print(f"ERROR: GloVe file not found: {glove_path}")
            return None
        
        embeddings_index = {}
        skipped_lines = 0
        
        def convert_fraction_to_float(s):
            """Convert fractions like '1/4' to float"""
            try:
                if '/' in s and '-' not in s:  # Handle fractions but not ranges like '820-2036'
                    parts = s.split('/')
                    if len(parts) == 2:
                        return float(parts[0]) / float(parts[1])
                return float(s)
            except:
                return None
        
        try:
            with open(glove_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        values = line.strip().split()
                        if len(values) < 2:  # Skip empty or malformed lines
                            continue
                        
                        word = values[0]
                        coef_strings = values[1:]
                        
                        # Try to convert all coefficients, handling fractions and long decimals
                        coefs = []
                        all_valid = True
                        
                        for coef_str in coef_strings:
                            coef = convert_fraction_to_float(coef_str)
                            if coef is not None:
                                coefs.append(coef)
                            else:
                                all_valid = False
                                # Show first few problematic values for debugging
                                if skipped_lines < 5:
                                    print(f"Debug: Failed to parse '{coef_str}' in word '{word}' at line {line_num}")
                                break
                        
                        if all_valid and len(coefs) > 0:
                            embeddings_index[word] = np.asarray(coefs, dtype='float32')
                        else:
                            skipped_lines += 1
                        
                        # Show progress for large files  
                        if line_num % 50000 == 0:
                            print(f"Processed {line_num:,} lines... (loaded {len(embeddings_index)}, skipped {skipped_lines})")
                            
                    except Exception as e:
                        skipped_lines += 1
                        if skipped_lines <= 5:  # Only show first 5 errors for debugging
                            print(f"Debug: Exception at line {line_num}: {e}")
                        continue
            
            print(f"Found {len(embeddings_index)} word vectors in custom embedding file.")
            print(f"Skipped {skipped_lines} malformed lines")
            
            # Show sample of loaded embeddings for verification
            if len(embeddings_index) > 0:
                sample_word = list(embeddings_index.keys())[0]
                sample_vector = embeddings_index[sample_word]
                print(f"🔍 Sample: '{sample_word}' -> vector of {len(sample_vector)} dimensions")
            
            # Check if we loaded any embeddings
            if len(embeddings_index) == 0:
                print(f"ERROR: No valid embeddings found in {glove_path}")
                return None
                
            return embeddings_index
            
        except Exception as e:
            print(f"ERROR: Failed to load GloVe embeddings: {e}")
            return None
    
    def create_embedding_matrix(self, embeddings_index):
        """Create embedding matrix from GloVe embeddings"""
        if self.tokenizer is None:
            raise ValueError("Tokenizer must be fitted before creating embedding matrix")
        
        # Get the actual embedding dimension from GloVe
        sample_embedding = next(iter(embeddings_index.values()))
        glove_dim = len(sample_embedding)
        
        print(f"GloVe embedding dimension: {glove_dim}")
        print(f"Config embedding dimension: {self.embedding_dim}")
        
        # Use GloVe dimension
        embedding_dim = glove_dim
        
        # Create embedding matrix
        embedding_matrix = np.zeros((self.max_features, embedding_dim))
        
        hits = 0
        misses = 0
        
        for word, i in self.tokenizer.word_index.items():
            if i >= self.max_features:
                continue
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
                hits += 1
            else:
                # Initialize missing words with random values
                embedding_matrix[i] = np.random.normal(scale=0.6, size=(embedding_dim,))
                misses += 1
        
        print(f"Embedding matrix created:")
        print(f"  - Vocabulary hits: {hits}")
        print(f"  - Vocabulary misses: {misses}")
        print(f"  - Coverage: {hits/(hits+misses)*100:.1f}%")
        
        # Update embedding dimension to match GloVe
        self.embedding_dim = embedding_dim
        
        return embedding_matrix
    
    def save_model_config(self, model, model_name):
        """Save model-specific configuration and architecture"""
        import json
        from datetime import datetime
        
        # Get model summary as string
        import io
        import sys
        
        # Capture model summary (model should be built after training)
        old_stdout = sys.stdout
        sys.stdout = buffer = io.StringIO()
        model.summary()
        model_summary = buffer.getvalue()
        sys.stdout = old_stdout
        
        # Count parameters (model is built after training)
        total_params = model.count_params()
        trainable_params = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
        non_trainable_params = total_params - trainable_params
        
        # Create model-specific config
        model_config = {
            'model_name': model_name,
            'timestamp': datetime.now().isoformat(),
            'total_parameters': int(total_params),
            'trainable_parameters': int(trainable_params),
            'non_trainable_parameters': int(non_trainable_params),
            'model_config': model.get_config(),
            'experiment_config': self.config.copy()
        }
        
        # Save model config as JSON
        model_config_file = os.path.join(self.output_dir, f'{model_name}_config.json')
        with open(model_config_file, 'w', encoding='utf-8') as f:
            json.dump(model_config, f, indent=2, default=str)
        
        # Save model architecture as text
        model_arch_file = os.path.join(self.output_dir, f'{model_name}_architecture.txt')
        with open(model_arch_file, 'w', encoding='utf-8') as f:
            f.write(f"MODEL ARCHITECTURE: {model_name}\n")
            f.write(f"{'='*50}\n\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Experiment: {self.experiment_name}\n\n")
            f.write(f"PARAMETER SUMMARY:\n")
            f.write(f"------------------\n")
            f.write(f"Total Parameters      : {total_params:,}\n")
            f.write(f"Trainable Parameters  : {trainable_params:,}\n")
            f.write(f"Non-trainable Parameters: {non_trainable_params:,}\n\n")
            f.write(f"TRAINING CONFIGURATION:\n")
            f.write(f"-----------------------\n")
            f.write(f"Vocabulary Size       : {self.config['MAX_FEATURES']}\n")
            f.write(f"Sequence Length       : {self.config['MAX_LEN']}\n")
            f.write(f"Embedding Dimension   : {self.config['EMBEDDING_DIM']}\n")
            f.write(f"Batch Size           : {self.config['BATCH_SIZE']}\n")
            f.write(f"Epochs               : {self.config['EPOCHS']}\n")
            f.write(f"Validation Split     : {self.config['VALIDATION_SPLIT']}\n\n")
            f.write(f"MODEL ARCHITECTURE:\n")
            f.write(f"-------------------\n")
            f.write(model_summary)
        
        print(f"Model config saved:")
        print(f"- {model_config_file}")
        print(f"- {model_arch_file}")

    def save_model_metrics(self, model_name, y_test, y_pred, y_pred_proba, accuracy, auc):
        """Save detailed model metrics to text files"""
        from datetime import datetime
        from sklearn.metrics import (
            precision_score, recall_score, f1_score, 
            classification_report, confusion_matrix,
            precision_recall_curve, average_precision_score
        )
        import numpy as np
        
        # Calculate additional metrics
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        avg_precision = average_precision_score(y_test, y_pred_proba)
        
        # Get confusion matrix values
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        # Calculate additional metrics
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0  # Positive Predictive Value
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0  # Negative Predictive Value
        
        # Get classification report as string
        class_report = classification_report(y_test, y_pred, target_names=['Negative', 'Positive'])
        
        # Save comprehensive metrics
        metrics_file = os.path.join(self.output_dir, f'{model_name}_metrics.txt')
        with open(metrics_file, 'w', encoding='utf-8') as f:
            f.write(f"DETAILED METRICS: {model_name}\n")
            f.write(f"{'='*50}\n\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Experiment: {self.experiment_name}\n\n")
            
            f.write(f"OVERALL PERFORMANCE:\n")
            f.write(f"--------------------\n")
            f.write(f"Accuracy             : {accuracy:.6f}\n")
            f.write(f"ROC AUC              : {auc:.6f}\n")
            f.write(f"Average Precision    : {avg_precision:.6f}\n\n")
            
            f.write(f"CLASSIFICATION METRICS:\n")
            f.write(f"-----------------------\n")
            f.write(f"Precision            : {precision:.6f}\n")
            f.write(f"Recall (Sensitivity) : {recall:.6f}\n")
            f.write(f"F1-Score             : {f1:.6f}\n")
            f.write(f"Specificity          : {specificity:.6f}\n\n")
            
            f.write(f"CONFUSION MATRIX VALUES:\n")
            f.write(f"------------------------\n")
            f.write(f"True Negatives  (TN) : {tn}\n")
            f.write(f"False Positives (FP) : {fp}\n")
            f.write(f"False Negatives (FN) : {fn}\n")
            f.write(f"True Positives  (TP) : {tp}\n\n")
            
            f.write(f"CONFUSION MATRIX:\n")
            f.write(f"-----------------\n")
            f.write(f"                Predicted\n")
            f.write(f"Actual       Neg    Pos\n")
            f.write(f"   Neg      {tn:4d}   {fp:4d}\n")
            f.write(f"   Pos      {fn:4d}   {tp:4d}\n\n")
            
            f.write(f"PREDICTIVE VALUES:\n")
            f.write(f"------------------\n")
            f.write(f"Positive Predictive Value (PPV): {ppv:.6f}\n")
            f.write(f"Negative Predictive Value (NPV): {npv:.6f}\n\n")
            
            f.write(f"ERROR RATES:\n")
            f.write(f"------------\n")
            f.write(f"False Positive Rate  : {fp/(fp+tn):.6f}\n")
            f.write(f"False Negative Rate  : {fn/(fn+tp):.6f}\n")
            f.write(f"False Discovery Rate : {fp/(fp+tp):.6f}\n\n")
            
            f.write(f"DETAILED CLASSIFICATION REPORT:\n")
            f.write(f"--------------------------------\n")
            f.write(class_report)
            f.write(f"\n")
            
            f.write(f"PREDICTION STATISTICS:\n")
            f.write(f"----------------------\n")
            f.write(f"Total Predictions    : {len(y_test)}\n")
            f.write(f"Correct Predictions  : {np.sum(y_test == y_pred)}\n")
            f.write(f"Incorrect Predictions: {np.sum(y_test != y_pred)}\n\n")
            
            f.write(f"CONFIDENCE STATISTICS:\n")
            f.write(f"----------------------\n")
            f.write(f"Mean Confidence      : {np.mean(y_pred_proba):.6f}\n")
            f.write(f"Std Confidence       : {np.std(y_pred_proba):.6f}\n")
            f.write(f"Min Confidence       : {np.min(y_pred_proba):.6f}\n")
            f.write(f"Max Confidence       : {np.max(y_pred_proba):.6f}\n")
            f.write(f"Median Confidence    : {np.median(y_pred_proba):.6f}\n\n")
            
            # Confidence distribution
            f.write(f"CONFIDENCE DISTRIBUTION:\n")
            f.write(f"------------------------\n")
            confidence_ranges = [(0.0, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.0)]
            for low, high in confidence_ranges:
                count = np.sum((y_pred_proba >= low) & (y_pred_proba < high))
                percentage = count / len(y_pred_proba) * 100
                f.write(f"{low:.1f}-{high:.1f}: {count:5d} predictions ({percentage:5.1f}%)\n")
        
        # Save metrics as CSV for easy analysis
        metrics_csv_file = os.path.join(self.output_dir, f'{model_name}_metrics.csv')
        metrics_data = {
            'metric': ['accuracy', 'roc_auc', 'precision', 'recall', 'f1_score', 'specificity', 
                      'avg_precision', 'ppv', 'npv', 'true_negatives', 'false_positives', 
                      'false_negatives', 'true_positives'],
            'value': [accuracy, auc, precision, recall, f1, specificity, avg_precision, 
                     ppv, npv, tn, fp, fn, tp]
        }
        
        metrics_df = pd.DataFrame(metrics_data)
        metrics_df.to_csv(metrics_csv_file, index=False)

        print(f"Detailed metrics saved:")
        print(f"- {metrics_file}")
        print(f"- {metrics_csv_file}")

    def load_and_preprocess_data(self):
        """Load and preprocess data using colleague's preprocessing"""
        print("Loading dataset...")
        df = pd.read_csv(self.config['DATA_PATH'])
        
        print("Preprocessing text...")
        df['clean_review'] = preprocess_series(df['review'])
        
        # Convert labels
        y = df['sentiment'].map({'positive': 1, 'negative': 0}).values
        texts = df['clean_review'].values
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            texts, y, test_size=0.2, random_state=self.config['RANDOM_STATE'], stratify=y
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
        
        # Load GloVe embeddings if specified
        if self.use_glove and self.glove_path:
            print(f"Attempting to load GloVe embeddings...")
            embeddings_index = self.load_glove_embeddings(self.glove_path)
            if embeddings_index is not None:
                self.embedding_matrix = self.create_embedding_matrix(embeddings_index)
                if self.embedding_matrix is not None:
                    print(f"GloVe embeddings loaded and embedding matrix created")
                else:
                    print(f"Failed to create embedding matrix, falling back to random initialization")
                    self.embedding_matrix = None
            else:
                print(f"Failed to load GloVe embeddings, falling back to random initialization")
                self.embedding_matrix = None
        else:
            if self.use_glove:
                print("USE_GLOVE is True but no GLOVE_PATH specified")
            print("Using random embedding initialization")
        
        # Store for later use
        self.X_train, self.X_test = X_train_pad, X_test_pad
        self.y_train, self.y_test = y_train, y_test
        self.raw_texts_train, self.raw_texts_test = X_train, X_test
        
        return X_train_pad, X_test_pad, y_train, y_test
    
    def build_simple_rnn(self):
        """Build Simple RNN model"""
        model = Sequential([
            Embedding(self.max_features, self.embedding_dim, input_length=self.max_len),
            SimpleRNN(128, return_sequences=True, dropout=0.2, recurrent_dropout=0.2),
            GlobalMaxPooling1D(),
            Dense(64, activation='relu'),
            Dropout(0.3),
            Dense(32, activation='relu'),
            Dropout(0.3),
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
        
        # Embedding with optional GloVe weights
        if self.embedding_matrix is not None:
            embedding_layer = Embedding(
                self.max_features, 
                self.embedding_dim, 
                weights=[self.embedding_matrix],
                trainable=True
            )(inputs)
            print("Using GloVe pre-trained embeddings for Transformer")
        else:
            embedding_layer = Embedding(self.max_features, self.embedding_dim)(inputs)
            print("Using random embedding initialization for Transformer")
        
        # Multi-head attention
        attention_output = MultiHeadAttention(
            num_heads=4, key_dim=self.embedding_dim//4, dropout=0.1
        )(embedding_layer, embedding_layer)
        
        # Add & Norm
        attention_output = LayerNormalization()(Add()([embedding_layer, attention_output]))
        
        # Feed forward
        ff_output = Dense(64, activation='relu')(attention_output)
        ff_output = Dropout(0.2)(ff_output)
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
            optimizer=Adam(learning_rate=0.0001),
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
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=1, min_lr=1e-6),
            ModelCheckpoint(
                os.path.join(self.output_dir, f'{model_name}_best.h5'),
                monitor='val_accuracy',
                save_best_only=True,
                mode='max'
            )
        ]
        
        # Train
        history = model.fit(
            X_train, y_train,
            batch_size=self.config['BATCH_SIZE'],
            epochs=self.config['EPOCHS'],
            validation_split=self.config['VALIDATION_SPLIT'],
            callbacks=callbacks,
            verbose=1
        )
        
        # Save model-specific configuration after training (when model is built)
        self.save_model_config(model, model_name)
        
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
        
        # Save detailed metrics to file
        self.save_model_metrics(model_name, y_test, y_pred, y_pred_proba, accuracy, auc)
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Negative', 'Positive'],
                   yticklabels=['Negative', 'Positive'])
        plt.title(f'Confusion Matrix - {model_name}')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f'confusion_matrix_{model_name}.png'))
        #plt.show()
        
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
        plt.savefig(os.path.join(self.output_dir, f'roc_curve_{model_name}.png'))
        #plt.show()
        
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
        plt.savefig(os.path.join(self.output_dir, f'training_history_{model_name}.png'))
        #plt.show()
    
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
            model.save(os.path.join(self.output_dir, f'{model_name}_model.h5'))
            print(f"Model {model_name} saved successfully!")
        
        # Compare all models
        self.compare_models(results)
        
        # Load and compare with classical models
        self.compare_with_classical_models(results)
        
        # Create experiment summary
        self.save_experiment_summary(results)
        
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
        comparison_df.to_csv(os.path.join(self.output_dir, 'deep_models_comparison.csv'), index=False)
        
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
        plt.savefig(os.path.join(self.output_dir, 'deep_models_comparison.png'))
        #plt.show()
    
    def compare_with_classical_models(self, deep_results):
        """Compare deep learning models with classical ML models"""
        print("\n" + "="*60)
        print("DEEP VS CLASSICAL MODELS COMPARISON")
        print("="*60)
        
        # Load classical model results
        try:
            classical_results = pd.read_csv(os.path.join(self.config['CLASSICAL_MODELS_DIR'], 'model_summary.csv'))
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
            plt.savefig(os.path.join(self.output_dir, 'classical_vs_deep_comparison.png'))
            #plt.show()
            
            # Save combined results
            combined_df.to_csv(os.path.join(self.output_dir, 'all_models_comparison.csv'), index=False)
            
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
        misclassified_df.to_csv(os.path.join(self.output_dir, f'misclassified_{model_name}.csv'), index=False)
        
        print("\nSample misclassified examples:")
        print(misclassified_df.head(5).to_string(index=False))
    
    def save_experiment_summary(self, results):
        """Save a comprehensive experiment summary"""
        from datetime import datetime
        import json
        
        # Create summary
        summary = {
            'experiment_name': self.experiment_name,
            'timestamp': datetime.now().isoformat(),
            'configuration': self.config.copy(),
            'models_trained': len(results),
            'results': []
        }
        
        for result in results:
            summary['results'].append({
                'model_name': result['model_name'],
                'accuracy': float(result['accuracy']),
                'roc_auc': float(result['auc'])
            })
        
        # Sort by performance
        summary['results'].sort(key=lambda x: x['roc_auc'], reverse=True)
        
        # Save JSON summary
        summary_file = os.path.join(self.output_dir, 'experiment_summary.json')
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)
        
        # Save readable summary
        summary_txt_file = os.path.join(self.output_dir, 'experiment_summary.txt')
        with open(summary_txt_file, 'w', encoding='utf-8') as f:
            f.write(f"EXPERIMENT SUMMARY\n")
            f.write(f"==================\n\n")
            f.write(f"Experiment: {self.experiment_name}\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Models Trained: {len(results)}\n\n")
            
            f.write(f"CONFIGURATION:\n")
            f.write(f"--------------\n")
            for key, value in self.config.items():
                f.write(f"{key:20}: {value}\n")
            f.write(f"\n")
            
            f.write(f"RESULTS (sorted by ROC AUC):\n")
            f.write(f"-----------------------------\n")
            f.write(f"{'Model':<15} {'Accuracy':<10} {'ROC AUC':<10}\n")
            f.write(f"{'-'*35}\n")
            
            for result in summary['results']:
                f.write(f"{result['model_name']:<15} {result['accuracy']:<10.4f} {result['roc_auc']:<10.4f}\n")
            
            if results:
                best_model = summary['results'][0]
                f.write(f"\nBEST MODEL: {best_model['model_name']}\n")
                f.write(f"Accuracy: {best_model['accuracy']:.4f}\n")
                f.write(f"ROC AUC: {best_model['roc_auc']:.4f}\n")
            
            f.write(f"\nFILES GENERATED:\n")
            f.write(f"----------------\n")
            try:
                files = sorted([f for f in os.listdir(self.output_dir) if f != 'experiment_summary.txt'])
                for file in files:
                    f.write(f"- {file}\n")
            except:
                f.write("- (file list not available)\n")
        
        print(f"\nExperiment summary saved:")
        print(f"- {summary_file}")
        print(f"- {summary_txt_file}")

        return summary
    
