#!/usr/bin/env python3
"""
Minimal Clarity Classification Script (No Large Model Downloads)

This script demonstrates clarity classification using a simple approach
that doesn't require downloading large pre-trained models.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
import pickle
import os
from pathlib import Path


class SimpleClarityClassifier:
    """Simple text classifier using TF-IDF + Logistic Regression."""

    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
        self.classifier = LogisticRegression(random_state=42, max_iter=1000)
        self.label_mapping = {
            0: "Clear Reply",
            1: "Clear Non-Reply",
            2: "Ambivalent Reply"
        }
        self.reverse_mapping = {v: k for k, v in self.label_mapping.items()}

    def encode_labels(self, clarity_labels):
        """Convert clarity labels to integers."""
        return [self.reverse_mapping[label] for label in clarity_labels]

    def decode_labels(self, predictions):
        """Convert integer predictions to clarity labels."""
        return [self.label_mapping[pred] for pred in predictions]

    def fit(self, texts, labels):
        """Train the classifier."""
        print("🔧 Training classifier...")

        # Convert labels to integers
        y = self.encode_labels(labels)

        # Create TF-IDF features
        X = self.vectorizer.fit_transform(texts)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Train classifier
        self.classifier.fit(X_train, y_train)

        # Evaluate
        y_pred = self.classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        print(".3f")
        print("\nClassification Report:")
        # Get unique classes in the test set
        unique_classes = sorted(set(y_test))
        target_names = [self.label_mapping[i] for i in unique_classes]
        print(classification_report(y_test, y_pred,
                                  target_names=target_names,
                                  labels=unique_classes))

        return accuracy

    def predict(self, texts):
        """Predict clarity labels for new texts."""
        X = self.vectorizer.transform(texts)
        predictions_int = self.classifier.predict(X)
        return self.decode_labels(predictions_int)

    def predict_proba(self, texts):
        """Predict with probabilities."""
        X = self.vectorizer.transform(texts)
        probabilities = self.classifier.predict_proba(X)
        predictions_int = np.argmax(probabilities, axis=1)
        predictions = self.decode_labels(predictions_int)

        return list(zip(predictions, probabilities.max(axis=1)))

    def save_model(self, filepath):
        """Save the trained model."""
        model_data = {
            'vectorizer': self.vectorizer,
            'classifier': self.classifier,
            'label_mapping': self.label_mapping,
            'reverse_mapping': self.reverse_mapping
        }

        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"💾 Model saved to {filepath}")

    def load_model(self, filepath):
        """Load a trained model."""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)

        self.vectorizer = model_data['vectorizer']
        self.classifier = model_data['classifier']
        self.label_mapping = model_data['label_mapping']
        self.reverse_mapping = model_data['reverse_mapping']
        print(f"📦 Model loaded from {filepath}")


def load_and_prepare_data():
    """Load QEvasion dataset and prepare for training."""
    print("📊 Loading QEvasion dataset...")

    dataset = load_dataset('ailsntua/QEvasion')
    print(f"Dataset splits: {list(dataset.keys())}")

    # Use training data
    train_data = dataset['train']
    print(f"Training examples: {len(train_data)}")

    # Extract texts and labels
    texts = []
    labels = []

    # Find the text field
    sample = train_data[0]
    text_field = 'interview_answer'  # Based on our earlier analysis

    print(f"Using text field: {text_field}")

    for item in train_data:
        text = str(item[text_field]) if item[text_field] else ""
        clarity_label = item['clarity_label']

        if text and clarity_label in ["Clear Reply", "Clear Non-Reply", "Ambivalent Reply"]:
            texts.append(text)
            labels.append(clarity_label)

    print(f"✅ Prepared {len(texts)} text-label pairs")

    # Show label distribution
    from collections import Counter
    label_counts = Counter(labels)
    print("Label distribution:")
    for label, count in label_counts.items():
        print(f"  {label}: {count}")

    return texts, labels


def main():
    """Main training function."""
    print("🎯 Simple Clarity Classification (No Large Models)")
    print("=" * 50)

    # Load and prepare data
    texts, labels = load_and_prepare_data()

    # Create and train classifier
    classifier = SimpleClarityClassifier()
    accuracy = classifier.fit(texts, labels)

    # Test predictions
    print("\n🧪 Testing predictions...")
    test_texts = [
        "Yes, I believe that is the correct approach.",
        "I'm not entirely sure about that.",
        "The answer to your question is clear."
    ]

    predictions = classifier.predict(test_texts)
    predictions_with_proba = classifier.predict_proba(test_texts)

    print("Sample Predictions:")
    for i, (text, pred, (pred_label, proba)) in enumerate(zip(test_texts, predictions, predictions_with_proba)):
        print(f"{i+1}. {text[:50]}...")
        print(f"   → {pred} (confidence: {proba:.3f})")

    # Save the model
    model_path = "clarity_classifier_simple.pkl"
    classifier.save_model(model_path)

    print("\n✅ Training completed!")
    print(".3f")
    print(f"🎯 Model saved as: {model_path}")
    print("\n💡 This model uses TF-IDF + Logistic Regression")
    print("   - No large transformer models required")
    print("   - Fast training and inference")
    print("   - Good baseline performance")

if __name__ == "__main__":
    main()