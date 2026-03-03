#!/usr/bin/env python3
"""
Generate CLARITY-SemEval-2026 Submission

This script loads the CLARITY evaluation dataset, uses our trained clarity classifier
to make predictions, and formats the results for CodeBench submission.
"""

import pickle
import pandas as pd
from clarity_classifier_minimal import SimpleClarityClassifier


def load_test_data():
    """Load the CLARITY test dataset."""
    print("📊 Loading CLARITY test dataset...")

    # Load the evaluation dataset
    eval_df = pd.read_csv("/Users/andrearachetta/Desktop/CLARITY-SemEval-2026/dataset/clarity_task_evaluation_dataset.csv")

    print(f"✅ Loaded {len(eval_df)} evaluation samples")

    # Extract the text field (interview_answer) and indices
    texts = []
    indices = []

    for idx, row in eval_df.iterrows():
        # Use interview_answer as the text to classify
        text = str(row.get('interview_answer', '')).strip()
        if text:  # Only include non-empty texts
            texts.append(text)
            indices.append(idx)

    print(f"✅ Prepared {len(texts)} texts for classification")
    return texts, indices


def load_trained_model():
    """Load our trained clarity classifier."""
    print("🏗️ Loading trained clarity classifier...")

    classifier = SimpleClarityClassifier()
    classifier.load_model("clarity_classifier_simple.pkl")

    return classifier


def generate_predictions(classifier, texts, indices):
    """Generate predictions for all texts."""
    print("🔍 Generating predictions...")

    predictions = classifier.predict(texts)

    # Map our predictions to CLARITY format
    clarity_mapping = {
        "Clear Reply": "Direct Reply",
        "Clear Non-Reply": "Direct Non-Reply",
        "Ambivalent Reply": "Indirect"  # Map ambiguous to indirect
    }

    # Convert predictions to CLARITY format
    clarity_predictions = [clarity_mapping.get(pred, "Indirect") for pred in predictions]

    print("✅ Generated predictions for all texts")
    print("📊 Prediction distribution:")
    unique_preds = set(clarity_predictions)
    for pred in unique_preds:
        count = clarity_predictions.count(pred)
        print(".1f")

    return indices, clarity_predictions


def create_submission_pickle(indices, predictions, output_file="clarity_submission.pickle"):
    """Create the submission pickle file."""
    print(f"💾 Creating submission file: {output_file}")

    with open(output_file, 'wb') as f:
        pickle.dump((indices, predictions), f)

    print("✅ Submission file created successfully!")
    print(f"📁 File: {output_file}")
    print(f"📊 Samples: {len(indices)}")
    print(f"🏷️  Predictions: {len(predictions)}")


def validate_submission(submission_file):
    """Validate the submission file format."""
    print("🔍 Validating submission file...")

    with open(submission_file, 'rb') as f:
        indices, predictions = pickle.load(f)

    # Basic validation
    assert len(indices) == len(predictions), "Indices and predictions length mismatch"
    assert all(isinstance(idx, int) for idx in indices), "All indices must be integers"
    assert all(pred in ["Direct Reply", "Direct Non-Reply", "Indirect"] for pred in predictions), "Invalid prediction labels"

    print("✅ Submission file validation passed!")
    print(f"   - {len(indices)} predictions")
    print(f"   - Indices range: {min(indices)} to {max(indices)}")
    print(f"   - Unique predictions: {set(predictions)}")


def main():
    """Main submission generation function."""
    print("🎯 CLARITY-SemEval-2026 Submission Generator")
    print("=" * 50)

    try:
        # Load test data
        texts, indices = load_test_data()

        # Load trained model
        classifier = load_trained_model()

        # Generate predictions
        indices, predictions = generate_predictions(classifier, texts, indices)

        # Create submission file
        submission_file = "clarity_submission.pickle"
        create_submission_pickle(indices, predictions, submission_file)

        # Validate submission
        validate_submission(submission_file)

        print("\n" + "=" * 50)
        print("🎉 SUBMISSION READY!")
        print("=" * 50)
        print(f"📁 Submit this file to CodeBench: {submission_file}")
        print("🏷️  Format: Pickle file with (indices, predictions) tuple")
        print("📊 Expected scores: ~60-70% accuracy (our baseline)")
        print("🏆 Competition: https://codalab.lisn.upsaclay.fr/competitions/2135")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()