"""
Train Random Forest classifier on AlphaEarth embeddings.

Usage:
    # 比利时模式
    python -m src.models.train_rf --tier b [--subsample 500]

    # 泛欧洲模式
    python -m src.models.train_rf --tier b --data eu
"""
import sys
import os
import argparse
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
from config.config import (
    FEATURES_TRAIN_PATH, FEATURES_VAL_PATH, FEATURES_TEST_PATH,
    EU_FEATURES_TRAIN_PATH, EU_FEATURES_VAL_PATH, EU_FEATURES_TEST_PATH,
    MODELS_DIR, RF_PARAMS, RANDOM_STATE, COL_AREA,
)
from src.models.evaluate import evaluate_model, print_results


def load_data(tier='b', data='be'):
    """Load feature matrices and labels for the specified tier.

    Args:
        tier: Classification tier ('a', 'b', 'c')
        data: Data source ('be' for Belgian, 'eu' for pan-European)
    """
    label_col = f'label_tier_{tier}'
    encoded_col = f'{label_col}_encoded'

    if data == 'eu':
        models_dir = os.path.join(MODELS_DIR, 'eu')
        train_path = EU_FEATURES_TRAIN_PATH
        val_path = EU_FEATURES_VAL_PATH
        test_path = EU_FEATURES_TEST_PATH
        area_col = COL_AREA
    else:
        models_dir = MODELS_DIR
        train_path = FEATURES_TRAIN_PATH
        val_path = FEATURES_VAL_PATH
        test_path = FEATURES_TEST_PATH
        area_col = 'GRAF_OPP'

    feature_cols = joblib.load(os.path.join(models_dir, 'feature_cols.joblib'))

    train = pd.read_parquet(train_path)
    val = pd.read_parquet(val_path)
    test = pd.read_parquet(test_path)

    X_train = train[feature_cols].fillna(0).values
    X_val = val[feature_cols].fillna(0).values
    X_test = test[feature_cols].fillna(0).values

    y_train = train[encoded_col].values
    y_val = val[encoded_col].values
    y_test = test[encoded_col].values

    le = joblib.load(os.path.join(models_dir, 'label_encoders.joblib'))
    class_names = le[label_col].classes_

    print(f"Tier {tier.upper()}: {len(class_names)} classes (data={data})")
    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    print(f"Features: {X_train.shape[1]}")

    return (X_train, y_train, X_val, y_val, X_test, y_test,
            class_names, train, val, test, feature_cols, area_col)


def subsample_train(X_train, y_train, n_per_class, seed=42):
    """Stratified subsampling: cap each class at n_per_class."""
    indices = []
    for cls in np.unique(y_train):
        cls_idx = np.where(y_train == cls)[0]
        n = min(n_per_class, len(cls_idx))
        rng = np.random.RandomState(seed)
        sampled = rng.choice(cls_idx, size=n, replace=False)
        indices.extend(sampled)

    indices = sorted(indices)
    return X_train[indices], y_train[indices]


def train_random_forest(X_train, y_train, X_val, y_val):
    """Train Random Forest with configured hyperparameters."""
    print("\n" + "=" * 60)
    print("TRAINING RANDOM FOREST")
    print("=" * 60)

    rf = RandomForestClassifier(**RF_PARAMS)
    print(f"Parameters: {RF_PARAMS}")
    print(f"Training on {len(X_train)} samples...")

    rf.fit(X_train, y_train)

    from sklearn.metrics import balanced_accuracy_score
    val_pred = rf.predict(X_val)
    val_ba = balanced_accuracy_score(y_val, val_pred)
    print(f"Validation Balanced Accuracy: {val_ba:.4f}")

    return rf


def main():
    parser = argparse.ArgumentParser(description='Train Random Forest')
    parser.add_argument('--tier', choices=['a', 'b', 'c'], required=True,
                        help='Classification tier')
    parser.add_argument('--subsample', type=int, default=0,
                        help='Max samples per class (0=all)')
    parser.add_argument('--data', choices=['be', 'eu'], default='be',
                        help='Data source: be (Belgian) or eu (pan-European)')
    args = parser.parse_args()

    # Load data
    (X_train, y_train, X_val, y_val, X_test, y_test,
     class_names, train_df, val_df, test_df,
     feature_cols, area_col) = load_data(args.tier, data=args.data)

    # Subsample if requested
    if args.subsample > 0:
        print(f"\nSubsampling to {args.subsample} per class...")
        X_train_sub, y_train_sub = subsample_train(
            X_train, y_train, args.subsample)
        print(f"Subsampled: {len(X_train_sub)} training samples")
    else:
        X_train_sub, y_train_sub = X_train, y_train

    # Train
    rf = train_random_forest(X_train_sub, y_train_sub, X_val, y_val)

    # Evaluate on test set
    print("\n" + "=" * 60)
    print("TEST SET EVALUATION")
    print("=" * 60)
    y_pred = rf.predict(X_test)

    areas = test_df[area_col].values if area_col in test_df.columns else None
    prefix = 'eu_' if args.data == 'eu' else ''
    model_name = f'{prefix}rf_tier_{args.tier}'

    results = evaluate_model(
        y_test, y_pred, class_names, areas,
        model_name=model_name, test_df=test_df
    )
    print_results(results)

    # Feature importance
    importances = rf.feature_importances_
    importance_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': importances
    }).sort_values('importance', ascending=False)
    print(f"\nTop 20 features:")
    print(importance_df.head(20).to_string(index=False))

    # Save model
    save_dir = os.path.join(MODELS_DIR, 'eu') if args.data == 'eu' else MODELS_DIR
    os.makedirs(save_dir, exist_ok=True)
    suffix = f'_sub{args.subsample}' if args.subsample > 0 else ''
    model_path = os.path.join(save_dir,
                              f'rf_tier_{args.tier}{suffix}.joblib')
    joblib.dump(rf, model_path)
    print(f"\nModel saved to {model_path}")

    importance_path = os.path.join(
        save_dir, f'rf_tier_{args.tier}{suffix}_importance.csv')
    importance_df.to_csv(importance_path, index=False)


if __name__ == '__main__':
    main()
