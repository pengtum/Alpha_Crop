"""
Train LightGBM classifier on AlphaEarth embeddings.

Usage:
    # 比利时模式
    python -m src.models.train_lgbm --tier b [--subsample 500]

    # 泛欧洲模式
    python -m src.models.train_lgbm --tier b --data eu
"""
import sys
import os
import argparse
import numpy as np
import pandas as pd
import joblib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
from config.config import (
    MODELS_DIR, LGBM_PARAMS, LGBM_EARLY_STOPPING, RANDOM_STATE
)
from src.models.train_rf import load_data, subsample_train
from src.models.evaluate import evaluate_model, print_results


def train_lightgbm(X_train, y_train, X_val, y_val, n_classes):
    """Train LightGBM with early stopping on validation set."""
    try:
        import lightgbm as lgb
    except ImportError:
        print("ERROR: LightGBM not installed.")
        print("Install with: pip install lightgbm")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("TRAINING LIGHTGBM")
    print("=" * 60)

    params = LGBM_PARAMS.copy()

    if n_classes > 2:
        params['objective'] = 'multiclass'
        params['num_class'] = n_classes
        params['metric'] = 'multi_logloss'

    print(f"Key parameters:")
    for k in ['n_estimators', 'learning_rate', 'num_leaves', 'class_weight']:
        print(f"  {k}: {params.get(k)}")

    model = lgb.LGBMClassifier(**params)

    print(f"Training on {len(X_train)} samples...")
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[
            lgb.early_stopping(LGBM_EARLY_STOPPING),
            lgb.log_evaluation(50)
        ]
    )

    from sklearn.metrics import balanced_accuracy_score
    val_pred = model.predict(X_val)
    val_ba = balanced_accuracy_score(y_val, val_pred)
    print(f"\nValidation Balanced Accuracy: {val_ba:.4f}")
    print(f"Best iteration: {model.best_iteration_}")

    return model


def main():
    parser = argparse.ArgumentParser(description='Train LightGBM')
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
    model = train_lightgbm(X_train_sub, y_train_sub, X_val, y_val,
                           n_classes=len(class_names))

    # Evaluate on test set
    print("\n" + "=" * 60)
    print("TEST SET EVALUATION")
    print("=" * 60)
    y_pred = model.predict(X_test)

    areas = test_df[area_col].values if area_col in test_df.columns else None
    prefix = 'eu_' if args.data == 'eu' else ''
    model_name = f'{prefix}lgbm_tier_{args.tier}'

    results = evaluate_model(
        y_test, y_pred, class_names, areas,
        model_name=model_name, test_df=test_df
    )
    print_results(results)

    # Feature importance
    importances = model.feature_importances_
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
                              f'lgbm_tier_{args.tier}{suffix}.joblib')
    joblib.dump(model, model_path)
    print(f"\nModel saved to {model_path}")

    importance_path = os.path.join(
        save_dir, f'lgbm_tier_{args.tier}{suffix}_importance.csv')
    importance_df.to_csv(importance_path, index=False)


if __name__ == '__main__':
    main()
