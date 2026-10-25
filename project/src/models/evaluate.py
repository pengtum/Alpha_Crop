"""
Evaluation metrics and visualization for crop classification models.

Computes: balanced accuracy, overall accuracy, Cohen's kappa,
area-weighted accuracy, per-class metrics, confusion matrix,
size-stratified evaluation, and per-country evaluation (EU mode).

Usage:
    Imported by train_rf.py and train_lgbm.py
    Or standalone: python -m src.models.evaluate --model-path models/rf_tier_b.joblib --tier b
"""
import sys
import os
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import (
    balanced_accuracy_score, accuracy_score, cohen_kappa_score,
    classification_report, confusion_matrix, f1_score
)
import joblib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
from config.config import (
    METRICS_DIR, FIGURES_DIR, MODELS_DIR,
    FEATURES_TEST_PATH, EU_FEATURES_TEST_PATH,
    COL_COUNTRY, COL_AREA,
)


def area_weighted_accuracy(y_true, y_pred, areas):
    """Compute accuracy weighted by parcel area."""
    correct = (y_true == y_pred).astype(float)
    return np.average(correct, weights=areas)


def size_stratified_eval(y_true, y_pred, areas):
    """Evaluate balanced accuracy by parcel size bins."""
    bins = {
        'tiny (<0.1 ha)': areas < 0.1,
        'small (0.1-0.5 ha)': (areas >= 0.1) & (areas < 0.5),
        'medium (0.5-2 ha)': (areas >= 0.5) & (areas < 2.0),
        'large (>2 ha)': areas >= 2.0
    }

    results = {}
    for name, mask in bins.items():
        if mask.sum() > 0:
            ba = balanced_accuracy_score(y_true[mask], y_pred[mask])
            oa = accuracy_score(y_true[mask], y_pred[mask])
            results[name] = {
                'balanced_accuracy': ba,
                'overall_accuracy': oa,
                'n_parcels': int(mask.sum())
            }
    return results


def evaluate_per_country(y_true, y_pred, test_df, class_names):
    """逐国评估指标（仅 EU 模式）。"""
    if COL_COUNTRY not in test_df.columns:
        return None

    country_results = {}
    for country in sorted(test_df[COL_COUNTRY].unique()):
        mask = (test_df[COL_COUNTRY] == country).values
        if mask.sum() < 10:
            continue

        y_t = y_true[mask]
        y_p = y_pred[mask]

        country_results[country] = {
            'balanced_accuracy': balanced_accuracy_score(y_t, y_p),
            'overall_accuracy': accuracy_score(y_t, y_p),
            'macro_f1': f1_score(y_t, y_p, average='macro'),
            'n_parcels': int(mask.sum()),
            'n_classes': len(np.unique(y_t)),
        }

    return country_results


def evaluate_model(y_true, y_pred, class_names, areas=None,
                   model_name='model', test_df=None):
    """Comprehensive model evaluation."""
    results = {}

    # Overall metrics
    results['balanced_accuracy'] = balanced_accuracy_score(y_true, y_pred)
    results['overall_accuracy'] = accuracy_score(y_true, y_pred)
    results['cohen_kappa'] = cohen_kappa_score(y_true, y_pred)
    results['macro_f1'] = f1_score(y_true, y_pred, average='macro')
    results['weighted_f1'] = f1_score(y_true, y_pred, average='weighted')

    if areas is not None:
        results['area_weighted_accuracy'] = area_weighted_accuracy(
            y_true, y_pred, areas)

    # Per-class report
    report = classification_report(
        y_true, y_pred, target_names=class_names, output_dict=True,
        zero_division=0
    )
    results['classification_report'] = report

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    results['confusion_matrix'] = cm

    # Size-stratified
    if areas is not None:
        results['size_stratified'] = size_stratified_eval(
            y_true, y_pred, areas)

    # Per-country evaluation (EU mode)
    if test_df is not None:
        country_results = evaluate_per_country(
            y_true, y_pred, test_df, class_names)
        if country_results:
            results['per_country'] = country_results

    # Save metrics
    os.makedirs(METRICS_DIR, exist_ok=True)

    report_df = pd.DataFrame(report).transpose()
    report_path = os.path.join(METRICS_DIR, f'{model_name}_report.csv')
    report_df.to_csv(report_path)

    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    cm_path = os.path.join(METRICS_DIR, f'{model_name}_confusion.csv')
    cm_df.to_csv(cm_path)

    # Save summary
    summary = {
        'model': model_name,
        'balanced_accuracy': results['balanced_accuracy'],
        'overall_accuracy': results['overall_accuracy'],
        'cohen_kappa': results['cohen_kappa'],
        'macro_f1': results['macro_f1'],
        'weighted_f1': results['weighted_f1'],
        'n_classes': len(class_names),
        'n_test': len(y_true),
    }
    if 'area_weighted_accuracy' in results:
        summary['area_weighted_accuracy'] = results['area_weighted_accuracy']

    summary_path = os.path.join(METRICS_DIR, 'summary_metrics.csv')
    if os.path.exists(summary_path):
        existing = pd.read_csv(summary_path)
        existing = existing[existing['model'] != model_name]
        summary_df = pd.concat([existing, pd.DataFrame([summary])],
                               ignore_index=True)
    else:
        summary_df = pd.DataFrame([summary])
    summary_df.to_csv(summary_path, index=False)

    # Save per-country metrics
    if 'per_country' in results:
        country_df = pd.DataFrame(results['per_country']).T
        country_df.index.name = 'country'
        country_path = os.path.join(METRICS_DIR,
                                     f'{model_name}_per_country.csv')
        country_df.to_csv(country_path)

    # Plot confusion matrix
    try:
        plot_confusion_matrix(cm, class_names, model_name)
    except Exception as e:
        print(f"Could not plot confusion matrix: {e}")

    return results


def plot_confusion_matrix(cm, class_names, model_name):
    """Plot and save confusion matrix heatmap."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns

    os.makedirs(FIGURES_DIR, exist_ok=True)

    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    cm_norm = np.nan_to_num(cm_norm)

    n_classes = len(class_names)
    fig_size = max(8, n_classes * 0.5)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))

    annot = n_classes <= 25
    fmt = '.2f' if annot else ''

    sns.heatmap(cm_norm, annot=annot, fmt=fmt, cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                ax=ax, vmin=0, vmax=1)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(f'Confusion Matrix - {model_name}')
    plt.xticks(rotation=45, ha='right',
               fontsize=max(6, 10 - n_classes // 10))
    plt.yticks(fontsize=max(6, 10 - n_classes // 10))
    plt.tight_layout()

    fig_path = os.path.join(FIGURES_DIR, f'{model_name}_confusion.png')
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"Confusion matrix saved to {fig_path}")


def print_results(results):
    """Pretty-print evaluation results."""
    print(f"\n{'=' * 50}")
    print(f"  Balanced Accuracy:      {results['balanced_accuracy']:.4f}")
    print(f"  Overall Accuracy:       {results['overall_accuracy']:.4f}")
    print(f"  Cohen's Kappa:          {results['cohen_kappa']:.4f}")
    print(f"  Macro F1:               {results['macro_f1']:.4f}")
    print(f"  Weighted F1:            {results['weighted_f1']:.4f}")
    if 'area_weighted_accuracy' in results:
        print(f"  Area-weighted Accuracy: "
              f"{results['area_weighted_accuracy']:.4f}")
    print(f"{'=' * 50}")

    # Size-stratified results
    if 'size_stratified' in results:
        print(f"\nSize-stratified evaluation:")
        for name, metrics in results['size_stratified'].items():
            print(f"  {name:25s}: BA={metrics['balanced_accuracy']:.3f}, "
                  f"OA={metrics['overall_accuracy']:.3f}, "
                  f"n={metrics['n_parcels']}")

    # Per-country results (EU mode)
    if 'per_country' in results:
        print(f"\nPer-country evaluation:")
        for country, metrics in sorted(results['per_country'].items()):
            print(f"  {country:4s}: BA={metrics['balanced_accuracy']:.3f}, "
                  f"F1={metrics['macro_f1']:.3f}, "
                  f"n={metrics['n_parcels']}, "
                  f"classes={metrics['n_classes']}")

    # Per-class summary (top/bottom 5 by F1)
    report = results['classification_report']
    class_metrics = {k: v for k, v in report.items()
                     if k not in ('accuracy', 'macro avg', 'weighted avg')}
    if class_metrics:
        sorted_classes = sorted(class_metrics.items(),
                                key=lambda x: x[1].get('f1-score', 0),
                                reverse=True)
        print(f"\nTop 5 classes by F1:")
        for name, m in sorted_classes[:5]:
            print(f"  {name:40s}: F1={m['f1-score']:.3f}, "
                  f"P={m['precision']:.3f}, R={m['recall']:.3f}, "
                  f"n={m['support']:.0f}")

        print(f"\nBottom 5 classes by F1:")
        for name, m in sorted_classes[-5:]:
            print(f"  {name:40s}: F1={m['f1-score']:.3f}, "
                  f"P={m['precision']:.3f}, R={m['recall']:.3f}, "
                  f"n={m['support']:.0f}")


def main():
    """Standalone evaluation of a saved model on the test set."""
    parser = argparse.ArgumentParser(description='Evaluate a saved model')
    parser.add_argument('--model-path', required=True,
                        help='Path to saved model (.joblib)')
    parser.add_argument('--tier', choices=['a', 'b', 'c'], required=True)
    parser.add_argument('--data', choices=['be', 'eu'], default='be',
                        help='Data source')
    args = parser.parse_args()

    model = joblib.load(args.model_path)

    if args.data == 'eu':
        models_dir = os.path.join(MODELS_DIR, 'eu')
        test_path = EU_FEATURES_TEST_PATH
        area_col = COL_AREA
    else:
        models_dir = MODELS_DIR
        test_path = FEATURES_TEST_PATH
        area_col = 'GRAF_OPP'

    feature_cols = joblib.load(os.path.join(models_dir, 'feature_cols.joblib'))
    le_dict = joblib.load(os.path.join(models_dir, 'label_encoders.joblib'))

    label_col = f'label_tier_{args.tier}'
    encoded_col = f'{label_col}_encoded'

    test = pd.read_parquet(test_path)
    X_test = test[feature_cols].fillna(0).values
    y_test = test[encoded_col].values
    class_names = le_dict[label_col].classes_

    y_pred = model.predict(X_test)
    model_name = os.path.basename(args.model_path).replace('.joblib', '')

    areas = test[area_col].values if area_col in test.columns else None
    results = evaluate_model(
        y_test, y_pred, class_names, areas,
        model_name=model_name, test_df=test
    )
    print_results(results)


if __name__ == '__main__':
    main()
