"""
Visualizations for model evaluation.

Generates:
- Confusion matrix heatmap
- Precision/recall bar chart
- Model comparison (Rules vs TF-IDF vs SBERT)
"""

import logging
from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd
from pathlib import Path

logger = logging.getLogger(__name__)


def confusion_matrix_html(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_names: List[str] = None,
    output_path: Path = None,
) -> str:
    """
    Generate HTML heatmap of confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        label_names: List of category names
        output_path: Path to save HTML (if None, returns string)
    
    Returns:
        HTML string for heatmap
    """
    from sklearn.metrics import confusion_matrix
    
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=label_names if label_names is not None else None)
    
    # Normalize by true label (recall)
    row_sums = cm.sum(axis=1, keepdims=True).astype(float)
    row_sums[row_sums == 0] = 1.0
    cm_normalized = cm.astype(float) / row_sums
    
    if label_names is None:
        label_names = [f"Class {i}" for i in range(len(cm))]
    
    logger.info(f"Confusion matrix shape: {cm.shape}")
    
    # Generate HTML table
    html = "<table border='1'>\n"
    html += "<tr><th>Predicted \\ True</th>"
    for label in label_names:
        html += f"<th>{label}</th>"
    html += "</tr>\n"
    
    for i, true_label in enumerate(label_names):
        html += f"<tr><th>{true_label}</th>"
        for j in range(len(label_names)):
            count = cm[i, j]
            pct = cm_normalized[i, j] * 100
            color = f"rgba(200, 255, 200)" if count > 0 else "white"
            html += f"<td style='background-color:{color}'>{count}<br/>({pct:.0f}%)</td>"
        html += "</tr>\n"
    
    html += "</table>"
    
    if output_path:
        Path(output_path).write_text(html)
        logger.info(f"Saved confusion matrix to {output_path}")
    
    return html


def precision_recall_chart_html(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_names: List[str] = None,
    output_path: Path = None,
) -> str:
    """
    Generate HTML bar chart of precision/recall per category.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        label_names: Category names
        output_path: Path to save HTML
    
    Returns:
        HTML string with chart
    """
    from sklearn.metrics import precision_recall_fscore_support
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=label_names if label_names is not None else None, zero_division=0
    )
    
    if label_names is None:
        label_names = [f"Class {i}" for i in range(len(precision))]
    
    # Build CSV data for chart
    data = []
    for i, label in enumerate(label_names):
        data.append({
            "Category": label,
            "Precision": f"{precision[i]:.2f}",
            "Recall": f"{recall[i]:.2f}",
            "F1": f"{f1[i]:.2f}",
        })
    
    # Convert to JSON for chart.js
    import json
    data_json = json.dumps({
        "labels": label_names,
        "precision": [f"{p:.2f}" for p in precision],
        "recall": [f"{r:.2f}" for r in recall],
        "f1": [f"{x:.2f}" for x in f1],
    })
    
    html = f"""
    <div style="width: 800px; height: 400px;">
        <canvas id="prChart"></canvas>
    </div>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script>
        const data = {data_json};
        const ctx = document.getElementById('prChart').getContext('2d');
        new Chart(ctx, {{
            type: 'bar',
            data: {{
                labels: data.labels,
                datasets: [
                    {{
                        label: 'Precision',
                        data: data.precision,
                        backgroundColor: 'rgba(75, 192, 192, 0.7)',
                    }},
                    {{
                        label: 'Recall',
                        data: data.recall,
                        backgroundColor: 'rgba(255, 159, 64, 0.7)',
                    }},
                ]
            }},
            options: {{
                responsive: true,
                title: {{text: 'Precision / Recall per Category'}}
            }}
        }});
    </script>
    """
    
    if output_path:
        Path(output_path).write_text(html)
        logger.info(f"Saved P/R chart to {output_path}")
    
    return html


def comparison_table_html(
    results: Dict[str, Dict[str, float]],
    output_path: Path = None,
) -> str:
    """
    Generate HTML comparison table for multiple models.
    
    Args:
        results: {
            "Rules": {"accuracy": 0.5, "precision": 0.6, ...},
            "TF-IDF": {"accuracy": 0.75, ...},
            "SBERT": {"accuracy": 0.85, ...},
        }
        output_path: Path to save HTML
    
    Returns:
        HTML table string
    """
    df = pd.DataFrame(results).T
    
    html = "<table border='1' style='border-collapse: collapse;'>\n"
    html += "<tr><th>Model</th>"
    for col in df.columns:
        html += f"<th>{col}</th>"
    html += "</tr>\n"
    
    for model_name, row in df.iterrows():
        html += f"<tr><th>{model_name}</th>"
        for col, value in row.items():
            # Highlight column best values.
            best = pd.to_numeric(df[col], errors="coerce").max() == value
            style = "background-color: lightgreen; font-weight: bold;" if best else ""
            html += f"<td style='{style}'>{value:.3f}</td>"
        html += "</tr>\n"
    
    html += "</table>"
    
    if output_path:
        Path(output_path).write_text(html)
        logger.info(f"Saved comparison table to {output_path}")
    
    return html


def generate_evaluation_report(
    y_true: np.ndarray,
    predictions: Dict[str, np.ndarray],  # {model_name: predictions}
    label_names: List[str],
    output_dir: Path = None,
) -> Dict[str, str]:
    """
    Generate full evaluation report (confusion matrix, P/R, comparison).
    
    Args:
        y_true: True labels
        predictions: {"Rules": [0,1,2,...], "TF-IDF": [...], "SBERT": [...]}
        label_names: Category names
        output_dir: Directory to save HTML files
    
    Returns:
        Dict of {filename: html_content}
    """
    if output_dir is None:
        output_dir = Path("data/ml/evaluation_report")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {}
    comparison_metrics = {}
    
    # Process each model
    for model_name, y_pred in predictions.items():
        logger.info(f"Generating visualizations for {model_name}...")
        
        # Confusion matrix
        cm_path = output_dir / f"confusion_matrix_{model_name}.html"
        confusion_matrix_html(y_true, y_pred, label_names, cm_path)
        
        # Precision/Recall
        pr_path = output_dir / f"precision_recall_{model_name}.html"
        precision_recall_chart_html(y_true, y_pred, label_names, pr_path)
        
        # Collect metrics for comparison
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        comparison_metrics[model_name] = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision_macro": precision_score(
                y_true, y_pred, labels=label_names, average='macro', zero_division=0
            ),
            "recall_macro": recall_score(
                y_true, y_pred, labels=label_names, average='macro', zero_division=0
            ),
            "f1_macro": f1_score(
                y_true, y_pred, labels=label_names, average='macro', zero_division=0
            ),
        }
        
        results[model_name] = str(cm_path)
    
    # Comparison table
    comp_path = output_dir / "model_comparison.html"
    comparison_table_html(comparison_metrics, comp_path)
    
    logger.info(f"Evaluation report saved to {output_dir}")
    
    # Create index
    index_html = "<html><body><h1>Model Evaluation Report</h1><ul>"
    for model_name in predictions.keys():
        index_html += f"<li><a href='confusion_matrix_{model_name}.html'>CM: {model_name}</a></li>"
        index_html += f"<li><a href='precision_recall_{model_name}.html'>P/R: {model_name}</a></li>"
    index_html += f"<li><a href='model_comparison.html'>Comparison</a></li>"
    index_html += "</ul></body></html>"
    
    (output_dir / "index.html").write_text(index_html)
    
    return {
        "comparison": str(comp_path),
        "index": str(output_dir / "index.html"),
    }
