import os
import json
import pandas as pd
from jinja2 import Template
from src.config import config
from src.utils.logger import setup_logger

logger = setup_logger('generate_report_logger', os.path.join(config.LOG_DIR, 'generate_report.log'))

def load_evaluation_results(model_name):
    """
    Loading evaluation results for given model.
    :param model_name: Multidisciplinary Deepfake Detection
    :return: Dictionary containing classification report and confusion matrix
    """
    logger.info(f"Loading evaluation results for {model_name} model...")
    
    report_path = os.path.join(config.REPORT_DIR, f'{model_name}_classification_report.json')
    cm_path = os.path.join(config.REPORT_DIR, f'{model_name}_confusion_matrix.csv')
    accuracy_path = os.path.join(config.REPORT_DIR, f'{model_name}_accuracy.txt')

    try:
        with open(report_path, 'r') as f:
            classification_report = json.load(f)
        
        confusion_matrix = pd.read_csv(cm_path, index_col=0)
        
        if os.path.exists(accuracy_path):
            with open(accuracy_path, 'r') as f:
                accuracy = f.read().strip()
        else:
            accuracy = None
        
        logger.info(f"Successfully loaded evaluation results for {model_name} model.")
        
        return {
            'classification_report': classification_report,
            'confusion_matrix': confusion_matrix,
            'accuracy': accuracy
        }
    except Exception as e:
        logger.error(f"Error loading evaluation results for {model_name} model: {e}")
        raise

def generate_html_report(models_results):
    """
    Generating report from evaluation results.
    :param models_results: Dictionary containing evaluation results for all models
    :return: HTML content as a string
    """
    logger.info("Generating HTML report...")

    template = Template("""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Model Evaluation Report</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; }
            h1, h2 { text-align: center; }
            table { width: 100%; border-collapse: collapse; margin-bottom: 40px; }
            th, td { border: 1px solid #ddd; padding: 8px; text-align: center; }
            th { background-color: #f2f2f2; }
            .confusion-matrix { margin-top: 20px; }
        </style>
    </head>
    <body>
        <h1>Model Evaluation Report</h1>
        {% for model_name, results in models_results.items() %}
            <h2>{{ model_name | capitalize }} Model</h2>
            <h3>Classification Report</h3>
            <pre>{{ results['classification_report'] | tojson(indent=4) }}</pre>
            
            <h3>Confusion Matrix</h3>
            <div class="confusion-matrix">
                {{ results['confusion_matrix'].to_html(classes='data', header=True, index=True) }}
            </div>
            
            {% if results['accuracy'] %}
                <h3>Accuracy</h3>
                <p>{{ results['accuracy'] }}</p>
            {% endif %}
            
            <hr>
        {% endfor %}
    </body>
    </html>
    """)
    
    html_content = template.render(models_results=models_results)
    
    logger.info("HTML report generation complete.")
    return html_content

def save_html_report(html_content, report_path):
    """
    Save the HTML report to a file.
    :param html_content: HTML content as a string
    :param report_path: Path to save the HTML report
    """
    logger.info(f"Saving HTML report to {report_path}...")
    
    try:
        with open(report_path, 'w') as f:
            f.write(html_content)
        
        logger.info(f"HTML report saved successfully to {report_path}.")
    except Exception as e:
        logger.error(f"Error saving HTML report: {e}")
        raise

if __name__ == "__main__":
    logger.info("Starting report generation process...")
    
    models = ["cnn", "transformer", "svm", "bayesian", "vision_transformer"]
    models_results = {}

    for model in models:
        models_results[model] = load_evaluation_results(model)
    
    html_content = generate_html_report(models_results)
    
    report_path = os.path.join(config.REPORT_DIR, 'model_evaluation_report.html')
    save_html_report(html_content, report_path)
    
    logger.info("Report generation process completed successfully.")
