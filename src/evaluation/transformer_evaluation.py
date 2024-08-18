import torch
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
import logging
import pandas as pd

def evaluate_transformer(model, dataloader, device):
    """
    Evaluating Transformer model.
    :param model: Trained Transformer model
    :param dataloader: DataLoader for the test data
    :param device: Device to perform evaluation on ('cpu' or 'cuda')
    :return: Dictionary of evaluation metrics
    """
    logger = logging.getLogger('evaluation_logger')
    
    try:
        model.eval()
        all_preds = []
        all_labels = []

        logger.info("Starting evaluation of Transformer model...")

        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                inputs, labels = batch
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                
                all_preds.append(preds.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
                
                logger.debug(f"Processed batch {batch_idx + 1}/{len(dataloader)}")

        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)

        # To calculate evaluation metrics
        logger.info("Calculating evaluation metrics...")
        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='weighted')
        precision = precision_score(all_labels, all_preds, average='weighted')
        recall = recall_score(all_labels, all_preds, average='weighted')
        report = classification_report(all_labels, all_preds)
        conf_matrix = confusion_matrix(all_labels, all_preds)

        logger.info(f"Transformer Model Accuracy: {accuracy}")
        logger.info(f"Transformer Model F1 Score: {f1}")
        logger.info(f"Transformer Model Precision: {precision}")
        logger.info(f"Transformer Model Recall: {recall}")
        logger.info(f"Classification Report:\n{report}")
        logger.info(f"Confusion Matrix:\n{conf_matrix}")

        report_path = 'path/to/report_dir/transformer_classification_report.json'
        cm_path = 'path/to/report_dir/transformer_confusion_matrix.csv'
        accuracy_path = 'path/to/report_dir/transformer_accuracy.txt'
        
        pd.DataFrame(conf_matrix).to_csv(cm_path, index=False)
        with open(report_path, 'w') as f:
            f.write(report)
        with open(accuracy_path, 'w') as f:
            f.write(str(accuracy))
        
        logger.info(f"Classification report saved to {report_path}")
        logger.info(f"Confusion matrix saved to {cm_path}")
        logger.info(f"Accuracy saved to {accuracy_path}")

        return {
            'accuracy': accuracy,
            'f1_score': f1,
            'precision': precision,
            'recall': recall,
            'classification_report': report,
            'confusion_matrix': conf_matrix
        }
    
    except Exception as e:
        logger.error(f"Error during Transformer model evaluation: {e}", exc_info=True)
        raise
