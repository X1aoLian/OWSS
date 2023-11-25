from sklearn.metrics import f1_score, recall_score, precision_score
def evaluate_metrics(true_labels, predicted_labels):
    # 1. Calculate accuracy for predicted values of 0 or 1
    mask = predicted_labels != 2
    accuracy = (true_labels[mask] == predicted_labels[mask]).float().mean().item()

    # 2.
    true_binary = (true_labels == 2).numpy()
    predicted_binary = (predicted_labels == 2).numpy()

    f1 = f1_score(true_binary, predicted_binary)
    recall = recall_score(true_binary, predicted_binary)
    precision = precision_score(true_binary, predicted_binary)

    print(f"Accuracy (only for predicted 0 or 1): {accuracy}")
    print(f"F1 Score: {f1}")
    print(f"Recall: {recall}")
    print(f"Precision: {precision}")
