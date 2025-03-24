def calculate_f1_score(precision, recall):
    """
    F1 Score = 2 * (Precision * Recall) / (Precision + Recall)
    """
    return 2 * (precision * recall) / (precision + recall)

f1 = calculate_f1_score(0.95251,0.96984)
print(f1)