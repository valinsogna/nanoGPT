
import nltk
from rouge import Rouge

def evaluate_metrics(self, predicted_output, actual_output):

    f1_score = self.calculate_f1(predicted_output, actual_output)
    bleu_score = self.calculate_bleu(predicted_output, actual_output)
    rouge_1, rouge_2, rouge_l = self.calculate_rouge(predicted_output, actual_output)

    return f1_score, bleu_score, rouge_1, rouge_2, rouge_l

def calculate_f1(self, predicted, actual):
    predicted_tokens = predicted.split()
    actual_tokens = actual.split()

    common_tokens = set(predicted_tokens) & set(actual_tokens)
    precision = len(common_tokens) / len(predicted_tokens) if len(predicted_tokens) > 0 else 0
    recall = len(common_tokens) / len(actual_tokens) if len(actual_tokens) > 0 else 0

    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return f1

def calculate_bleu(self, predicted, actual):
    # Convert the input sentences into lists of tokens
    predicted_tokens = predicted.split()
    actual_tokens = actual.split()

    # Calculate BLEU score for 4-grams (BLEU-4)
    weights = (0.25, 0.25, 0.25, 0.25)  # Equal weights for 1-gram, 2-gram, 3-gram, and 4-gram
    bleu_score = nltk.translate.bleu_score.sentence_bleu([actual_tokens], predicted_tokens, weights)
    return bleu_score

def calculate_rouge(self, predicted, actual):
    rouge = Rouge()

    # Calculate ROUGE score (ROUGE-1, ROUGE-2, and ROUGE-L)
    scores = rouge.get_scores(predicted, actual)
    rouge_1 = scores[0]['rouge-1']['f']
    rouge_2 = scores[0]['rouge-2']['f']
    rouge_l = scores[0]['rouge-l']['f']

    return rouge_1, rouge_2, rouge_l
