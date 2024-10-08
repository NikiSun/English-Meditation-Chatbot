import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from transformers import logging as transformers_logging
import torch
import warnings

# Suppress warnings from transformers and BERTScore
transformers_logging.set_verbosity_error()
warnings.filterwarnings("ignore", category=UserWarning, module='bert_score')

# Function to tokenize words
def tokenize_words(sentence):
    return word_tokenize(sentence.lower())

# Define the file paths
reference_file_path = r''  # Path to the reference file
cleaned_generated_file_path = r'' # Path to the cleaned generated file

# Read the content from the reference file
with open(reference_file_path, 'r', encoding='utf-8') as ref_file:
    reference_content = ref_file.read()

# Read the content from the cleaned generated file
with open(cleaned_generated_file_path, 'r', encoding='utf-8') as gen_file:
    generated_content = gen_file.read()

# Ensure nltk resources are downloaded
nltk.download('punkt')
nltk.download('wordnet')

# # Debug: Print reference and generated content
# print("Reference Content:", reference_content)
# print("Generated Content:", generated_content)

# Tokenize the entire reference and generated content as one block each
tokenized_reference_content = tokenize_words(reference_content)
tokenized_generated_content = tokenize_words(generated_content)

# Compute BLEU score with smoothing for the entire generated content compared to the entire reference content
smoothing_function = SmoothingFunction().method1
bleu_score = sentence_bleu([tokenized_reference_content], tokenized_generated_content, smoothing_function=smoothing_function)

print("BLEU Score for Entire Content with Smoothing:", bleu_score)

# Compute ROUGE scores by comparing the entire generated content to the entire reference content
rouge_scorer_instance = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
rouge_scores = rouge_scorer_instance.score(reference_content, generated_content)

average_rouge_scores = {key: rouge_scores[key].fmeasure for key in rouge_scores}
print("ROUGE Scores for Entire Content:", average_rouge_scores)

# Compute METEOR score by comparing the entire generated content to the entire reference content
meteor_score = nltk.translate.meteor_score.meteor_score([tokenize_words(reference_content)], tokenized_generated_content)
print("METEOR Score for Entire Content:", meteor_score)

# Compute BERTScore by comparing the entire generated content to the entire reference content
from bert_score import score as bert_score

# Ensure the BERTScore model is on the correct device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

try:
    P, R, F1 = bert_score([generated_content], [reference_content], lang='en', rescale_with_baseline=False, device=device)
    print("BERTScore Precision (P) for Entire Content:", P.mean().item())
    print("BERTScore Recall (R) for Entire Content:", R.mean().item())
    print("BERTScore F1 for Entire Content:", F1.mean().item())
except Exception as e:
    print("Error computing BERTScore:", e)
