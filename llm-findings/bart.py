from transformers import pipeline

# Load the zero-shot classification pipeline with a different model
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Define the candidate labels
candidate_labels = ["positive", "negative", "neutral"]

# Define a list of reviews
reviews = [
    "This product was great! I absolutely loved it.",
    "Worst purchase I have made. Completely disappointed.",
    "Okay product, but would not buy again.",
    "I dont know what to think about this product.",
    "its not bad but not good either."
]

# Classify the sentiment of each review
for review in reviews:
    result = classifier(review, candidate_labels)
    print(f"Review: {review}\nSentiment: {result['labels'][0]}, Score: {result['scores'][0]:.2f}\n")