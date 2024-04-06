import spacy

def extract_keywords_spacy(text_data, max_keywords=8):
    """
    Extract and list the most important keywords from the provided text data using spaCy.

    Args:
        text_data (list): List of strings representing the text data.
        max_keywords (int): Maximum number of keywords to extract and list. Default is 5.

    Returns:
        list: List of most important keywords.
    """
    # Load the English language model in spaCy
    nlp = spacy.load("en_core_web_sm")

    # Initialize a dictionary to store keywords and their frequencies
    keywords = {}

    # Process each text document
    for doc in nlp.pipe(text_data):
        # Iterate through tokens in the document
        for token in doc:
            # Check if the token is a noun or proper noun
            if token.pos_ in ["NOUN", "PROPN"]:
                # Convert token to lowercase and remove leading/trailing whitespace
                keyword = token.text.lower().strip()
                # Update the frequency count of the keyword
                keywords[keyword] = keywords.get(keyword, 0) + 1

    # Sort keywords by frequency in descending order
    sorted_keywords = sorted(keywords.items(), key=lambda x: x[1], reverse=True)

    # Extract the most important keywords
    most_important_keywords = [keyword for keyword, _ in sorted_keywords[:max_keywords]]

    return most_important_keywords

# Example usage:
text_data = [
    "This is a sample text document.",
    "TF-IDF is used for keyword extraction in natural language processing.",
    "Keywords are important for text analysis and information retrieval.",
    "Text data processing involves various techniques and algorithms.",
    "Machine learning models can be trained on text data for classification tasks.",
    "Data preprocessing is essential for cleaning and preparing text data.",
    "NLP libraries such as NLTK and spaCy provide tools for text analysis.",
    "Word embeddings capture semantic relationships between words in text data.",
    "Deep learning models like GPT-3 excel at generating human-like text.",
    "Sentiment analysis determines the emotional tone of text data.",
    "Named entity recognition identifies entities such as names and locations in text.",
    "Summarization algorithms condense text data while retaining key information.",
    "Topic modeling algorithms uncover latent topics present in text data.",
    "Text classification categorizes text data into predefined classes or categories.",
    "Dependency parsing analyzes syntactic structure and relationships in text data.",
    "Part-of-speech tagging labels words with their corresponding grammatical categories.",
    "Information extraction involves identifying and extracting structured information from text data.",
    "Text similarity measures quantify the similarity between two pieces of text.",
    "BERT, a transformer-based model, achieves state-of-the-art performance on various NLP tasks.",
    "Word frequency analysis helps identify common and rare words in text data."
]

most_important_keywords_spacy = extract_keywords_spacy(text_data)
with open("output.txt", "w") as f:
    f.write(f"Most important keywords (spaCy):, {most_important_keywords_spacy}")
    print("Most important keywords (spaCy):", most_important_keywords_spacy)

