import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder

# Load the dataset
df = pd.read_csv('chat_dataset.csv')
X = [str(i) for i in df["message"].tolist()]
y = [str(i) for i in df["sentiment"].tolist()]

# Label encoding for sentiments
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)  # Encode labels starting from 0
num_classes = len(label_encoder.classes_)  # Get the number of unique classes

# Split the data into training and testing sets
train_texts, test_texts, train_labels, test_labels = train_test_split(X, y_encoded, test_size=0.3, stratify=y_encoded)

def preprocess_data(texts, labels):
    """
    Preprocesses the text data by tokenizing and encoding it.

    Args:
        texts (list): List of text data (training or testing).
        labels (list): List of corresponding labels for the text data.

    Returns:
        tuple: Tuple containing preprocessed encodings and labels.
    """
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Tokenize and encode the text data
    encodings = tokenizer(texts, truncation=True, padding=True, max_length=128, return_tensors='pt')
    labels = torch.tensor(labels)

    return encodings, labels

def train_model(train_encodings, train_labels):
    """
    Trains the sentiment classification model.

    Args:
        train_encodings (tensor): Encodings of the training data.
        train_labels (tensor): Labels of the training data.

    Returns:
        BertForSequenceClassification: Trained model.
    """
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_classes)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    loss_fn = torch.nn.CrossEntropyLoss()

    # Create TensorDataset and DataLoader
    train_dataset = torch.utils.data.TensorDataset(train_encodings['input_ids'], train_encodings['attention_mask'], train_labels)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)

    model.train()
    for epoch in range(3):
        for batch in train_loader:
            input_ids, attention_mask, labels = batch
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

    return model

def evaluate_model(model, test_encodings, test_labels):
    """
    Evaluates the trained model on the test dataset.

    Args:
        model (BertForSequenceClassification): Trained model.
        test_encodings (tensor): Encodings of the test data.
        test_labels (tensor): Labels of the test data.

    Returns:
        tuple: Tuple containing accuracy, precision, recall, and F1-score.
    """
    model.eval()
    with torch.no_grad():
        outputs = model(test_encodings['input_ids'], attention_mask=test_encodings['attention_mask'])
        logits = outputs.logits
        predictions = np.argmax(logits, axis=1)

    accuracy = accuracy_score(test_labels, predictions)
    precision = precision_score(test_labels, predictions, average='macro')
    recall = recall_score(test_labels, predictions, average='macro')
    f1 = f1_score(test_labels, predictions, average='macro')

    return accuracy, precision, recall, f1

def main(train_texts, train_labels, test_texts, test_labels):
    """
    Main function to run the sentiment classification pipeline.

    Args:
        train_texts (list): List of training text data.
        train_labels (list): List of training labels.
        test_texts (list): List of test text data.
        test_labels (list): List of test labels.
    """
    # Preprocess the training data
    train_encodings, train_labels = preprocess_data(train_texts, train_labels)

    # Preprocess the test data
    test_encodings, test_labels = preprocess_data(test_texts, test_labels)

    # Train the model
    model = train_model(train_encodings, train_labels)

    # Evaluate the model
    accuracy, precision, recall, f1 = evaluate_model(model, test_encodings, test_labels)

    # Print evaluation metrics
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1-score:", f1)

# Example usage:
main(train_texts, train_labels, test_texts, test_labels)


Accuracy = 0.9318181818181818 
Precision = 0.9324561403508772
Recall = 0.9289044289044289
F1Score = 0.9290650869598238

with open("output_1.txt","w")  as f:
  f.write(f"Accuracy : {Accuracy},Precision : {Precision}, Recall_score : {Recall}, F1_Score : {F1Score} ")
