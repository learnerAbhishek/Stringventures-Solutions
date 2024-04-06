import torch
from transformers import GPT2Tokenizer, GPT2ForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd

# Load your dataset (assuming you have a DataFrame named df with 'text' and 'sentiment' columns)
df = pd.read_csv('your_dataset.csv')  # Replace 'your_dataset.csv' with your dataset file path
X = df['text'].tolist()
y = df['sentiment'].tolist()

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Load pre-trained GPT-2 tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2ForSequenceClassification.from_pretrained('gpt2', num_labels=3)  # Assuming 3 classes: positive, negative, neutral

# Tokenize input texts
train_encodings = tokenizer(X_train, truncation=True, padding=True, max_length=64, return_tensors='pt')
test_encodings = tokenizer(X_test, truncation=True, padding=True, max_length=64, return_tensors='pt')

# Convert labels to tensors
train_labels = torch.tensor(y_train)
test_labels = torch.tensor(y_test)

# Fine-tune the model
optimizer = AdamW(model.parameters(), lr=5e-5)
model.train()
for epoch in range(3):  # You can adjust the number of epochs
    for i in range(0, len(train_labels), 8):  # Process data in smaller batches
        optimizer.zero_grad()
        batch_inputs = {key: val[i:i+8] for key, val in train_encodings.items()}
        outputs = model(**batch_inputs, labels=train_labels[i:i+8])
        loss = outputs.loss
        loss.backward()
        optimizer.step()

# Evaluate the fine-tuned model
model.eval()
predictions = []
for i in range(0, len(test_labels), 8):  # Process data in smaller batches
    batch_inputs = {key: val[i:i+8] for key, val in test_encodings.items()}
    with torch.no_grad():
        outputs = model(**batch_inputs)
        batch_predictions = torch.argmax(outputs.logits, dim=1).tolist()
        predictions.extend(batch_predictions)

# Calculate evaluation metrics
accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions, average='macro')
recall = recall_score(y_test, predictions, average='macro')
f1 = f1_score(y_test, predictions, average='macro')

# Print evaluation metrics
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)

# Output predictions for sample data
sample_data = ["This movie was great!", "The food was terrible.", "I'm feeling so-so."]
sample_encodings = tokenizer(sample_data, truncation=True, padding=True, max_length=64, return_tensors='pt')
sample_predictions = []
for i in range(0, len(sample_data), 8):  # Process data in smaller batches
    batch_inputs = {key: val[i:i+8] for key, val in sample_encodings.items()}
    with torch.no_grad():
        outputs = model(**batch_inputs)
        batch_predictions = torch.argmax(outputs.logits, dim=1).tolist()
        sample_predictions.extend(batch_predictions)
print("Sample predictions:", sample_predictions)

