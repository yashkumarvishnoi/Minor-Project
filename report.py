
import pandas as pd
import torch
import numpy as np
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset

# Define LegalDataset class
class LegalDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
        
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item
    
    def __len__(self):
        return len(self.labels)

# Define categories
categories = ['administrative', 'criminal', 'civil', 'constitutional', 'family', 'commercial']

# 1. Load and preprocess dataset to recreate validation set
df = pd.read_csv('smart_legal_cases_dataset.csv')

# Category processing
df['category'] = df[categories].idxmax(axis=1)
df['label'] = df['category'].map({cat: idx for idx, cat in enumerate(categories)})

# Clean text
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.replace('\n', ' ').replace('\r', ' ')
    return ' '.join(text.split()).strip()

df['case_text'] = df['case_text'].apply(clean_text)
df = df[df['case_text'].str.len() > 50]

# Train-validation split
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df['case_text'].tolist(),
    df['label'].tolist(),
    test_size=0.2,
    random_state=42,
    stratify=df['label']
)

# Tokenize validation texts
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased', model_max_length=512)
val_encodings = tokenizer(val_texts, truncation=True, padding='max_length', max_length=512)

# Create validation dataset
val_dataset = LegalDataset(val_encodings, val_labels)

# 2. Load the trained model
model_path = "./final_model"
model = DistilBertForSequenceClassification.from_pretrained(model_path)

# 3. Define training arguments (minimal, for evaluation)
training_args = TrainingArguments(
    output_dir='./results',
    per_device_eval_batch_size=8,
    evaluation_strategy="epoch",
    fp16=torch.cuda.is_available(),
    report_to="none",
)

# 4. Initialize Trainer for evaluation
trainer = Trainer(
    model=model,
    args=training_args,
    eval_dataset=val_dataset,
    tokenizer=tokenizer
)

# 5. Get predictions on validation set
print("Generating predictions on validation set...")
predictions = trainer.predict(val_dataset)
pred_labels = np.argmax(predictions.predictions, axis=-1)
true_labels = val_labels

# 6. Generate classification report
target_names = categories
report = classification_report(true_labels, pred_labels, target_names=target_names, digits=4)

# 7. Compute additional metrics
accuracy = accuracy_score(true_labels, pred_labels)
f1_macro = f1_score(true_labels, pred_labels, average='macro')
f1_weighted = f1_score(true_labels, pred_labels, average='weighted')

# 8. Print results
print("\nClassification Report:")
print(report)
print("\nSummary Metrics:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Macro F1: {f1_macro:.4f}")
print(f"Weighted F1: {f1_weighted:.4f}")

# 9. Save classification report to file
with open("classification_report.txt", "w") as f:
    f.write("Classification Report:\n")
    f.write(report)
    f.write("\n\nSummary Metrics:\n")
    f.write(f"Accuracy: {accuracy:.4f}\n")
    f.write(f"Macro F1: {f1_macro:.4f}\n")
    f.write(f"Weighted F1: {f1_weighted:.4f}\n")

print("\nClassification report saved to 'classification_report.txt'")
