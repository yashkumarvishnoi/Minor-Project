import pandas as pd
import torch
from sklearn.metrics import classification_report
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    Trainer
)
from torch.utils.data import Dataset
import numpy as np

# 1. Load the saved model and tokenizer
model_path = "./final_model"

# Load tokenizer
tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)

# Load model
model = DistilBertForSequenceClassification.from_pretrained(model_path)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# 2. Prepare the test data (assuming you have a test set)
# If you don't have a separate test set, you can use your validation set
test_df = pd.read_csv('balanced_legal_cases_dataset.csv')

# Apply the same preprocessing as during training
categories = ['administrative', 'criminal', 'civil', 'constitutional', 'family', 'commercial']
test_df['category'] = test_df[categories].idxmax(axis=1)
test_df['label'] = test_df['category'].map({cat: idx for idx, cat in enumerate(categories)})

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.replace('\n', ' ').replace('\r', ' ')
    return ' '.join(text.split()).strip()

test_df['case_text'] = test_df['case_text'].apply(clean_text)
test_df = test_df[test_df['case_text'].str.len() > 50]

test_texts = test_df['case_text'].tolist()
test_labels = test_df['label'].tolist()

# 3. Tokenize the test data
test_encodings = tokenizer(test_texts, truncation=True, padding='max_length', max_length=512)

# 4. Create Dataset class (same as during training)
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

test_dataset = LegalDataset(test_encodings, test_labels)

# 5. Create a Trainer instance for prediction
trainer = Trainer(model=model)

# 6. Make predictions
predictions = trainer.predict(test_dataset)
predicted_labels = np.argmax(predictions.predictions, axis=1)

# 7. Generate classification report
print("Detailed Classification Report:")
print(classification_report(
    test_labels, 
    predicted_labels, 
    target_names=categories,
    digits=4
))

# 8. Generate confusion matrix (optional)
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

cm = confusion_matrix(test_labels, predicted_labels)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=categories, 
            yticklabels=categories)
plt.title('Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

# 9. Calculate per-class accuracy
class_accuracy = {}
for i, category in enumerate(categories):
    mask = np.array(test_labels) == i
    class_accuracy[category] = (predicted_labels[mask] == i).mean()

print("\nPer-Class Accuracy:")
for category, acc in class_accuracy.items():
    print(f"{category:<15}: {acc:.4f}")