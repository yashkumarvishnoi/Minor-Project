
from flask import Flask, request, render_template
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM, DistilBertTokenizer, DistilBertForSequenceClassification
import torch
from typing import List
import os

app = Flask(__name__)

MAX_INPUT_CHARS = 20000
CHUNK_SIZE = 1024
CHUNK_SUMMARY_LEN = 150
FINAL_SUMMARY_LEN = 170
MIN_SUMMARY_LEN = 150
LOCAL_MODEL_DIR = "./flan-t5-xsum-model"  # Your existing model folder

# Load FLAN-T5-XSum model FROM LOCAL FOLDER ONLY
def load_summarization_model():
    """Load FLAN-T5-XSum model from local directory"""
    if not os.path.exists(LOCAL_MODEL_DIR):
        raise FileNotFoundError(f"Model directory not found at {LOCAL_MODEL_DIR}")
    
    print("Loading FLAN-T5-XSum from local folder...")
    tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_DIR)
    model = AutoModelForSeq2SeqLM.from_pretrained(LOCAL_MODEL_DIR)
    
    device = 0 if torch.cuda.is_available() else -1
    return pipeline(
        "summarization",
        model=model,
        tokenizer=tokenizer,
        device=device
    )

# Initialize models (will fail fast if model missing)
summarizer = load_summarization_model()

# Load classification model
classification_tokenizer = DistilBertTokenizer.from_pretrained('./final_model')
classification_model = DistilBertForSequenceClassification.from_pretrained('./final_model')
classification_model.eval()

# Category mapping
label_to_category = {
    0: 'administrative',
    1: 'criminal', 
    2: 'civil',
    3: 'constitutional',
    4: 'family',
    5: 'commercial'
}

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE) -> List[str]:
    """Split text into chunks respecting sentence boundaries"""
    sentences = text.split('. ')
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        if len(current_chunk) + len(sentence) < chunk_size:
            current_chunk += sentence + '. '
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + '. '
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

def summarize_chunk(chunk: str) -> str:
    """Summarize a single chunk of text"""
    result = summarizer(
        chunk,
        max_length=CHUNK_SUMMARY_LEN,
        min_length=int(CHUNK_SUMMARY_LEN*0.7),
        do_sample=False,
        truncation=True
    )
    return result[0]['summary_text']

def hierarchical_summarize(text: str) -> str:
    """Handle long documents through chunked summarization"""
    # Pre-process text
    text = text.strip()
    if not text:
        return ""
    
    # Safety limit
    if len(text) > MAX_INPUT_CHARS:
        text = text[:MAX_INPUT_CHARS]
    
    # Direct summarization for short texts
    if len(text) <= CHUNK_SIZE:
        return summarize_chunk(text)
    
    # Chunk and summarize
    chunks = chunk_text(text)
    chunk_summaries = [summarize_chunk(chunk) for chunk in chunks]
    combined_summary = " ".join(chunk_summaries)
    
    # Final summarization
    final_summary = summarizer(
        combined_summary,
        max_length=FINAL_SUMMARY_LEN,
        min_length=MIN_SUMMARY_LEN,
        do_sample=False
    )
    return final_summary[0]['summary_text']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process_case():
    case_text = request.form.get('case_text', '')
    
    if not case_text.strip():
        return render_template('index.html', error="Please provide case text")
    
    # Step 1: Hierarchical summarization
    summarized_text = hierarchical_summarize(case_text)
    
    # Step 2: Classification
    inputs = classification_tokenizer(
        summarized_text, 
        padding=True, 
        truncation=True, 
        return_tensors="pt"
    )
    
    with torch.no_grad():
        outputs = classification_model(**inputs)
        predictions = torch.argmax(outputs.logits, dim=-1)
        predicted_category = label_to_category[predictions[0].item()]
    
    # Render results
    return render_template(
        'results.html',
        original_length=len(case_text),
        summary=summarized_text,
        category=predicted_category,
        summary_length=len(summarized_text)
    )

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)