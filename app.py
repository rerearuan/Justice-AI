from flask import Flask, request, render_template, jsonify
import google.generativeai as genai
import torch
from transformers import AutoTokenizer, BertModel
import fitz  # PyMuPDF
import re
import torch.nn as nn
import pickle

from dotenv import load_dotenv
import os

load_dotenv()

import nltk
nltk.download('punkt')  
nltk.download('stopwords')
nltk.download('punkt_tab') 
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

app = Flask(__name__)

# Konfigurasi API key untuk Google Gemini
genai.configure(api_key=os.getenv('GEMINI_API_KEY'))

# Dictionary untuk perbaikan kata
fix_words_dict = {
    "put": "putusan",
    "hkmah": "mahkamah",
    "no": "nomor",
    "repub": "republik",
    "paska": "pasca",
    "hkama": "mahkamah",
    "ahkamah": "mahkamah",
    "pt": "pengadilan",
    "tks": "terima kasih",
    "bpk": "bapak",
    "ibu": "ibu",
    "dlm": "dalam",
}

def fix_typo_and_abbr(text, fix_dict):
    if text is None:
        return ''
    words = text.split()
    corrected_words = [fix_dict.get(word, word) for word in words]
    return ' '.join(corrected_words)

def preprocess_text(text):
    if not isinstance(text, str):
        return ''
    # Convert to lowercase
    text = text.lower()

    # Clean up text patterns
    text = re.sub(r'(\d)([a-zA-Z])', r'\1 \2', text)
    text = re.sub(r'([a-zA-Z])(\d)', r'\1 \2', text)
    text = re.sub(r'\b([a-zA-Z])\s([a-zA-Z])\s([a-zA-Z])\s([a-zA-Z])\s([a-zA-Z])\s([a-zA-Z])\b', r'\1\2\3\4\5\6', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\bhalaman\b(\s\d+)*(\shalaman\b)*', '', text)
    text = re.sub(r'\d+\s*(?=\bputusan\b)', '', text)
    text = re.sub(r'\b(berdasarkan ketentuan|menimbang)\b', '', text)

    # Remove stopwords
    stop_words = set(stopwords.words('indonesian'))
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in stop_words]
    
    # Join tokens and clean up spaces
    processed_text = ' '.join(tokens)
    processed_text = re.sub(r'\s+', ' ', processed_text).strip()
    
    # Fix typos and abbreviations
    processed_text = fix_typo_and_abbr(processed_text, fix_words_dict)
    
    return processed_text

def extract_text_from_pdf(pdf_file):
    try:
        doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text()
        return text
    except Exception as e:
        return str(e)

def summarize_text_with_gemini(text):
    try:
        response = genai.GenerativeModel('gemini-1.5-flash').generate_content(
            f"Ringkas dokumen hukum berikut dalam bahasa Indonesia tanpa menggunakan format Markdown, tanpa teks tebal (**), tanpa teks miring (*), atau simbol lainnya: {text}"
        )
        return response.text.replace("**", "").replace("*", "").strip()
    except Exception as e:
        return str(e)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not file.filename.endswith('.pdf'):
        return jsonify({'error': 'Please upload a PDF file'}), 400
    try:
        raw_text = extract_text_from_pdf(file)
        print("Raw text extracted:", raw_text[:100])  # Print first 100 characters of raw text
        
        processed_text = preprocess_text(raw_text)
        print("Processed text:", processed_text[:100])  # Print first 100 characters of processed text
        
        summary = summarize_text_with_gemini(processed_text)
        print("Summary:", summary)
        
        return jsonify({
            'raw_text': raw_text[:1000] + '...',
            'processed_text': processed_text[:1000] + '...',
            'summary': summary
        })
    except Exception as e:
        print("Error:", str(e))  # Print the error message
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
