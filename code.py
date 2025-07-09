import os
import csv
import re
from bs4 import BeautifulSoup
import justext
import trafilatura
from readability import Document
from rouge_score import rouge_scorer
import requests 
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
import numpy as np
from collections import Counter
import math
import time

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

ROOT_DIR = "data/website-data"
SCORE_FILE = "evaluations/enhanced_qualitative_scores.csv"
METHODS = ['jusText', 'Trafilatura', 'Readability', 'Baseline', 'Ollama']

# Initialize scorers
rouge_scorer_obj = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

# --- Extraction methods ---
def extract_justext(html):
    paragraphs = justext.justext(html, justext.get_stoplist("German"))
    return ' '.join(p.text for p in paragraphs if not p.is_boilerplate)

def extract_trafilatura(html):
    result = trafilatura.extract(html)
    return result if result else ""

def extract_readability(html):
    doc = Document(html)
    summary_html = doc.summary()
    return BeautifulSoup(summary_html, "html.parser").get_text()

def extract_baseline(html):
    soup = BeautifulSoup(html, 'html.parser')
    title = soup.title.string if soup.title else ''
    paragraphs = ' '.join(p.get_text() for p in soup.find_all('p'))
    return title + ' ' + paragraphs

def extract_ollama(html, max_retries=10, delay_seconds=5):
    """
    Attempt to extract content using Ollama with retries.
    """
    prompt_template = (
        "You are an expert at extracting clean and relevant content from HTML pages. "
        "Given the following HTML document, extract only the main article or business description text. "
        "Ignore boilerplate such as navigation menus, footers, ads, links, and legal disclaimers.\n\n"
        "HTML:\n"
        f"{html[:5000]}"
    )

    for attempt in range(1, max_retries + 1):
        try:
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": "llama3.2:latest",
                    "prompt": prompt_template,
                    "stream": False
                },
                timeout=30
            )

            if response.status_code == 200:
                result = response.json()
                content = result.get("response", "").strip()
                if content:
                    return content
            print(f"⚠️ Attempt {attempt}: Ollama response was empty or failed with status {response.status_code}")
        except requests.exceptions.ConnectionError:
            print(f"⚠️ Attempt {attempt}: Ollama server is not running.")
        except Exception as e:
            print(f"⚠️ Attempt {attempt}: Ollama failed with exception: {e}")
        
        if attempt < max_retries:
            print(f"🔁 Retrying in {delay_seconds} seconds...")
            time.sleep(delay_seconds)

    print("❌ Ollama extraction failed after maximum retries.")
    return ""

# --- Enhanced evaluation metrics ---

def calculate_rouge_scores(reference, candidate):
    """Calculate ROUGE-1, ROUGE-2, and ROUGE-L scores"""
    if not candidate or not reference:
        return {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0}
    
    scores = rouge_scorer_obj.score(reference, candidate)
    return {
        'rouge1': scores['rouge1'].fmeasure,
        'rouge2': scores['rouge2'].fmeasure, 
        'rougeL': scores['rougeL'].fmeasure
    }


def calculate_precision_recall_f1(reference, candidate):
    """Calculate precision, recall, and F1 score based on word overlap"""
    if not candidate or not reference:
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
    
    ref_words = set(word_tokenize(reference.lower()))
    cand_words = set(word_tokenize(candidate.lower()))
    
    if not cand_words:
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
    
    intersection = ref_words.intersection(cand_words)
    
    precision = len(intersection) / len(cand_words) if cand_words else 0.0
    recall = len(intersection) / len(ref_words) if ref_words else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {'precision': precision, 'recall': recall, 'f1': f1}

def calculate_content_noise_ratio(text, html_content=None):
    """
    Enhanced content-to-noise ratio calculation
    Considers multiple noise indicators
    """
    if not text:
        return 0.0
    
    total_words = len(text.split())
    if total_words == 0:
        return 0.0
    
    # Noise indicators
    noise_patterns = [
        r'\b(impressum|kontakt|agb|datenschutz|privacy|cookie|terms)\b',  # Legal terms
        r'\b(href=|www\.|http|@|\.(com|de|org))\b',  # URLs and links
        r'\b(navigation|menu|footer|header|sidebar)\b',  # Layout elements
        r'\b(advertisement|ads|banner|sponsored)\b',  # Ads
        r'\b(login|register|subscribe|newsletter)\b',  # User actions
        r'\b(share|like|tweet|facebook|twitter)\b',  # Social media
        r'\b(copyright|©|®|™)\b',  # Copyright symbols
    ]
    
    noise_count = 0
    for pattern in noise_patterns:
        noise_count += len(re.findall(pattern, text.lower()))
    
    # Additional structural noise indicators
    repeated_chars = len(re.findall(r'(.)\1{3,}', text))  # Repeated characters
    excessive_punctuation = len(re.findall(r'[!?]{2,}|\.{3,}', text))  # Multiple punctuation
    
    total_noise = noise_count + repeated_chars + excessive_punctuation
    
    # Content-to-noise ratio (higher is better)
    content_ratio = max(0.0, 1.0 - (total_noise / total_words))
    return content_ratio

def calculate_semantic_coherence(text):
    """
    Calculate semantic coherence based on sentence structure and vocabulary diversity
    """
    if not text:
        return 0.0
    
    try:
        sentences = sent_tokenize(text)
        if len(sentences) < 2:
            return 0.5  # Neutral score for single sentences
        
        words = word_tokenize(text.lower())
        if len(words) < 10:
            return 0.3  # Low score for very short texts
        
        # Vocabulary diversity (Type-Token Ratio)
        unique_words = len(set(words))
        ttr = unique_words / len(words)
        
        # Average sentence length (indicator of complexity)
        avg_sentence_length = len(words) / len(sentences)
        
        # Normalize scores
        ttr_score = min(ttr * 2, 1.0)  # Scale TTR
        length_score = min(avg_sentence_length / 20, 1.0)  # Normalize to reasonable range
        
        # Combined coherence score
        coherence = (ttr_score + length_score) / 2
        return coherence
        
    except:
        return 0.0

def calculate_comprehensive_scores(reference, candidate, method, company):
    """Calculate all evaluation metrics"""
    if not candidate:
        candidate = ""
    if not reference:
        reference = ""
    
    # ROUGE scores
    rouge_scores = calculate_rouge_scores(reference, candidate)
    
    
    
    # Precision, Recall, F1
    prf_scores = calculate_precision_recall_f1(reference, candidate)
    
    # Content-to-noise ratio
    noise_ratio = calculate_content_noise_ratio(candidate)
    
    # Semantic coherence
    coherence = calculate_semantic_coherence(candidate)
    
    # Convert to 0-5 scale for consistency
    def to_score_scale(x, scale=5):
        return round(x * scale, 3)
    
    metrics = {
        'rouge1': to_score_scale(rouge_scores['rouge1']),
        'rouge2': to_score_scale(rouge_scores['rouge2']),
        'rougeL': to_score_scale(rouge_scores['rougeL']),
        'precision': to_score_scale(prf_scores['precision']),
        'recall': to_score_scale(prf_scores['recall']),
        'f1_score': to_score_scale(prf_scores['f1']),
        'content_noise_ratio': to_score_scale(noise_ratio),
        'semantic_coherence': to_score_scale(coherence)
    }
    
    print(f"[{company} - {method}] ROUGE-L={metrics['rougeL']:.2f}, "f" F1={metrics['f1_score']:.2f}, "f"Noise={metrics['content_noise_ratio']:.2f}")
    
    return metrics

# --- File handling functions ---
def get_html_txt_pairs(folder_path):
    files = os.listdir(folder_path)
    html_files = {os.path.splitext(f)[0]: f for f in files if f.endswith(".html")}
    txt_files = {os.path.splitext(f)[0]: f for f in files if f.endswith(".txt")}
    common_keys = set(html_files.keys()).intersection(txt_files.keys())
    return [(html_files[k], txt_files[k], k) for k in common_keys]

def load_pair(folder_path, html_file, txt_file):
    with open(os.path.join(folder_path, html_file), 'r', encoding='utf-8') as f:
        html = f.read()
    with open(os.path.join(folder_path, txt_file), 'r', encoding='utf-8') as f:
        reference = f.read()
    return html, reference.strip().replace('\n', ' ')

def save_enhanced_scores(company, method, metrics):
    """Save comprehensive evaluation scores to CSV"""
    os.makedirs("evaluations", exist_ok=True)
    
    header = ['Company', 'Method', 'ROUGE-1', 'ROUGE-2', 'ROUGE-L',  
              'Precision', 'Recall', 'F1-Score', 'Content-Noise-Ratio', 'Semantic-Coherence']
    
    file_exists = os.path.isfile(SCORE_FILE)
    
    with open(SCORE_FILE, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(header)
        
        row = [company, method,
               metrics['rouge1'], metrics['rouge2'], metrics['rougeL'],
               metrics['precision'], metrics['recall'],
               metrics['f1_score'], metrics['content_noise_ratio'], 
               metrics['semantic_coherence']]
        writer.writerow(row)

# --- Main execution function ---
def run_comprehensive_evaluation():
    """Run evaluation with enhanced metrics on all data pairs"""
    folders = [d for d in os.listdir(ROOT_DIR) if os.path.isdir(os.path.join(ROOT_DIR, d))]
    
    print("Starting comprehensive content extraction evaluation...")
    print(f"Using  metrics: ROUGE,  Precision/Recall/F1, Content-Noise Ratio, Semantic Coherence")
    


    for folder in folders:
        folder_path = os.path.join(ROOT_DIR, folder)
        try:
            pairs = get_html_txt_pairs(folder_path)
            print(f"\nProcessing folder: {folder} ({len(pairs)} pairs)")

            for html_file, txt_file, base_name in pairs:
            

                html, reference = load_pair(folder_path, html_file, txt_file)

                # Extract content using all methods
                extraction_results = {
                    'jusText': extract_justext(html),
                    'Trafilatura': extract_trafilatura(html),
                    'Readability': extract_readability(html),
                    'Baseline': extract_baseline(html),
                    'Ollama': extract_ollama(html),
                }

                # Evaluate each method
                for method in METHODS:
                    result = extraction_results[method]
                    if result is None:
                        result = ""

                    metrics = calculate_comprehensive_scores(reference, result, method, base_name)
                    save_enhanced_scores(base_name, method, metrics)

                print(f"Completed evaluation for {base_name}")


        except Exception as e:
            print(f"Error processing folder {folder}: {e}")
    
    print(f"\n Evaluation complete! Results saved to: {SCORE_FILE}")
   

# --- Execution ---
if __name__ == "__main__":
    run_comprehensive_evaluation()

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from math import pi

# Configuration
CSV_PATH = "evaluations/enhanced_qualitative_scores.csv"
SELECTED_METRICS = ['ROUGE-L', 'F1-Score', 'Content-Noise-Ratio', 'Semantic-Coherence']
ALL_METRICS = ['ROUGE-1', 'ROUGE-2', 'ROUGE-L', 'F1-Score', 'Content-Noise-Ratio', 'Semantic-Coherence']
OUTPUT_DIR = "figures"

# Style for LaTeX export
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("colorblind")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load and preprocess data
df = pd.read_csv(CSV_PATH)

# Remove rows where all metric values are 0
df = df[~(df[SELECTED_METRICS] == 0).all(axis=1)]

# 1. Selected Metric Boxplots
def create_selected_metric_boxplots(df):
    fig, axes = plt.subplots(1, len(SELECTED_METRICS), figsize=(20, 5))

    for i, metric in enumerate(SELECTED_METRICS):
        sns.boxplot(data=df, x="Method", y=metric, ax=axes[i])
        axes[i].set_title(metric, fontsize=12, fontweight='bold')
        axes[i].set_ylabel('Score (0–5)')
        axes[i].tick_params(axis='x', rotation=30)
        axes[i].grid(True, alpha=0.3)

        # Add red dots for method means
        means = df.groupby("Method")[metric].mean()
        for j, method in enumerate(means.index):
            axes[i].scatter(j, means[method], color='red', s=50, zorder=3)

    plt.suptitle('Boxplots for Key Evaluation Metrics', fontsize=14, fontweight='bold')
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, "boxplots_selected_metrics.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()

# 2. Radar Chart
def create_radar_chart(df):
    method_scores = df.groupby("Method")[ALL_METRICS].mean()
    categories = ALL_METRICS
    N = len(categories)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

    for method in method_scores.index:
        values = method_scores.loc[method].tolist()
        values += values[:1]
        ax.plot(angles, values, linewidth=2, label=method)
        ax.fill(angles, values, alpha=0.25)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.set_ylim(0, 5)
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.set_yticklabels(['1', '2', '3', '4', '5'])
    ax.grid(True)

    plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1.05))
    plt.title('Method Performance Across All Metrics', size=16, fontweight='bold', pad=20)
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, "radar_chart_methods.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()

# Run visualizations
if __name__ == "__main__":
    create_selected_metric_boxplots(df)
    create_radar_chart(df)
