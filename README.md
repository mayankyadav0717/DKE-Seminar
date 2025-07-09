# Main Content Extraction from Business Web Pages

## An Empirical Evaluation of Heuristic and ML-Based Methods

This repository contains the code and data supporting the seminar project **“Main Content Extraction from Business Web Pages: An Empirical Evaluation of Heuristic and ML-Based Methods”**, completed as part of the Data and Knowledge Engineering program at Otto von Guericke University Magdeburg.

---

## 📚 Overview

Business websites often contain complex layouts, embedded legal content, and multilingual elements that make extracting meaningful main text challenging. This project systematically evaluates five content extraction techniques:

* **jusText** – heuristic, language-aware block filtering
* **Trafilatura** – flexible multi-strategy extractor
* **Readability** – DOM-centric structural extraction
* **Ollama (Llama3)** – local LLM-based semantic extraction
* **Baseline** – simple tag-based extraction using BeautifulSoup

We benchmark these methods on a curated dataset of German business websites, using comprehensive metrics:

* ROUGE-1, ROUGE-2, ROUGE-L (semantic overlap)
* Precision, Recall, F1-Score (content relevance)
* Content-to-Noise Ratio (boilerplate suppression)
* Semantic Coherence (vocabulary diversity & sentence structure)

---

## 📝 Structure

```
.
├── code.py                  # Main script: extraction, evaluation, metric computation, CSV logging
├── new_code.ipynb           # Notebook: data exploration, visualization (boxplots, radar charts)
├── requirements.txt         # Python dependencies
├── Paper_Draft.pdf          # Research paper (final)
├── evaluations/             # Output CSV files with detailed metrics
├── figures/                 # Generated plots
└── data/website-data/       # HTML and reference text files (not included here)
```

---

## ⚙️ Installation

Create a virtual environment (recommended) and install dependencies:

```bash
pip install -r requirements.txt
```

Requirements include:

* beautifulsoup4
* justext
* trafilatura
* readability-lxml
* rouge-score
* requests
* nltk
* pandas, matplotlib, seaborn (for visualization)

---

## 🚀 Running the Evaluation

To perform extraction and generate metrics:

```bash
python code.py
```

This script:

* Extracts content using all five methods.
* Compares each output to manual references.
* Computes ROUGE, precision, recall, F1, content-noise ratio, semantic coherence.
* Saves results to `evaluations/enhanced_qualitative_scores.csv`.

---

## 📊 Visualization

To produce summary boxplots and radar charts:

```bash
python new_code.ipynb
```

Outputs are saved in the `figures/` directory.

---

## 🔍 Results Summary

| Method      | ROUGE-L  | F1-Score | C-N Ratio | Sem Coherence |
| ----------- | -------- | -------- | --------- | ------------- |
| Baseline    | **2.92** | 4.67     | 4.23      | **4.49**      |
| Trafilatura | 2.56     | **4.73** | 4.21      | 4.36          |
| jusText     | 2.32     | 4.12     | 3.67      | 3.54          |
| Readability | 1.71     | 4.58     | 3.96      | 3.82          |
| Ollama      | 0.32     | **4.74** | **4.52**  | 4.44          |

*Higher scores indicate better performance across all metrics.*

---

## ✍️ Citation & License

If you use this project or its metrics framework, please cite the seminar report:

> Mayank Yadav.
> *Main Content Extraction from Business Web Pages: An Empirical Evaluation of Heuristic and ML-Based Methods.*
> Otto von Guericke University Magdeburg, 2025.

This repository is shared for academic and educational purposes. Please contact [mayank.yadav@st.ovgu.de](mailto:mayank.yadav@st.ovgu.de) for reuse or collaboration.

---

## 🤝 Acknowledgments

* Libraries: BeautifulSoup, jusText, Trafilatura, Readability, Rouge-Score, NLTK
* Seminar supervisor: \[Name, if applicable]

---

✅ If you’d like, I can also create:

* a **shorter minimal README**,
* or add **example commands + sample outputs**.
  Just tell me what style you prefer!
