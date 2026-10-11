# Sentiment Analysis of Movie Reviews Using a Multi-Stage NLP Pipeline

A movie review says more than "positive" or "negative". It can be sarcastic, imply a star rating, express specific emotions, and praise the acting while panning the plot. This project builds a **multi-stage NLP pipeline on IMDb reviews** that reads a raw review and reports all of that together:

| Output | How it is produced |
|---|---|
| **Overall polarity** (positive / negative / mixed-uncertain, with a sarcasm flag) | TF-IDF + LinearSVC, cross-checked by DistilBERT and a RoBERTa irony detector, combined by a small stacked meta-classifier |
| **Star rating (1–10)** | Ridge regression on TF-IDF features, trained on star-labelled IMDb reviews |
| **Top-2 emotions** | RoBERTa GoEmotions, filtered so emotions that contradict the polarity are dropped |
| **Aspect-level sentiment** (acting, plot, visuals, music, dialogue, pacing) | Clause-level keyword matching scored by the SVM |
| **Topics in praise vs. complaints** | LDA topic modelling |

It shows how a fast, interpretable classical baseline and pre-trained transformers can complement each other in an explainable review-analysis workflow.

## Highlights

- **91.71% test accuracy** (macro F1 0.92) for the TF-IDF (1–3-gram) LinearSVC on 9,917 unseen reviews.
- **Sarcasm-aware:** the SVM's literal reading is compared with a transformer sentiment model and an irony detector, so "Oh wow, what a masterpiece" is not blindly labelled positive.
- **Stacking improves the hard cases:** on 1,000 held-out reviews, the stacked model beats the SVM alone, most noticeably on borderline reviews (see Results).
- **One call, four answers:** the interactive analyser returns polarity, rating, emotions and aspects for any review you type.
- **Modular:** each stage can be evaluated or replaced independently (e.g. swap in a fine-tuned transformer).

## How it works

```
IMDB_Dataset.csv  ──►  clean (strip HTML, lowercase, drop duplicates)
                          │
                          ▼
        TF-IDF (1–3-grams, min_df=2, max_df=0.9, sublinear tf) + LinearSVC (C=0.5)
                          │                         91.71% test accuracy
                          ▼
   Nuance features per review (split into ~60-word chunks):
     • SVM scores: whole review, mean / min / max, ending, "positive-start → negative-end" turn
     • RoBERTa irony: max, mean, last chunk
     • DistilBERT SST-2: mean, last chunk, spread
                          │
                          ▼
   HistGradientBoosting meta-classifier  →  P(positive)
        ≥ 0.65  → positive      ≤ 0.35 → negative      otherwise → mixed / uncertain
        (label disagrees with the SVM's literal reading → "possible sarcasm")
        (reviews under 15 words → SVM vs. DistilBERT agreement only)

Separate branches on the same review:
   Ridge regression (TF-IDF 1–2-grams, aclImdb star labels)  → rating 1–10
   RoBERTa GoEmotions + polarity filter                      → top-2 emotions
   Aspect keyword sets + clause splitting + SVM              → aspect → sentiment
```

**Data**
- `IMDB_Dataset.csv`: labelled positive/negative reviews (duplicates removed; 80/20 stratified split, 4,940 negative / 4,977 positive in the test set).
- [`aclImdb_v1`](https://ai.stanford.edu/~amaas/data/sentiment/): 50,000 reviews with 1–10 star ratings (rating is encoded in each filename). Downloaded automatically by the notebook.

**Pre-trained models (Hugging Face, used without further training)**
- [`distilbert-base-uncased-finetuned-sst-2-english`](https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english): literal sentiment
- [`cardiffnlp/twitter-roberta-base-irony`](https://huggingface.co/cardiffnlp/twitter-roberta-base-irony): irony detection
- [`SamLowe/roberta-base-go_emotions`](https://huggingface.co/SamLowe/roberta-base-go_emotions): emotion profiling

## Results

| Module | Metric | Result |
|---|---|---|
| LinearSVC polarity | Test accuracy / macro F1 | **91.71% / 0.92** |
| Stacked polarity model | Accuracy on 1,000 held-out reviews (5-fold CV) | **92.5%** vs. 91.6% for the SVM alone |
| Ridge star rating | Mean absolute error (1–10 scale) | ≈ 1.6 stars |
| Emotions, aspects, topics | Qualitative inspection | See examples below |

**Where stacking helps most**

| Subset | SVM alone | Stacked |
|---|---|---|
| All reviews (n = 1,000) | 91.6% | 92.5% |
| Long reviews (> 300 words, n = 243) | 94.2% | 96.3% |
| SVM unsure (\|score\| < 0.3, n = 180) | 71.7% | 76.7% |

**Example**

```
Enter a review: I hated how much I loved this movie. It ruined my entire weekend ...
Overall:  negative
Rating:   5 / 10
Emotion:  disgust, disappointment
Aspects:  {'acting': 'negative', 'plot': 'positive', 'music': 'positive'}
```

Sentiment toward aspects is judged on literal clause polarity, so sarcastic or implicit aspect phrases can be mislabelled. Sarcasm, emotion and aspect outputs are evaluated qualitatively, not with numerical test metrics.

## Getting started

The project is a single Jupyter notebook developed in **Google Colab** (a GPU runtime speeds up the transformer stages).

1. Open `10_1_NLP_Final_SHORT.ipynb` in Colab.
2. Upload `IMDB_Dataset.csv` (Colab Files sidebar) next to the notebook.
3. Install dependencies if running locally:
   ```bash
   pip install scikit-learn pandas numpy joblib torch transformers
   ```
4. Run the cells top to bottom. The Hugging Face models and `aclImdb_v1` download automatically.
5. The last cell starts an interactive loop. Type a review to analyse it, or `quit` to stop.

Saved artefacts: `sentiment_model.joblib` (SVM pipeline) and `sentiment_stack.joblib` (SVM + meta-classifier).

## Tech stack

Python · scikit-learn · Hugging Face Transformers · PyTorch · pandas · NumPy · joblib

## Limitations and future work

- Only polarity and rating have numerical test metrics; a labelled movie-review sarcasm test set is needed to report precision/recall/F1 for sarcasm.
- The irony model is trained on tweets, not movie reviews.
- Keyword-based aspect matching misses implicit aspects and can be fooled by sarcasm inside an aspect phrase. Learned aspect-based sentiment models are the natural upgrade.
- The aclImdb corpus has no 5- or 6-star reviews, so the rating model under-represents middling ratings. Ordinal regression with neutral examples would help.
- Next steps: fine-tune a transformer on IMDb-style text and compare it with the LinearSVC baseline, and serve the pipeline as an API.

## References

1. J. R. Jim et al., "Recent advancements and challenges of NLP-based sentiment analysis: A state-of-the-art review," *Natural Language Processing Journal*, vol. 6, 2024.
2. M. S. Islam et al., "Challenges and future in deep learning for sentiment analysis: a comprehensive review and a proposed novel hybrid approach," *Artificial Intelligence Review*, vol. 57, 2024.
3. P. Atandoh et al., "Scalable deep learning framework for sentiment analysis prediction for online movie reviews," *Heliyon*, vol. 10, no. 10, 2024.
4. S. Ouyang, "Deep learning for sentiment analysis on IMDB movie reviews using N-gram features," *Applied and Computational Engineering*, vol. 35, 2024.
5. H. Liu et al., "Sarcasm driven by sentiment: A sentiment-aware hierarchical fusion network for multimodal sarcasm detection," *Information Fusion*, vol. 108, 2024.
6. S. Khan et al., "A novel transformer attention-based approach for sarcasm detection," *Expert Systems*, vol. 42, no. 1, 2025.
7. S.-V. Oprea and A. Bâra, "LLM-as-a-judge for sarcasm detection using supervised fine-tuning of transformers," *J. King Saud Univ. – Comput. Inf. Sci.*, vol. 37, 2025.
8. Y. C. Hua et al., "A systematic review of aspect-based sentiment analysis: domains, methods, and trends," *Artificial Intelligence Review*, vol. 57, 2024.
9. K. Aziz et al., "Unifying aspect-based sentiment analysis BERT and multi-layered graph convolutional networks for comprehensive sentiment dissection," *Scientific Reports*, vol. 14, 2024.
10. W. Zou et al., "A syntactic features and interactive learning model for aspect-based sentiment analysis," *Complex & Intelligent Systems*, vol. 10, 2024.
11. Z. Su et al., "CGT: A Clause Graph Transformer Structure for aspect-based sentiment analysis," *Data & Knowledge Engineering*, vol. 153, 2024.
12. H. Mao, Y. Fan, and M. Tong, "Research on aspect-based sentiment analysis of movie reviews based on deep learning," *Journal of Information Science*, 2024.
