# Sentiment-Analysis-in-Python-VADER-RoBERTa
This project demonstrates sentiment analysis using two approaches on Amazon Food Reviews: Techniques Used VADER (Valence Aware Dictionary and sEntiment Reasoner) A lexicon- and rule-based model using a bag-of-words approach via nltk.  

## Techniques Used

### 1. VADER (Valence Aware Dictionary and sEntiment Reasoner)
- Lexicon and rule-based sentiment analyzer from NLTK.
- Works best for social media and short text.
- Uses a bag-of-words model and outputs `compound`, `pos`, `neu`, and `neg` scores.

### 2. RoBERTa (via Hugging Face Transformers)
- A pretrained deep learning model that captures context.
- Provides more accurate sentiment predictions on longer reviews.

## Project Workflow

1. **Exploratory Data Analysis (EDA):**
   - Load and clean Amazon Fine Food Reviews dataset.
   - Visualize distribution of review scores.

2. **NLTK Preprocessing:**
   - Tokenize and POS tag review text.
   - Use `ne_chunk()` to extract named entities.

3. **VADER Sentiment Scoring:**
   - Apply `SentimentIntensityAnalyzer` to each review.
   - Store `compound` scores for comparison.

4. **RoBERTa Sentiment Classification:**
   - Tokenize and feed review text to RoBERTa.
   - Collect predicted probabilities for `NEGATIVE`, `NEUTRAL`, and `POSITIVE`.

5. **Compare & Visualize Results:**
   - Highlight mismatches between review star rating and predicted sentiment.
   - Plot VADER vs RoBERTa outputs for insight.
