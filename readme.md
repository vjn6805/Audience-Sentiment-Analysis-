# Audience Sentiment Analysis for Media

A machine learning-powered web application that analyzes sentiment in social media posts, comments, and reviews to understand audience reactions to media content.

## Features

- **Single Text Analysis**: Analyze individual tweets, comments, or reviews
- **Batch Processing**: Upload CSV files for bulk sentiment analysis
- **Interactive Dashboard**: Visual sentiment distribution with charts
- **Real-time Predictions**: Instant sentiment classification
- **Multi-platform Support**: Handles content from Twitter, Instagram, Facebook

## Tech Stack

- **Frontend**: Streamlit
- **Machine Learning**: Scikit-learn (Naive Bayes, TF-IDF)
- **Data Processing**: Pandas, NumPy
- **Visualization**: Matplotlib
- **Model Persistence**: Joblib

## Project Structure

```
AIML Lab Project/
├── frontend.py           # Streamlit web application
├── train_model.py        # Model training script
├── project.ipynb         # Jupyter notebook for analysis
├── sentimentdataset.csv  # Training dataset
├── sentiment_model.pkl   # Trained ML model
├── vectorizer.pkl        # TF-IDF vectorizer
└── README.md            # Project documentation
```

## Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd your_project_file   
   ```

2. **Install dependencies**
   ```bash
   pip install streamlit pandas scikit-learn matplotlib joblib
   ```

3. **Train the model** (if pkl files don't exist)
   ```bash
   python train_model.py
   ```

## Usage

### Running the Web Application
```bash
streamlit run frontend.py
```

### Training a New Model
```bash
python train_model.py
```

### Using Jupyter Notebook
```bash
jupyter notebook project.ipynb
```

## Model Details

- **Algorithm**: Multinomial Naive Bayes
- **Vectorization**: TF-IDF (5000 features, English stop words removed)
- **Classes**: Positive, Negative, Neutral
- **Text Preprocessing**: 
  - Lowercase conversion
  - URL/mention removal
  - Punctuation removal
  - Whitespace normalization

## Dataset

The model is trained on social media data containing:
- **Text**: Social media posts/comments
- **Sentiment**: Labeled sentiments (positive/negative/neutral)
- **Platform**: Source platform (Twitter, Instagram, Facebook)
- **Engagement**: Likes, retweets, shares
- **Geography**: Country information

## Use Cases

- **Media Companies**: Analyze audience reaction to content
- **Marketing Teams**: Monitor campaign sentiment
- **Social Media Managers**: Track brand perception
- **Content Creators**: Understand audience feedback
- **Researchers**: Study social media sentiment trends

## How It Works

1. **Text Preprocessing**: Clean and normalize input text
2. **Vectorization**: Convert text to TF-IDF features
3. **Classification**: Predict sentiment using Naive Bayes
4. **Visualization**: Display results with interactive charts

## Input Formats

### Single Text Input
- Enter any text in the web interface
- Get instant sentiment prediction with emoji indicators

### CSV Upload
- CSV file must contain a 'Text' column
- Supports batch processing of multiple texts
- Generates sentiment summary and detailed results

## Web Interface

- **Clean Design**: User-friendly Streamlit interface
- **Emoji Indicators**: Visual sentiment representation (😊😐😠)
- **Interactive Charts**: Bar charts for sentiment distribution
- **Data Tables**: Detailed results with sample predictions

## Customization

### Adding New Sentiment Categories
Modify the `map_sentiment()` function in `train_model.py`:
```python
def map_sentiment(s):
    # Add your custom sentiment mappings
    if s in ['custom_positive_terms']:
        return 'positive'
    # ... rest of the function
```

### Adjusting Model Parameters
In `train_model.py`, modify:
```python
vectorizer = TfidfVectorizer(
    max_features=5000,  # Adjust feature count
    stop_words='english'  # Change language
)
```

## Performance Metrics

The model provides:
- **Accuracy Score**: Overall prediction accuracy
- **Classification Report**: Precision, recall, F1-score per class
- **Confusion Matrix**: Detailed performance breakdown


---

*Built with ❤️ for understanding audience sentiment in the digital age*
