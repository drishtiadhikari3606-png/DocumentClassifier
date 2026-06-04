DocumentClassifier

An enterprise-grade 20-domain document text classifier built with Python, Flask, and scikit-learn, featuring a custom Lexi AI dark-mode dashboard interface
Lexi AI // Advanced 20-Domain Document Classifier

Lexi AI is an enterprise-grade natural language processing (NLP) application designed to classify complex, unstructured text documents into 20 distinct thematic domains. Built on top of a machine learning pipeline using scikit-learn, the system processes raw text inputs through an optimized TF-IDF vectorization engine and a Multinomial Naive Bayes classifier. The application exposes its predictive capabilities via a sleek, responsive, glassmorphic dark-mode web dashboard powered by Flask and Tailwind CSS.
🚀 Key Features

    Comprehensive 20-Domain Classification: Capable of scanning text and mapping it accurately across a highly overlapping matrix of 20 categories (spanning technology, religion, politics, hardware, and sports).
    Live Probability Vectors: Instead of just outputting a flat prediction, the interface displays a real-time sorted breakdown of the confidence metrics for all 20 categories.
    Intelligent Custom Interceptors: Implements a rule-based override mechanism to catch historical dataset biases (such as resolving modern geopolitical/theological terms like "Hinduism" seamlessly).
    High-Performance Pipeline Optimization: Utilizes an NLTK-driven LemmaTokenizer that morphs words down to their root semantic forms, drastically lowering feature dimensionality while boosting accuracy.
    Modern Glassmorphic UI: A beautifully engineered web workspace with animated status indicators, custom scrolling metric panels, and visual probability bars.

🛠️ The Architecture & Naming System

The machine learning core is trained using a snapshot of early internet communications grouped into specific domain directories. This explains the technical forum-style hierarchy labels displayed on the dashboard, such as:

    comp.os.ms-windows.misc → Computer Operating Systems / Microsoft Windows / Miscellaneous (Troubleshooting, tips, and configuration).
    comp.graphics → Computer Graphics (Rendering, animation, and vector processing).
    talk.religion.misc → Miscellaneous Religious and Theological Debates.
    rec.motorcycles → Recreational Motorcycling and Mechanical Discussions.

📊 How the AI Engine Works Under the Hood

The machine learning execution operates in two independent phases: Training (train.py) and Inference (app.py).
1. Text Preprocessing & Lemmatization

When text enters the pipeline, standard tokenization is insufficient because words like "running", "runs", and "ran" are treated as completely independent features. Lexi AI utilizes an integrated WordNetLemmatizer via NLTK to strip suffixes and map words to their base linguistic dictionary form ("run"). Punctuation and numerical noise are automatically filtered out using alpha-character constraints.
2. Feature Extraction (TF-IDF)

The text is vectorized using Term Frequency-Inverse Document Frequency (TF-IDF). This mathematical approach counts how often a word appears in a specific document but penalizes it if it appears frequently across all documents (like "the", "is", or "and"). This ensures that unique domain-specific keywords carry the heaviest mathematical weight.
3. Classification (Multinomial Naive Bayes)

Using Bayes' Theorem, the classifier calculates the conditional probability of the document belonging to each of the 20 categories based on the distribution of its words. The category yielding the highest posterior probability is chosen as the primary classification.
📁 Project Directory Structure

DocumentClassifier/ │ ├── app.py # Main Flask web application server & inference loop ├── train.py # Machine learning training pipeline script ├── document_classifier.pkl # Serialized pre-trained ML model & target names │ ├── templates/ │ └── index.html # Tailwind CSS glassmorphic frontend UI dashboard │ └── README.md # Comprehensive project documentation
Quick Start Guide (For External Users)

If someone else wants to download and run this project on their computer, they just need to clone/download these files and execute these three basic steps in their terminal:

    Open VS code and Install the Core Dependencies Run this command to download the essential machine learning and web server software libraries your app runs on:

pip install flask sklearn joblib nltk

    Download the Linguistic Dictionaries Because Lexi AI utilizes advanced natural language lemmatization, run this command once to pull down the mandatory language components:

python -c "import nltk; nltk.download('punkt_tab'); nltk.download('wordnet'); nltk.download('stopwords')"

    Launch the Application Dashboard Now the environment is completely ready! Start the local web application server using:

python app.py
