# Mental Health Social Media Topic Modelling

> **Author:** Rohit Malik | **Language:** Python | **Type:** Academic Project | **Purpose:** Portfolio & Educational

***

## Overview

A complete **NLP pipeline** for analysing mental health-related discussions on social media. The project applies topic modelling and clustering techniques to identify hidden themes in Twitter posts related to conditions such as anxiety and depression.

Both probabilistic and clustering-based approaches are implemented and compared, demonstrating a thorough understanding of unsupervised machine learning and text analytics.

***

## Project Purpose

The goal of this project is to extract meaningful topics from unstructured social media text and understand patterns in how mental health is discussed online.

By applying LDA and K-Means across a real-world dataset, the project evaluates how different modelling techniques perform in identifying coherent and interpretable themes from noisy, short-form text data.

***

## Key Features

| Feature | Description |
|---|---|
| **Text Preprocessing** | Cleaning, tokenization, stopword removal, and lemmatization |
| **Feature Extraction** | TF-IDF vectorization for representing text numerically |
| **Topic Modelling (LDA)** | Latent Dirichlet Allocation to identify probabilistic topic distributions |
| **Clustering (K-Means)** | Groups posts into clusters based on semantic similarity |
| **Model Evaluation** | Perplexity score for LDA and Silhouette score for K-Means |
| **Dimensionality Reduction** | t-SNE for 2D visualisation of high-dimensional clusters |
| **Word Cloud Generation** | Visual representation of dominant terms per topic |
| **Labelled Output** | Produces a topic-labelled dataset as final output |

***

## Repository Structure

```
mental-health-social-media-topic-modelling/
│
├── README.md
│
├── data/
│   ├── Mental-Health-Twitter.csv          # Raw dataset from Kaggle
│   └── final_topic_results.csv            # Output dataset with topic labels
│
├── notebooks/
│   └── topic_modeling_analysis.ipynb      # Full analysis and results
│
├── src/
│   └── topic_modeling_pipeline.py         # Modular pipeline script
│
└── reports/
    └── topic_modeling_report.docx         # Written analysis and findings
```

**Dataset:** [Depression: Twitter Dataset + Feature Extraction — Kaggle](https://www.kaggle.com/datasets/infamouscoder/mental-health-social-media)

***

## Technologies Used






| Category | Libraries / Tools |
|---|---|
| **Data Handling** | Python, Pandas, NumPy |
| **NLP** | NLTK, WordCloud |
| **Machine Learning** | Scikit-learn (LDA, K-Means, TF-IDF, t-SNE) |
| **Visualisation** | Matplotlib, Seaborn, Plotly |

***

## Key Concepts Demonstrated

| Concept | Application |
|---|---|
| **Natural Language Processing** | Full text preprocessing pipeline on real-world social media data |
| **Topic Modelling (LDA)** | Probabilistic identification of hidden topics across 20K+ posts |
| **Clustering (K-Means)** | Grouping posts by semantic similarity with silhouette evaluation |
| **Feature Engineering** | TF-IDF representation for downstream ML tasks |
| **Model Evaluation** | Comparing LDA perplexity and K-Means silhouette scores |
| **Data Visualisation** | t-SNE cluster plots, word clouds, and topic distribution charts |

***

## Results & Insights

- Identified 3 major topic clusters in mental health discussions
- Topic categories included emotional expression, clinical terminology, and casual social interactions
- Achieved average topic confidence of ~0.79, indicating strong model reliability
- LDA performed best at 3 topics based on lowest perplexity score
- K-Means clustering showed moderate separation with silhouette score evaluation
- t-SNE visualisations confirmed distinct but overlapping topic structures

These results demonstrate how NLP can be used to extract meaningful insights from unstructured social media data related to mental health.

## My Role

The data analysis and modelling aspects of this project were my primary focus. This included cleaning and preprocessing raw Twitter data, applying NLP techniques, building and comparing topic models, and visualising the results.

This project developed my ability to work with unstructured text data at scale, apply unsupervised machine learning techniques, evaluate competing models, and communicate findings through clear visualisations.

***

## Usage & Credit

This project is shared for **portfolio and educational purposes**. If you use or reference this work, please provide appropriate credit to the author.

***

## Author

**Rohit Malik**
- Email: rohitmalik180904@gmail.com
- GitHub: [RohitMalik7](https://github.com/RohitMalik7)
