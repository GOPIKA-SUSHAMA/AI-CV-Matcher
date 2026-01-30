# AI-CV-Matcher


This project is an AI-powered web application that semantically matches a candidate’s CV text with a job description using NLP.

The system uses transformer-based embeddings to compute a similarity score between CV and job posting, producing a match score with a simple “Strong / Moderate / Weak” indicator.

 Overview

- Built using **Streamlit** for web UI
- Uses **Sentence-BERT embeddings** for semantic similarity
- Can be run online in **Google Colab** with **ngrok** for a temporary website
- Inspired by research on semantic similarity in recruitment systems:
  
  Ajjam, M.-H. & Al-Raweshidy, H. (2026). *AI-driven semantic similarity-based job matching framework for recruitment systems.* Information Sciences.  
  This paper discusses semantic similarity embeddings to match candidates to jobs, which inspired the approach used in this project.


 Features

- Paste candidate CV text
- Paste job description text
- Get a semantic match score (0–1)
- Score indicator: Strong / Moderate / Weak
- Temporary website via Colab + ngrok

 Usage (Colab Setup)

1. Open `colab_demo.ipynb` in **Google Colab**  
2. Install libraries:

```python
!pip install streamlit sentence-transformers torch pyngrok

