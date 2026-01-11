# AI Scientist Roadmap

A complete path from zero to AI Scientist. Free resources, hands-on projects, and clear milestones.

```
🌱 Beginner → 🌿 Intermediate → 🌳 Advanced → 🔬 Scientist
```

## How to Use This Roadmap

1. Follow the phases in order
2. Complete the projects before moving on
3. Mark your progress with checkboxes
4. Estimated time: 12-18 months (2-3 hours/day)

---

## Phase 0: Prerequisites (4-6 weeks)

> Foundation before AI. Don't skip this.

### Mathematics

- [ ] **Linear Algebra** - Vectors, matrices, eigenvalues
  - [3Blue1Brown: Essence of Linear Algebra](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab) (free)
  - [MIT 18.06 Linear Algebra](https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/) (free)

- [ ] **Calculus** - Derivatives, gradients, chain rule
  - [3Blue1Brown: Essence of Calculus](https://www.youtube.com/playlist?list=PLZHQObOWTQDMsr9K-rj53DwVRMYO3t5Yr) (free)

- [ ] **Probability & Statistics** - Distributions, Bayes theorem, hypothesis testing
  - [Khan Academy Statistics](https://www.khanacademy.org/math/statistics-probability) (free)
  - [StatQuest YouTube](https://www.youtube.com/c/joshstarmer) (free)

### Programming

- [ ] **Python Fundamentals**
  - [Python Official Tutorial](https://docs.python.org/3/tutorial/) (free)
  - [Automate the Boring Stuff](https://automatetheboringstuff.com/) (free)

- [ ] **NumPy & Pandas**
  - [NumPy Official Tutorial](https://numpy.org/doc/stable/user/quickstart.html) (free)
  - [Pandas Getting Started](https://pandas.pydata.org/docs/getting_started/) (free)

### 🎯 Phase 0 Project
Build a data analysis project: Load a dataset, clean it, visualize insights, compute statistics.

---

## Phase 1: Machine Learning Foundations (8-10 weeks)

> Understanding the classics before deep learning.

### Core Concepts

- [ ] **Supervised Learning** - Regression, Classification
- [ ] **Unsupervised Learning** - Clustering, Dimensionality Reduction
- [ ] **Model Evaluation** - Train/test split, cross-validation, metrics

### Algorithms to Master

| Algorithm | Type | Learn It |
|-----------|------|----------|
| Linear Regression | Supervised | Week 1-2 |
| Logistic Regression | Supervised | Week 2-3 |
| Decision Trees | Supervised | Week 3-4 |
| Random Forests | Ensemble | Week 4-5 |
| K-Means | Unsupervised | Week 5-6 |
| PCA | Dimensionality | Week 6-7 |
| SVM | Supervised | Week 7-8 |

### Resources

- [ ] [Scikit-learn Tutorials](https://scikit-learn.org/stable/tutorial/index.html) (free)
- [ ] [Google ML Crash Course](https://developers.google.com/machine-learning/crash-course) (free)
- [ ] [StatQuest ML Playlist](https://www.youtube.com/playlist?list=PLblh5JKOoLUICTaGLRoHQDuF_7q2GfuJF) (free)

### 📚 Books

- "Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow" - Aurélien Géron
- "The Hundred-Page Machine Learning Book" - Andriy Burkov (free online)

### 🎯 Phase 1 Projects

1. **Predict house prices** - Regression with feature engineering
2. **Classify iris species** - Multi-class classification
3. **Customer segmentation** - K-means clustering
4. **Build ML pipeline** - End-to-end with scikit-learn

---

## Phase 2: Deep Learning (10-12 weeks)

> Neural networks and the modern AI stack.

### Fundamentals

- [ ] **Neural Network Basics** - Perceptrons, activation functions, backpropagation
- [ ] **Optimization** - SGD, Adam, learning rate scheduling
- [ ] **Regularization** - Dropout, batch normalization, early stopping

### Architectures

| Architecture | Use Case | Weeks |
|--------------|----------|-------|
| MLP | Tabular data | 1-2 |
| CNN | Images | 3-4 |
| RNN/LSTM | Sequences | 5-6 |
| Transformers | Text, everything | 7-10 |

### Frameworks

- [ ] **PyTorch** (recommended)
  - [PyTorch Official Tutorials](https://pytorch.org/tutorials/) (free)
  - [Deep Learning with PyTorch](https://pytorch.org/assets/deep-learning/Deep-Learning-with-PyTorch.pdf) (free book)

- [ ] **TensorFlow/Keras** (alternative)
  - [TensorFlow Tutorials](https://www.tensorflow.org/tutorials) (free)

### Resources

- [ ] [Fast.ai Course](https://course.fast.ai/) (free) - Practical deep learning
- [ ] [CS231n: CNNs for Visual Recognition](http://cs231n.stanford.edu/) (free)
- [ ] [CS224n: NLP with Deep Learning](http://web.stanford.edu/class/cs224n/) (free)

### 📚 Books

- "Deep Learning" - Goodfellow, Bengio, Courville (free online)
- "Dive into Deep Learning" - d2l.ai (free online, interactive)

### 🎯 Phase 2 Projects

1. **Image classifier** - CNN on CIFAR-10 or custom dataset
2. **Sentiment analysis** - RNN/LSTM on movie reviews
3. **Neural style transfer** - Artistic image transformation
4. **Train a small transformer** - From scratch understanding

---

## Phase 3: Specialization Tracks (12-16 weeks)

> Choose your path. You can do multiple.

### Track A: Computer Vision

- [ ] Object Detection (YOLO, Faster R-CNN)
- [ ] Image Segmentation (U-Net, Mask R-CNN)
- [ ] Generative Models (GANs, Diffusion)
- [ ] Video Understanding

**Projects:**
- Real-time object detection system
- Image generation with Stable Diffusion
- Face recognition pipeline

### Track B: Natural Language Processing

- [ ] Word Embeddings (Word2Vec, GloVe)
- [ ] Transformer Architecture (deep dive)
- [ ] Large Language Models (GPT, BERT, LLaMA)
- [ ] Prompt Engineering & Fine-tuning

**Projects:**
- Build a chatbot with fine-tuned model
- Document Q&A system with RAG
- Sentiment analysis API

### Track C: Reinforcement Learning

- [ ] MDPs and Bellman Equations
- [ ] Q-Learning, DQN
- [ ] Policy Gradients, A2C, PPO
- [ ] Multi-Agent RL

**Resources:**
- [Spinning Up in Deep RL](https://spinningup.openai.com/) (free, OpenAI)
- [David Silver's RL Course](https://www.davidsilver.uk/teaching/) (free)

**Projects:**
- Train agent to play Atari games
- Robot simulation with MuJoCo
- Multi-agent environment

### Track D: AI Safety & Alignment

- [ ] Alignment Problem Overview
- [ ] RLHF (Reinforcement Learning from Human Feedback)
- [ ] Interpretability & Explainability
- [ ] Red-teaming and adversarial attacks

**Resources:**
- [AI Safety Fundamentals](https://www.aisafetyfundamentals.com/) (free)
- [Anthropic Research Papers](https://www.anthropic.com/research)

---

## Phase 4: Research Skills (8-12 weeks)

> From practitioner to scientist.

### Reading Papers

- [ ] Learn to read papers efficiently
  - [How to Read a Paper](https://web.stanford.edu/class/ee384m/Handouts/HowtoReadPaper.pdf) (free)
- [ ] Follow arXiv daily (cs.AI, cs.LG, cs.CL)
- [ ] Use [Papers With Code](https://paperswithcode.com/)

### Reproducing Research

- [ ] Pick 3 papers and reproduce results
- [ ] Document discrepancies
- [ ] Understand ablation studies

### Experimentation

- [ ] Hyperparameter tuning (Optuna, Weights & Biases)
- [ ] Experiment tracking (MLflow, W&B)
- [ ] Statistical significance testing

### Writing

- [ ] Learn LaTeX
- [ ] Understand paper structure (Abstract, Intro, Method, Results, Discussion)
- [ ] Write a technical blog post

### 🎯 Phase 4 Projects

1. **Reproduce a paper** - Pick SOTA from 2022-2023, reproduce it
2. **Improve on it** - Add your own modification, measure impact
3. **Write it up** - Blog post or arXiv preprint

---

## Phase 5: Original Research (Ongoing)

> Contributing new knowledge.

### Finding Research Questions

- [ ] Identify gaps in existing work
- [ ] Start with "What if...?" questions
- [ ] Build on recent papers

### Research Workflow

```
Idea → Literature Review → Hypothesis → Experiment → Analysis → Write → Peer Review
```

### Collaboration

- [ ] Join research communities (Discord, Twitter/X)
- [ ] Attend conferences (NeurIPS, ICML, ICLR, ACL)
- [ ] Contribute to open source research

### Publication Venues

| Type | Examples |
|------|----------|
| Top Conferences | NeurIPS, ICML, ICLR, CVPR, ACL, EMNLP |
| Journals | JMLR, TMLR, Nature Machine Intelligence |
| Preprints | arXiv |
| Workshops | Conference workshops (easier entry) |

---

## Tools & Environment

### Essential Setup

```bash
# Environment
python >= 3.9
conda or venv
jupyter notebook/lab
git

# Core libraries
numpy, pandas, matplotlib, seaborn
scikit-learn
pytorch or tensorflow

# Experiment tracking
wandb or mlflow

# Paper writing
LaTeX (Overleaf)
```

### Compute Resources

| Resource | Type | Cost |
|----------|------|------|
| Google Colab | GPU | Free tier available |
| Kaggle Notebooks | GPU | Free |
| Lambda Labs | Cloud GPU | Paid |
| Vast.ai | Cloud GPU | Cheap |
| Your own GPU | Local | RTX 3090+ recommended |

---

## Timeline Overview

| Phase | Duration | Focus |
|-------|----------|-------|
| 0 | 4-6 weeks | Math + Python |
| 1 | 8-10 weeks | Classical ML |
| 2 | 10-12 weeks | Deep Learning |
| 3 | 12-16 weeks | Specialization |
| 4 | 8-12 weeks | Research Skills |
| 5 | Ongoing | Original Research |

**Total: 12-18 months to research-ready**

---

## Contributing

Found a great resource? PRs welcome!

## License

[![CC0](https://licensebuttons.net/p/zero/1.0/88x31.png)](https://creativecommons.org/publicdomain/zero/1.0/)

---

*Start today. The best time to begin was yesterday. The second best time is now.*
