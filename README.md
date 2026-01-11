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
  - [MIT 18.06 Linear Algebra - Gilbert Strang](https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/) (free)
  - [Khan Academy Linear Algebra](https://www.khanacademy.org/math/linear-algebra) (free)
  - [Interactive Linear Algebra](https://textbooks.math.gatech.edu/ila/) (free book)

- [ ] **Calculus** - Derivatives, gradients, chain rule
  - [3Blue1Brown: Essence of Calculus](https://www.youtube.com/playlist?list=PLZHQObOWTQDMsr9K-rj53DwVRMYO3t5Yr) (free)
  - [MIT 18.01 Single Variable Calculus](https://ocw.mit.edu/courses/18-01sc-single-variable-calculus-fall-2010/) (free)
  - [Khan Academy Calculus](https://www.khanacademy.org/math/calculus-1) (free)

- [ ] **Probability & Statistics** - Distributions, Bayes theorem, hypothesis testing
  - [Khan Academy Statistics](https://www.khanacademy.org/math/statistics-probability) (free)
  - [StatQuest YouTube](https://www.youtube.com/c/joshstarmer) (free)
  - [MIT 18.05 Probability and Statistics](https://ocw.mit.edu/courses/18-05-introduction-to-probability-and-statistics-spring-2022/) (free)
  - [Seeing Theory](https://seeing-theory.brown.edu/) (interactive visualizations)

### Programming

- [ ] **Python Fundamentals**
  - [Python Official Tutorial](https://docs.python.org/3/tutorial/) (free)
  - [Automate the Boring Stuff](https://automatetheboringstuff.com/) (free)
  - [Real Python Tutorials](https://realpython.com/) (free)
  - [Python for Everybody - Dr. Chuck](https://www.py4e.com/) (free)

- [ ] **NumPy & Pandas**
  - [NumPy Official Tutorial](https://numpy.org/doc/stable/user/quickstart.html) (free)
  - [Pandas Getting Started](https://pandas.pydata.org/docs/getting_started/) (free)
  - [Kaggle Pandas Course](https://www.kaggle.com/learn/pandas) (free)

- [ ] **Data Visualization**
  - [Matplotlib Tutorials](https://matplotlib.org/stable/tutorials/index.html) (free)
  - [Seaborn Tutorial](https://seaborn.pydata.org/tutorial.html) (free)

### 📚 Phase 0 Books

| Book | Author | Focus |
|------|--------|-------|
| [Mathematics for Machine Learning](https://mml-book.github.io/) | Deisenroth et al. | Math foundations (free PDF) |
| [Think Stats](https://greenteapress.com/thinkstats2/) | Allen Downey | Statistics with Python (free) |
| [Python Data Science Handbook](https://jakevdp.github.io/PythonDataScienceHandbook/) | Jake VanderPlas | NumPy, Pandas, Matplotlib (free) |

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

### 🎓 Courses

| Course | Platform | Duration | Level |
|--------|----------|----------|-------|
| [Machine Learning Specialization](https://www.coursera.org/specializations/machine-learning-introduction) | Coursera (Andrew Ng) | 3 months | Beginner |
| [Google ML Crash Course](https://developers.google.com/machine-learning/crash-course) | Google | 15 hours | Beginner |
| [StatQuest ML Playlist](https://www.youtube.com/playlist?list=PLblh5JKOoLUICTaGLRoHQDuF_7q2GfuJF) | YouTube | 20+ hours | Beginner |
| [CS229: Machine Learning](https://cs229.stanford.edu/) | Stanford | Full semester | Intermediate |
| [Scikit-learn Tutorials](https://scikit-learn.org/stable/tutorial/index.html) | Official | Self-paced | Beginner |
| [Kaggle ML Course](https://www.kaggle.com/learn/intro-to-machine-learning) | Kaggle | 3 hours | Beginner |
| [Elements of AI](https://www.elementsofai.com/) | University of Helsinki | 30 hours | Beginner |

### 📚 Books

| Book | Author | Level | Notes |
|------|--------|-------|-------|
| Hands-On ML with Scikit-Learn, Keras & TensorFlow | Aurélien Géron | Beginner | Best practical intro |
| [The Hundred-Page ML Book](http://themlbook.com/) | Andriy Burkov | Beginner | Free online, concise |
| [An Introduction to Statistical Learning](https://www.statlearning.com/) | James et al. | Intermediate | Free PDF, classic |
| Pattern Recognition and ML | Christopher Bishop | Advanced | Theoretical depth |
| [Probabilistic ML](https://probml.github.io/pml-book/) | Kevin Murphy | Advanced | Free, comprehensive |

### 🎯 Phase 1 Projects

1. **Predict house prices** - Regression with feature engineering
2. **Classify iris species** - Multi-class classification
3. **Customer segmentation** - K-means clustering
4. **Build ML pipeline** - End-to-end with scikit-learn
5. **Kaggle competition** - Join a beginner-friendly competition

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

### 🎓 Courses

| Course | Platform | Duration | Level |
|--------|----------|----------|-------|
| [Fast.ai Practical Deep Learning](https://course.fast.ai/) | Fast.ai | 7 weeks | Beginner |
| [Deep Learning Specialization](https://www.coursera.org/specializations/deep-learning) | Coursera (Andrew Ng) | 5 months | Intermediate |
| [CS231n: CNNs for Visual Recognition](http://cs231n.stanford.edu/) | Stanford | Full semester | Intermediate |
| [CS224n: NLP with Deep Learning](http://web.stanford.edu/class/cs224n/) | Stanford | Full semester | Intermediate |
| [MIT 6.S191: Intro to Deep Learning](http://introtodeeplearning.com/) | MIT | 1 week intensive | Beginner |
| [NYU Deep Learning](https://atcold.github.io/NYU-DLSP21/) | NYU (Yann LeCun) | Full semester | Intermediate |
| [Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html) | Andrej Karpathy | 10+ hours | Beginner |
| [PyTorch Official Tutorials](https://pytorch.org/tutorials/) | PyTorch | Self-paced | Beginner |
| [TensorFlow Tutorials](https://www.tensorflow.org/tutorials) | TensorFlow | Self-paced | Beginner |
| [Hugging Face NLP Course](https://huggingface.co/learn/nlp-course) | Hugging Face | Self-paced | Intermediate |

### 📚 Books

| Book | Author | Level | Notes |
|------|--------|-------|-------|
| [Deep Learning](https://www.deeplearningbook.org/) | Goodfellow, Bengio, Courville | Intermediate | The bible (free online) |
| [Dive into Deep Learning](https://d2l.ai/) | Zhang et al. | Intermediate | Interactive, free |
| Deep Learning with Python | François Chollet | Beginner | Keras creator |
| Grokking Deep Learning | Andrew Trask | Beginner | Build from scratch |
| Deep Learning from Scratch | Seth Weidman | Beginner | Fundamentals |
| [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/) | Michael Nielsen | Beginner | Free online, intuitive |
| [Understanding Deep Learning](https://udlbook.github.io/udlbook/) | Simon Prince | Intermediate | Free PDF, 2023 |

### 🎯 Phase 2 Projects

1. **Image classifier** - CNN on CIFAR-10 or custom dataset
2. **Sentiment analysis** - RNN/LSTM on movie reviews
3. **Neural style transfer** - Artistic image transformation
4. **Train a small transformer** - From scratch understanding
5. **Fine-tune a pretrained model** - Transfer learning

---

## Phase 3: Specialization Tracks (12-16 weeks)

> Choose your path. You can do multiple.

### Track A: Computer Vision

- [ ] Object Detection (YOLO, Faster R-CNN)
- [ ] Image Segmentation (U-Net, Mask R-CNN)
- [ ] Generative Models (GANs, Diffusion)
- [ ] Video Understanding
- [ ] 3D Vision

**Courses:**
| Course | Platform | Level |
|--------|----------|-------|
| [CS231n: Deep Learning for Computer Vision](http://cs231n.stanford.edu/) | Stanford | Intermediate |
| [CS231A: Computer Vision](http://web.stanford.edu/class/cs231a/) | Stanford | Advanced |
| [First Principles of Computer Vision](https://www.youtube.com/@firstprinciplesofcomputerv3258) | YouTube | Intermediate |
| [Roboflow Computer Vision](https://roboflow.com/learn) | Roboflow | Beginner |

**Books:**
- "Deep Learning for Computer Vision" - Rajalingappaa Shanmugamani
- "Computer Vision: Algorithms and Applications" - Richard Szeliski (free online)

**Projects:**
- Real-time object detection system
- Image generation with Stable Diffusion
- Face recognition pipeline

### Track B: Natural Language Processing

- [ ] Word Embeddings (Word2Vec, GloVe)
- [ ] Transformer Architecture (deep dive)
- [ ] Large Language Models (GPT, BERT, LLaMA)
- [ ] Prompt Engineering & Fine-tuning
- [ ] RAG Systems

**Courses:**
| Course | Platform | Level |
|--------|----------|-------|
| [CS224n: NLP with Deep Learning](http://web.stanford.edu/class/cs224n/) | Stanford | Intermediate |
| [Hugging Face NLP Course](https://huggingface.co/learn/nlp-course) | Hugging Face | Intermediate |
| [CS685: Advanced NLP](https://www.youtube.com/playlist?list=PLWnsVgP6CzaelCF_jmn5HrpOXzRAPNjWj) | UMass | Advanced |
| [CMU Advanced NLP](https://www.youtube.com/playlist?list=PL8PYTP1V4I8DZprnWryM4nR8IZl1ZXDjg) | CMU (Graham Neubig) | Advanced |
| [Prompt Engineering Guide](https://www.promptingguide.ai/) | DAIR.AI | Beginner |

**Books:**
- "Speech and Language Processing" - Jurafsky & Martin (free online)
- "Natural Language Processing with Transformers" - Tunstall et al.

**Projects:**
- Build a chatbot with fine-tuned model
- Document Q&A system with RAG
- Sentiment analysis API

### Track C: Reinforcement Learning

- [ ] MDPs and Bellman Equations
- [ ] Q-Learning, DQN
- [ ] Policy Gradients, A2C, PPO
- [ ] Multi-Agent RL
- [ ] Model-Based RL

**Courses:**
| Course | Platform | Level |
|--------|----------|-------|
| [Spinning Up in Deep RL](https://spinningup.openai.com/) | OpenAI | Intermediate |
| [David Silver's RL Course](https://www.davidsilver.uk/teaching/) | DeepMind | Intermediate |
| [CS285: Deep RL](http://rail.eecs.berkeley.edu/deeprlcourse/) | UC Berkeley | Advanced |
| [Hugging Face Deep RL Course](https://huggingface.co/learn/deep-rl-course) | Hugging Face | Intermediate |

**Books:**
- "Reinforcement Learning: An Introduction" - Sutton & Barto (free online)
- "Deep Reinforcement Learning Hands-On" - Maxim Lapan

**Projects:**
- Train agent to play Atari games
- Robot simulation with MuJoCo/Isaac Gym
- Multi-agent environment

### Track D: Generative AI & LLMs

- [ ] Transformer architecture deep dive
- [ ] Training LLMs (pretraining, SFT, RLHF)
- [ ] Prompt engineering
- [ ] Fine-tuning (LoRA, QLoRA)
- [ ] RAG and agents

**Courses:**
| Course | Platform | Level |
|--------|----------|-------|
| [Generative AI with LLMs](https://www.coursera.org/learn/generative-ai-with-llms) | Coursera (AWS/DeepLearning.AI) | Intermediate |
| [LLM Course](https://github.com/mlabonne/llm-course) | GitHub | Intermediate |
| [Full Stack LLM Bootcamp](https://fullstackdeeplearning.com/llm-bootcamp/) | FSDL | Intermediate |
| [Prompt Engineering for ChatGPT](https://www.coursera.org/learn/prompt-engineering) | Vanderbilt | Beginner |

**Projects:**
- Build a RAG application
- Fine-tune LLaMA on custom data
- Create an AI agent with tool use

### Track E: AI Safety & Alignment

- [ ] Alignment Problem Overview
- [ ] RLHF (Reinforcement Learning from Human Feedback)
- [ ] Interpretability & Explainability
- [ ] Red-teaming and adversarial attacks
- [ ] Constitutional AI

**Resources:**
| Resource | Type | Level |
|----------|------|-------|
| [AI Safety Fundamentals](https://www.aisafetyfundamentals.com/) | Course | Beginner |
| [Anthropic Research](https://www.anthropic.com/research) | Papers | Advanced |
| [AI Alignment Forum](https://www.alignmentforum.org/) | Community | All |
| [MIRI Research](https://intelligence.org/research/) | Papers | Advanced |

---

## Phase 4: Research Skills (8-12 weeks)

> From practitioner to scientist.

### Reading Papers

- [ ] Learn to read papers efficiently
  - [How to Read a Paper](https://web.stanford.edu/class/ee384m/Handouts/HowtoReadPaper.pdf) (free)
  - [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
- [ ] Follow arXiv daily (cs.AI, cs.LG, cs.CL)
- [ ] Use [Papers With Code](https://paperswithcode.com/)
- [ ] Follow [The Batch](https://www.deeplearning.ai/the-batch/) newsletter

### Reproducing Research

- [ ] Pick 3 papers and reproduce results
- [ ] Document discrepancies
- [ ] Understand ablation studies

### Experimentation

- [ ] Hyperparameter tuning (Optuna, Ray Tune)
- [ ] Experiment tracking (MLflow, Weights & Biases)
- [ ] Statistical significance testing

### Writing

- [ ] Learn LaTeX (Overleaf)
- [ ] Understand paper structure
- [ ] Write a technical blog post

### 🎯 Phase 4 Projects

1. **Reproduce a paper** - Pick SOTA from 2023-2024, reproduce it
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

## Additional Resources

### YouTube Channels

| Channel | Focus |
|---------|-------|
| [3Blue1Brown](https://www.youtube.com/@3blue1brown) | Math intuition |
| [StatQuest](https://www.youtube.com/@statquest) | Statistics & ML |
| [Andrej Karpathy](https://www.youtube.com/@AndrejKarpathy) | Deep learning from scratch |
| [Yannic Kilcher](https://www.youtube.com/@YannicKilcher) | Paper explanations |
| [Two Minute Papers](https://www.youtube.com/@TwoMinutePapers) | Research summaries |
| [Sentdex](https://www.youtube.com/@sentdex) | Practical Python ML |
| [Jeremy Howard](https://www.youtube.com/@holopin) | Fast.ai |
| [DeepMind](https://www.youtube.com/@Google_DeepMind) | Research talks |
| [Lex Fridman](https://www.youtube.com/@lexfridman) | AI interviews |

### Podcasts

- [Lex Fridman Podcast](https://lexfridman.com/podcast/)
- [Machine Learning Street Talk](https://www.youtube.com/@MachineLearningStreetTalk)
- [The TWIML AI Podcast](https://twimlai.com/)
- [Gradient Dissent](https://wandb.ai/fully-connected/gradient-dissent)

### Newsletters

- [The Batch](https://www.deeplearning.ai/the-batch/) - Andrew Ng
- [AI Weekly](https://aiweekly.co/)
- [Papers With Code Newsletter](https://paperswithcode.com/)
- [Import AI](https://importai.substack.com/)

### Communities

| Community | Platform |
|-----------|----------|
| r/MachineLearning | Reddit |
| r/learnmachinelearning | Reddit |
| ML Discord servers | Discord |
| Hugging Face Forums | Web |
| Kaggle Forums | Web |

---

## Tools & Environment

### Essential Setup

```bash
# Environment
python >= 3.10
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
| Kaggle Notebooks | GPU | Free (30h/week) |
| Lightning.ai | GPU | Free tier |
| Lambda Labs | Cloud GPU | Paid |
| Vast.ai | Cloud GPU | Cheap |
| RunPod | Cloud GPU | Cheap |
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
