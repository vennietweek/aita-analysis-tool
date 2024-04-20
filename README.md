# Morality in the Digital Age: Predictive Modelling on AITA Subreddit Posts

Project Team 5's Submission for CS5246, taken in AY23/24 S2 at the National University of Singapore.

## Objectives 

The "Am I the A******?" (AITA) subreddit provides a rich source of real-world scenarios where individuals seek moral guidance from an online community regarding their behaviour. In this project, we aim to develop a predictive model to anticipate the likely moral judgement (e.g., "You’re the A******" or "Not the A******") of a given situation based on the text of AITA subreddit posts.

The primary objective of this project is to build a predictive model capable of discerning the moral judgement associated with AITA subreddit posts. By analysing the textual content of these posts, our model will strive to accurately predict whether the individual in the scenario is perceived as acting morally or immorally by the community.

## Dataset Files : 
- Non Summarised : [Train](https://github.com/vennietweek/aita-analysis-tool/blob/main/data/balanced/train.csv) and [Test](https://github.com/vennietweek/aita-analysis-tool/blob/main/data/balanced/test.csv)
- Summarised with GPT2 : [Train](https://github.com/vennietweek/aita-analysis-tool/blob/main/data/summarised/train_summarised_gpt2.csv) and [Test](https://github.com/vennietweek/aita-analysis-tool/blob/main/data/summarised/test_summarised_gpt2.csv)
- Summarised with PageRank : [Train (part 1)](https://github.com/vennietweek/aita-analysis-tool/blob/main/data/train_with_pagerank_part1.csv), [Train (part 2)](https://github.com/vennietweek/aita-analysis-tool/blob/main/data/train_with_pagerank_part2.csv) and [Test](https://github.com/vennietweek/aita-analysis-tool/blob/main/data/test_with_pagerank.csv) 

## Model Experimentation 

Our systematic exploration of diverse modelling approaches, encompassing both traditional machine learning algorithms and cutting-edge deep learning architectures includes :  

* **Traditional ML**: Deployed [classical machine learning algorithms](https://github.com/vennietweek/aita-analysis-tool/blob/main/models/TraditionalApproach.ipynb) such as SVM, Naive Bayes, KNN, or decision trees for classification. Data preprocessing steps include stopword removal, bag-of-words representation or TF-IDF.
* **Ensemble Learning**: Using a mix of traditional ML + rule-based approaches to create an (Ensemble model)[https://github.com/vennietweek/aita-analysis-tool/blob/main/models/Ensemble_Approach.ipynb]. 
* **Recurrent and Convolutional Neural Networks**: [Bi-LSTMs, CNNs](https://github.com/vennietweek/aita-analysis-tool/blob/main/models/model_architectures.py) to capture sequential dependencies in text data and perform classification.
* **Transformer-based Models**: A [fine-tuned BERT model](https://github.com/vennietweek/aita-analysis-tool/blob/main/models/BERT.ipynb).


### Installation Instructions ###

1. Set up an empty folder and clone the repository into your folder
  ```
  git clone https://github.com/vennietweek/aita-analysis-tool.git
  ```
2. Initialise virtual environment in the project root folder:
  ```
  python -m venv venv
  ```
3. Activate the virtual environment:
  ```
  source venv/bin/activate
  ```
4. Upgrade pip:
  ```
  pip install --upgrade pip setuptools wheel
  ```
5. Install project dependencies:
  ```
  pip install -r requirements.txt
  ```
