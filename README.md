### Objectives ###

The "Am I the A******?" (AITA) subreddit provides a rich source of real-world scenarios where individuals seek moral guidance from an online community regarding their behaviour. In this project, we aim to develop a predictive model to anticipate the likely moral judgement (e.g., "You’re the A******" or "Not the A******") of a given situation based on the text of AITA subreddit posts.

The primary objective of this project is to build a predictive model capable of discerning the moral judgement associated with AITA subreddit posts. By analysing the textual content of these posts, our model will strive to accurately predict whether the individual in the scenario is perceived as acting morally or immorally by the community.

### Model Experimentation ###

We will embark on a systematic exploration of diverse modelling approaches, encompassing both traditional machine learning algorithms and cutting-edge deep learning architectures. 

* **Traditional ML**: Use classical machine learning algorithms such as SVM, Naive Bayes, KNN, or decision trees for classification. Data preprocessing steps include stopword removal, bag-of-words representation or TF-IDF.
* **Ensemble Learning**: Using a mix of traditional ML + rule-based approaches to create an ensemble model. 
* **Recurrent Neural Networks (RNNs)**: Utilize RNN architectures like LSTM or GRU to capture sequential dependencies in text data and perform classification.
* **Transformer-based Models**: Employed a fine-tuned BERT model.


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
