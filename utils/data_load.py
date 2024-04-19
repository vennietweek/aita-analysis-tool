import pandas as pd
import numpy as np

# Load data
df_train = pd.read_csv("../data/balanced/train.csv")
df_test = pd.read_csv("../data/balanced/test.csv")

# Prepare train and test data and labels
train_texts = df_train['content']
test_texts = df_test['content']
train_labels = np.asarray(df_train['flag']).astype('float32')
test_labels = np.asarray(df_test['flag']).astype('float32')

# Load GPT-2 summarised data

df_train_gpt2 = pd.read_csv("../data/summarised/train_summarised_gpt2.csv")
df_test_gpt2 = pd.read_csv("../data/summarised/test_summarised_gpt2.csv")

# Prepare train and test data and labels
train_texts_gpt2 = df_train_gpt2['summarised']
test_texts_gpt2 = df_test_gpt2['summarised']

# Load Pagerank-summarised data

df_train_pagerank_1 = pd.read_csv("../data/train_with_pagerank_part1.csv")
df_train_pagerank_2 = pd.read_csv("../data/train_with_pagerank_part2.csv")
df_test_pagerank = pd.read_csv("../data/test_with_pagerank.csv")

# Combine the two parts of the training data
df_train_pagerank = pd.concat([df_train_pagerank_1, df_train_pagerank_2])

# Prepare train and test data and labels
train_texts_pagerank = df_train_pagerank['pagerank']
test_texts_pagerank = df_test_pagerank['pagerank']

# Save the data

np.save("../data/npy/train_labels.npy", train_labels)
np.save("../data/npy/test_labels.npy", test_labels)

np.save("../data/npy/train_texts.npy", train_texts)
np.save("../data/npy/test_texts.npy", test_texts)

np.save("../data/npy/train_texts_gpt2.npy", train_texts_gpt2)
np.save("../data/npy/test_texts_gpt2.npy", test_texts_gpt2)

np.save("../data/npy/train_texts_pagerank.npy", train_texts_pagerank)
np.save("../data/npy/test_texts_pagerank.npy", test_texts_pagerank)