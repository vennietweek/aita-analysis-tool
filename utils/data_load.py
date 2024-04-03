import pandas as pd
from sklearn.model_selection import train_test_split

# Load the data
part1 = pd.read_csv('../data/raw_data_1.csv', parse_dates=[0])
part2 = pd.read_csv('../data/raw_data_2.csv', parse_dates=[0])
df = pd.concat([part1, part2])

# Clean the data
df['label'] = df['AITA'].apply(lambda x : 1 if x == "Asshole" else 0)
df_cleaned = df.drop(['created_time','upvotes', 'score', 'id', 'upvotes', 'title', 'AITA'],axis=1)

# Train test split
train, test = train_test_split(df_cleaned, test_size=0.2, random_state=42)

# Save the data
train.to_csv('../data/train.csv', index=False)
test.to_csv('../data/test.csv', index=False)