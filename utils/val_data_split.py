import pandas as pd

# !wget https://raw.githubusercontent.com/vennietweek/aita-analysis-tool/main/data/train.csv

train = pd.read_csv('train.csv')

group_0 = train[train['label'] == 0]
group_1 = train[train['label'] == 1]

# get the min of each label counts and divide by 2 
min_size = round((min(len(group_0), len(group_1)))/2)

val_group_0 = group_0.sample(n=min_size, random_state=42)  # 705
val_group_1 = group_1.sample(n=min_size, random_state=42)  # 705 

validation_set = pd.concat([val_group_0, val_group_1])
train_set = train.drop(validation_set.index) # orig train len - val len = 9538 - 1410 = 8128 rows 

print(validation_set['label'].value_counts()) 
len(validation_set) # 1410 total rows 

validation_set.to_csv("validation.csv",index=False)
train_set.to_csv("train.csv",index=False)
