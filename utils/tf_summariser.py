from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
import pandas as pd
import numpy as np
from tqdm.auto import tqdm

def batch_process_summaries(texts, batch_size=8, model_name='t5-small', max_length=512, summary_length=100):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name).to(device)
    model.eval()

    batches = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]
    summaries = []

    for batch in tqdm(batches, desc="Summarizing batches"):
        batch_to_summarize = ["summarize: " + text for text in batch]
        inputs = tokenizer(batch_to_summarize, return_tensors='pt', padding=True, truncation=True, max_length=max_length)
        input_ids = inputs.input_ids.to(device)
        attention_mask = inputs.attention_mask.to(device)
        
        with torch.no_grad():
            summary_ids = model.generate(input_ids, attention_mask=attention_mask, max_length=summary_length, num_beams=4, length_penalty=1.0, early_stopping=True)
        
        batch_summaries = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in summary_ids]
        summaries.extend(batch_summaries)

    return summaries

# Load data
df_train = pd.read_csv("../data/balanced/train.csv")
df_test = pd.read_csv("../data/balanced/test.csv")

print("Summarizing train texts...")
train_summaries = batch_process_summaries(df_train['content'].tolist(), batch_size=8, model_name='t5-small', summary_length=100)
df_train['summary'] = train_summaries

print("Summarizing test texts...")
test_summaries = batch_process_summaries(df_test['content'].tolist(), batch_size=8, model_name='t5-small', summary_length=100)
df_test['summary'] = test_summaries

# Save the modified DataFrames to new CSV files
df_train.to_csv("../data/balanced/train_summarised_t5.csv", index=False)
df_test.to_csv("../data/balanced/test_summarised_t5.csv", index=False)

print("Processing complete. Files saved.")
