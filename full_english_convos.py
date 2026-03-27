import pandas as pd
from langdetect import detect

def is_english(text):
    try:
        return detect(str(text)) == 'en'
    except:
        return False

df = pd.read_csv("prompt_answer_pairs.csv")
print(f"Original: {len(df)} rows, {df['chatgpt_url'].nunique()} conversations")

# Mark each row as english or not
print("Detecting languages...")
df["is_english"] = df["prompt"].apply(is_english)

# Find conversations where ALL turns are english
all_english_convos = df.groupby("chatgpt_url")["is_english"].all()
all_english_urls = all_english_convos[all_english_convos].index

df = df[df["chatgpt_url"].isin(all_english_urls)].drop(columns=["is_english"])
print(f"After english filter: {len(df)} rows, {df['chatgpt_url'].nunique()} conversations")

# Keep only complete conversations
actual_counts = df.groupby("chatgpt_url")["conv_index"].count().reset_index()
actual_counts.columns = ["chatgpt_url", "actual_count"]
df = df.merge(actual_counts, on="chatgpt_url")
df = df[df["actual_count"] == df["num_prompts_in_convo"]].drop(columns=["actual_count"])
print(f"After completeness filter: {len(df)} rows, {df['chatgpt_url'].nunique()} conversations")

df.to_csv("prompt_answer_pairs_clean.csv", index=False, encoding="utf-8")
print("Saved to prompt_answer_pairs_clean.csv")