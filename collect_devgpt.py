import os
import json
import csv
import glob

REPO_PATH = "./DevGPT"
OUTPUT_FILE = "prompt_answer_pairs.csv"

SOURCE_TYPES = [
    "issue_sharings",
    "pr_sharings",
    "discussion_sharings",
    "commit_sharings",
    "file_sharings",
    "hn_sharings",
]

def extract_conversations(data, source_type, snapshot):
    rows = []
    records = data if isinstance(data, list) else data.get("Sources", [])

    for record in records:
        chatgpt_sharings = record.get("ChatgptSharing", [])
        for sharing in chatgpt_sharings:
            if sharing.get("Status") != 200:
                continue
            conversations = sharing.get("Conversations", [])
            for i, conv in enumerate(conversations):
                prompt = conv.get("Prompt", "")
                answer = conv.get("Answer", "")
                if prompt and answer:
                    rows.append({
                        "snapshot": snapshot,
                        "source_type": source_type,
                        "source_url": record.get("URL", ""),
                        "chatgpt_url": sharing.get("URL", ""),
                        "model": sharing.get("Model", ""),
                        "date_of_conversation": sharing.get("DateOfConversation", ""),
                        "conv_index": i,
                        "num_prompts_in_convo": sharing.get("NumberOfPrompts", ""),
                        "prompt": prompt,
                        "answer": answer,
                    })
    return rows

all_rows = []

for snapshot_dir in sorted(glob.glob(os.path.join(REPO_PATH, "snapshot_*"))):
    snapshot_name = os.path.basename(snapshot_dir)
    for source_type in SOURCE_TYPES:
        matches = glob.glob(os.path.join(snapshot_dir, f"*_{source_type}.json"))
        for filepath in matches:
            print(f"Processing: {filepath}")
            with open(filepath, "r", encoding="utf-8") as f:
                try:
                    data = json.load(f)
                    rows = extract_conversations(data, source_type, snapshot_name)
                    all_rows.extend(rows)
                    print(f"  → {len(rows)} rows")
                except json.JSONDecodeError as e:
                    print(f"  ERROR: {e}")

print(f"\nTotal rows collected: {len(all_rows)}")

if all_rows:
    # Deduplicate by chatgpt_url + conv_index (same convo across snapshots)
    seen = set()
    deduped = []
    for row in all_rows:
        key = (row["chatgpt_url"], row["conv_index"])
        if key not in seen:
            seen.add(key)
            deduped.append(row)

    print(f"After deduplication: {len(deduped)} rows")

    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=deduped[0].keys())
        writer.writeheader()
        writer.writerows(deduped)

    print(f"Saved to {OUTPUT_FILE}")
else:
    print("No rows found — check that JSON files have 'ChatgptSharing' with Status 200.")