# Context-Aware Seq2Seq Prompt Prediction

**Christopher Kaldas, Eli Reinhold, Stephen Radice**  
Department of Computer Science | CS 584 A | Stevens Institute of Technology

Presented at the **Stevens iCNS AI Engineering and Science Symposium, May 2026**

---

## Overview
This project works backwards: given a ChatGPT response, predict what the original user prompt was. We fine-tuned two T5-Small Seq2Seq models to reverse-engineer user prompts from AI-generated responses — one without context and one with a sliding context window. We found that added context degraded model performance, as deeper conversation turns made it harder for the model to isolate relevant intent.

---

## Poster
See [`poster.pdf`](./poster.pdf) for the full research poster presented at the symposium.

## Report
See [`report.pdf`](./report.pdf) for the full research report.

---

## Data
The data is extracted from [DevGPT](https://github.com/NAIST-SE/DevGPT).  
To create a working local copy of the clean dataset:
1. Download all files in this repo onto your local machine
2. Run the collection script:
```
python collect_devgpt.py
```
3. Run the cleaning script:
```
python full_english_convos.py
```

You should now have a file called `prompt_answer_pairs_clean.csv` containing complete, English-only ChatGPT conversations.

---

## Results
- Model performance degrades as conversation depth increases
- Model with context performed worse than the model without context. This may be because it was difficult for the model to recognize which parts were truly relevant to only the most recent prompt
- Outperformed GPT-4o mini on the prompt prediction task

---

## References
1. Neural Question Generation — Du et al., ACL 2017
2. Limits of Transfer Learning — Raffel et al., JMLR 2020
3. Context-Aware Generation — Dong et al., U2BigData 2024
4. Prompt Engineering for LLMs — Medium, 2026
5. Sliding Window Attention — Fu et al., 2025
6. Steering Target Atoms for LLM Control — Wang et al., ACL 2025
7. Prompt Steerability of LLMs — Miehling et al., NAACL 2025
