## Data

The data is extracted from [DevGPT](https://github.com/NAIST-SE/DevGPT).

To create a working local copy of the clean dataset:

1. Clone the DevGPT repo into this project folder:
```
   git clone https://github.com/NAIST-SE/DevGPT.git
```
2. Run the collection script:
```
   python collect_devgpt.py
```
3. Run the cleaning script:
```
   python full_english_convos.py
```

You should now have a file called `prompt_answer_pairs_clean.csv` containing complete, English-only ChatGPT conversations.
