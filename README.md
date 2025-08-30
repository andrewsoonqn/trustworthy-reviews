# Train and Run the Model
1. `pip install -r requirements.txt`
2. `python pipeline.py`

# Preprocessing
We have cleaned and labelled provided datasets. However, to process new datasets, you can follow these steps
## Setup
1. `python -m spacy download en_core_web_sm`

2. Install [Ollama](https://ollama.com/download). Then, `ollama pull gemma3:4b`

3. Modify the input and output file names in `preprocess.py` and `llm.py`.
Run `python preprocess.py` then `llm.py`

Modify file name in `pipeline.py` accordingly too
