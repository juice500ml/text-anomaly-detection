## 2017 Naver Hackday Winter Project
Useful when class is too many for normal classification.
Currently extracts features with fasttext and detects anomaly using PyNomaly.

## [preprocess.py](preprocess.py)
Clean up Korean text.
- Remove uncommon English words with dictionary of 10000 words(google-10000-english)
- Split if Korean and English words are concatenated
- Remove special characters
- Split by morphs with Konlpy

## [find.py](find.py)
Find least-from-anomaly and supposed-anomaly element from given features.

## [score_model.py](score_model.py)
For specific class C, choose 1000 element from the class (Positive Sampling) and 25 element from other than the class (Negative Sampling).
Find 50 supposed-anomaly and count the found negative samples.

## [total.py](total.py)
Build model with variants and find out the best model with metric below
1. Build model with variants as follows
    - Model Type
    - Feature Dimension 
    - Learning Rate
    - Epochs
    - Use pretrained model or not
2. Extract feature vector from each element for each model
3. Evaluate metric for each model as follows
    - For specific class C, choose 1000 element from the class (Positive Sampling) and 50 element from other than the class (Negative Sampling).
    - Find 100 supposed-anomaly and count the found negative samples.

## Dependencies
- Naver Shopping Data (Not available to public)
- Python3 with [libraries](requirements.txt)
- [fasttext](https://fasttext.cc) and [pretrained models](https://fasttext.cc/docs/en/pretrained-vectors.html)
- [google-10000-english](https://github.com/first20hours/google-10000-english)
