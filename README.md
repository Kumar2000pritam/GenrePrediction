#  Multi-Label Genre Classification Pipeline

This project builds an **end-to-end machine learning pipeline** for predicting multiple genres of movies/TV shows using:

- Logistic Regression
- XGBoost
- Text embeddings (Sentence Transformers) : paraphrase-multilingual-MiniLM-L12-v2 
- Tabular feature engineering

---

#  Project Structure
    GenrePrediction/
    │
    ├── trainer.py 
    ├── validate.py 
    ├── inference.py 
    │
    ├── preprocess.py
    ├── encoding.py
    ├── utils.py
    ├── plotting.py
    │
    ├── train_multilabel_logreg.py
    ├── xgb_multilabel.py
    │
    ├── artifacts_handler.py
    ├── mlpipeline.py
    │
    └── data/
        └── tv-shows.csv
---

# ⚙️ Setup Instructions

## 1. Clone the Repository
```bash
git clone https://github.com/Kumar2000pritam/GenrePrediction.git
```
## 2. Enter into project folder
```bash
cd GenrePrediction
```
## 3. Create Virtual Environment & Activate

###  Windows
```bash
python -m venv venv
venv\Scripts\activate
```
## 4. Install Dependencies from requirements.txt

```bash
pip install -r requirements.txt
```
## 5. Train both models
--out: folder name of artifacts
--data: data path
Training process will take time
```bash
python trainer.py --data data/tv-shows.csv --out ml_artifacts
```
## 6. Validate custom data with both models
--rows: number of record to validate
--data: path of data
After training try it
```bash
python validate.py --data data/tv-shows.csv --artifacts ml_artifacts --rows 200
```
## 7. Inference both models

```bash
python inference.py --artifacts_path ml_artifacts --title "Maggie Simpson in The Longest Daycare" --director "" --cast "" --description "In this Oscar-nominated short from The Simpsons, Maggie navigates an eventful first day at daycare." --duration "5 min" --release_year 2012 --rating "PG"
```
## Others Details
### Metrics Used
- Micro F1 Score  
- Macro F1 Score  
- Hamming Loss 
### Features Used
#### Tabular Features
- Release year  
- Duration  
- Rating  

#### Text Features
- Title  
- Description  
- Director  
- Cast  

## Outputs Saved
- Trained models (LogReg + XGBoost)
- Encoders (MLB, rating, duration,)
- Feature columns
- Thresholds
- Metrics
- Text embedding model

