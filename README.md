# Analisis Sentimen IKN

Due to the limit on GitHub storage, I put the model on Google Drive. Download it if you want to try it:
[Download Model](https://drive.google.com/file/d/1jsnuwsdA_GM9JmOO5N3b_RDsi_7Zq7KO/view?usp=sharing)

## Setup

Clone the repository

```bash
$ git clone <repository-url>
$ cd <repository-directory>
```

Install the requirements

```bash
$ pip3 install -r requirements.txt
```

Download the fine-tuned model and place it in the model directory

```bash
$ unzip model.zip
$ mv fine_tuned_model/ model/ && mv xgboost_model.json model/
```

Run the program

```bash
$ streamlit run app.py
```
