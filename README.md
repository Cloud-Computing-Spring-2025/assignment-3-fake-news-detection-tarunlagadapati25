# Assignment-5-FakeNews-Detection

##  Overview

This project builds a simple **machine learning pipeline** using **Apache Spark MLlib** to classify news articles as **FAKE** or **REAL** based on their content. The pipeline includes:

- Text preprocessing
- Feature extraction using TF-IDF
- Model training with Logistic Regression
- Evaluation using Accuracy and F1 Score

---

##  Dataset

The dataset used is: `fake_news_sample.csv`  
It contains the following columns:

- `id`: Unique article ID  
- `title`: Headline of the news article  
- `text`: Full text/content of the article  
- `label`: Classification label (`FAKE` or `REAL`)

---

##  Tasks

###  Task 1: Load & Basic Exploration

- Load `fake_news_sample.csv` with inferred schema.
- Create a temporary view: `news_data`.
- Perform basic queries:
  - Show first 5 rows
  - Count total articles
  - Retrieve distinct labels


---

###  Task 2: Text Preprocessing

- Convert text to lowercase.
- Tokenize text into words.
- Remove stopwords.
- Create a new column: `filtered_words`.


---

###  Task 3: Feature Extraction

- Apply `HashingTF` and `IDF` to generate TF-IDF features.
- Index the label (`FAKE` → 0, `REAL` → 1).
- Use `VectorAssembler` to create a single feature vector.


---

###  Task 4: Model Training

- Split dataset (80% training, 20% test).
- Train a Logistic Regression model using Spark MLlib.
- Generate predictions on the test set.


---

###  Task 5: Evaluate the Model

- Evaluate the model using:
  - Accuracy
  - F1 Score

 **Evaluation Output:**

| Metric    | Value |
|-----------|-------|
| Accuracy  | 0.89  |
| F1 Score  | 0.88  |


---

##  How to Run the Project

###  Prerequisites

- Apache Spark
- Python 3.x (with PySpark)
- Java 8/11
- Jupyter Notebook or IDE

###  Steps

1. Start a SparkSession:
    ```python
    from pyspark.sql import SparkSession

    spark = SparkSession.builder \
        .appName("FakeNewsClassification") \
        .getOrCreate()
    ```

2. Run each task (as `.py` scripts or in Jupyter cells):
    - `task1_load_and_explore.py`
    - `task2_preprocessing.py`
    - `task3_feature_extraction.py`
    - `task4_model_training.py`
    - `task5_evaluation.py`

3. Output will be saved to CSV files (`taskX_output.csv`).

4. Shut down Spark session:
    ```python
    spark.stop()
    ```

---

