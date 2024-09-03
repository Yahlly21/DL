**Email Classification Using NLP Techniques: Identifying Phishing Emails - NLP Course - Yahlly Schein & Ron Elias**

This repository contains the code and resources for our project on email classification using Natural Language Processing (NLP) techniques. 
The project is divided into two main parts: **Part A** covers the statistical analysis and preprocessing of the data, while **Part B** involves training models using pre-trained models from Hugging Face, followed by various model compressing techniques.
**A detailed PDF** document summarizing the project, including problem introduction, methodologies, model selection framework and results, is available in the repository for reference.

**Repository Structure**
**Part A: Statistical Analysis and Data Preprocessing**

A.ipynb: Provides an overview of the dataset used in our study, followed by a detailed exploration of the data. It discusses the specific preprocessing steps applied to transform the raw text into a format suitable for NLP model input.

Key Techniques and Libraries Used:
Word Clouds for visualizing word frequency distributions.
Tokenization, Stopwords Removal, and N-grams using nltk.
Sentiment Analysis using nltk.sentiment.vader.
Readability Scores using textstat.

**Part B: Model Training and Shrinking**

1. Distillation_Roberta.ipynb: Distills the roberta-large model to a smaller roberta-base model.

2. Evaluation.ipynb: Evaluates model performance using a GPU.

3. Evaluation_CPU.ipynb: Evaluates model performance using a CPU.

4. Model_Selection_Dash_App.ipynb: A Dash app that helps in model selection using a utility function formulated in the project.

6. Pruning_Roberta.ipynb: Prunes the Roberta model.

8. Quantization_Roberta.ipynb: Quantizes the Roberta model.

9. Training_Albert.ipynb: Trains the Albert model on the email dataset.

10. Training_Deberta.ipynb: Trains the Deberta model.

11. Training_Roberta.ipynb: Trains the Roberta model.

12. Training_Distilbert.ipynb: Trains the Distilbert model.

13. results.csv: Contains the results from all trained models, including metrics such as accuracy, precision, recall, F1-score, and AUC.

14. finetuning_last_2_layers: Trains all the models using the second finetuing tecnique.

15. result_last_2_layers: Contains the results from all trained models using the second  finetuing tecnique.

16. Assets Folder 'assets/': Contains necessary files for the Model Selection Dash app.

**How to Use This Repository** - 

Prerequisites Part A: Python 3.x, 
The following Python packages are required: collections, WordCloud, nltk, sklearn, textstat

Prerequisites Part B:
CUDA Toolkit: Ensure you have the CUDA toolkit installed to leverage GPU acceleration.
cuDNN: Install NVIDIA's cuDNN library for deep learning acceleration.
PyTorch: Install one of these deep learning frameworks with GPU support

**Setup Instructions**

1. Download the Data: Obtain the dataset from Kaggle.Need to download the phishing_email.csv file from here: https://www.kaggle.com/datasets/naserabdullahalam/phishing-email-dataset/data

2. Run Part A: Execute the A.ipynb notebook to preprocess the data and partition it for model training (the train-test split happens in the end of this notebook).

3. Train and Shrink Models:

4. Run the training notebooks in Part B to train base models. Note that there are saperated notebooks for the 2 finetuning techniques.
Apply distillation, pruning, and quantization to compress the models.

**Model compressing Techniques**
Distillation: Reduces the size of a large model by training a smaller model (student model) to mimic the behavior of a larger model (teacher model).
Pruning: Reduces the size of a model by removing weights or neurons that have minimal impact on the model's predictions.
Quantization: Converts a model's weights from floating-point numbers to lower-bit representations, reducing the model's size and improving inference efficiency.

**Results**

The results of all trained and shrunk models are documented in the results.csv file, providing key performance metrics such as accuracy, precision, recall, F1-score, and AUC.
