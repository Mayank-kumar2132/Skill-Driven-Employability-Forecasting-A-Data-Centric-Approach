# Skill-Driven Employability Forecasting: A Data-Centric Approach
### Abstract
Employability prediction is a crucial task in today’s dynamic job market, aiding in efficient resource allocation and career planning. This project explores the application of both traditional machine learning and advanced deep learning techniques to predict the employability of individuals based on their academic qualifications, skill sets, and relevant experience.

Leveraging a comprehensive dataset comprising diverse attributes of candidates, we employ feature engineering to extract meaningful insights and construct predictive models. This study compares the performance of various machine learning algorithms, including:

Logistic Regression
Decision Trees
Random Forests
Additionally, we employ deep learning architectures such as:

Feedforward Neural Networks
Recurrent Neural Networks (RNNs)
Through rigorous experimentation and cross-validation, we demonstrate the superior predictive accuracy of deep learning models in capturing complex patterns within the data. This research contributes to the advancement of predictive analytics in human resources management and offers valuable implications for recruitment processes and career counseling services.

### Project Overview
This project aims to predict employability based on multiple factors, including academic qualifications and skill sets. The project involves the following steps:

### Feature Engineering: PCA (Principal Component Analysis) and LDA (Linear Discriminant Analysis) were used for dimensionality reduction and enhanced feature extraction.
Model Comparison: We compared traditional machine learning models with advanced deep learning techniques.
Interpretability: We focused on understanding key factors influencing employability prediction and the interpretability of the models.
### Technologies and Libraries Used
Machine Learning Libraries:
Scikit-Learn
Logistic Regression, Decision Trees, Random Forests
PCA & LDA for feature extraction
### Deep Learning Libraries:
TensorFlow/Keras (Feedforward Neural Networks, RNNs)
Data Analysis and Visualization:
Pandas, Numpy for Data Handling
Matplotlib, Seaborn for Visualization
### Natural Language Processing:
Word Cloud Generation for skills analysis
NLTK or WordCloud Libraries
### Key Features
Employability Prediction: Predicts whether a candidate is employable based on skills, experience, and academic background.
Machine Learning Models: Implements and compares traditional algorithms such as Logistic Regression, Decision Trees, and Random Forests.
Deep Learning Models: Employs advanced architectures like Feedforward Neural Networks and RNNs to enhance predictive accuracy.
Dimensionality Reduction: Uses PCA and LDA to improve model performance and interpretability.
Skill Word Cloud: A visual representation of key skills from the dataset using Word Cloud.
Files in the Repository
megaData.csv: The dataset used for prediction modeling.
Emplobility Prediction on Skill Data.ipynb: Jupyter notebook containing the analysis, feature engineering, and machine learning/deep learning implementation.
LDA.ipynb/PCA.ipynb: Jupyter notebook focusing on Linear Discriminant Analysis for feature extraction.
Word Cloud of Skills
A word cloud of the key skills present in the dataset has been created to visually represent the most important skills contributing to employability. The size of the words in the cloud indicates the significance of each skill in the dataset.

### Results
Model Accuracy: Deep learning models, especially RNNs, demonstrated superior accuracy compared to traditional machine learning methods.
Feature Importance: Key features influencing employability predictions include education level, specific technical skills, and years of experience.
### Acknowledgements
TensorFlow: For providing the deep learning framework.
Scikit-Learn: For machine learning model implementations.
Blog Reference: This project draws inspiration from various sources such as the TensorFlow blog and Scikit-learn documentation.
