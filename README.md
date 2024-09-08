# Text Classification Project: Educational vs Non-Educational Content

This project is focused on classifying text data into two categories: **educational** and **non-educational** content. The dataset used consists of text data from various websites that fall into one of these two categories. The classification models were built using different machine learning algorithms including Logistic Regression, Support Vector Machines (SVM), and Naive Bayes. The project involves cleaning and pre-processing the text data, followed by training and testing multiple models to achieve high classification accuracy.

## Dataset

The dataset consists of text from different websites, grouped into educational (edu) and non-educational (nonedu) content. The dataset is divided into training and validation sets. Below is a summary of the data used:

- **Educational Content**: Text extracted from various educational websites.
- **Non-Educational Content**: Text extracted from non-educational websites.
- The dataset includes a total of over **520,000** text samples.
  
Columns in the dataset:
- **subjects**: The subject category of the text (e.g., English, Social Studies)
- **text**: The actual text data
- **file_name**: File name of the text
- **host**: Website from which the data was sourced
- **url**: URL of the text source
- **cate**: Category (educational or non-educational)

## Key Steps in the Project

### Data Preprocessing
- Text cleaning and tokenization.
- Removal of stopwords using an extended stopword list.
- Data split into training, testing, and validation sets.
- Use of TF-IDF (Term Frequency-Inverse Document Frequency) for text vectorization to convert raw text into numerical features.

### Models Used
1. **Logistic Regression**:
   - Vectorized the text using `TfidfVectorizer` and applied Logistic Regression for binary classification.
   - Saved the model using `pickle` for future predictions.
   - Achieved an accuracy of **98.69%** on the test set and **98.06%** on the validation set.

2. **Support Vector Machines (SVM)**:
   - Used `TfidfVectorizer` with SVM classifier.
   - Achieved high precision and recall metrics on both test and validation sets.
   - The SVM model was saved as `SVM_Binary.sav` for future use.

3. **Naive Bayes (Multinomial)**:
   - Implemented using the `MultinomialNB` classifier.
   - Performed well on the dataset with fast execution times.
   - The model was saved as `NB_Binary.sav`.

### Model Evaluation
The models were evaluated using the following metrics:
- **Precision**
- **Recall**
- **F1-Score**
- **Accuracy**

#### Example Results for Logistic Regression:

          precision    recall  f1-score   support

     edu     0.9781    0.9992    0.9885     37131
  nonedu     0.9989    0.9709    0.9847     28511

accuracy                         0.9869     65642


## Technologies Used
- **Python**: Main programming language for the project.
- **Libraries**: 
  - `pandas`, `numpy`: Data handling and manipulation.
  - `scikit-learn`: Machine learning models and evaluation metrics.
  - `nltk`: Natural Language Processing for tokenization and stopword removal.
  - `pickle`: Saving and loading models for future use.

## How to Use
1. **Data Preprocessing**:
   - Ensure the dataset is in a structured format.
   - Use the provided code to clean and preprocess the data.

2. **Training the Model**:
   - Use the `TfidfVectorizer` to vectorize the text data.
   - Train one of the classifiers (Logistic Regression, SVM, or Naive Bayes).
   - Save the trained model using `pickle` for future predictions.

3. **Prediction**:
   - The `my_prediction()` function allows you to classify new text data into educational or non-educational categories by using a threshold to define the probability of being educational.

4. **Evaluation**:
   - Run the evaluation scripts to check the accuracy, precision, recall, and F1-score of the models on both the test and validation datasets.

## Conclusion
This project successfully implements various classification models to distinguish between educational and non-educational content with high accuracy. By leveraging TF-IDF vectorization and different machine learning algorithms, the models were able to achieve strong performance on both the test and validation sets.

## Future Improvements
- Implement additional models like Random Forest or Gradient Boosting for further exploration.
- Fine-tune the hyperparameters for even better accuracy and generalization.

## Execution Time
- The model training and evaluation were completed in under 6 minutes for each classifier.

---

**Author**: Shudi Zhao  
**Date**: May 8, 2022
