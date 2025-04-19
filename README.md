# Income Classification using Machine Learning Models

This project applies various supervised learning algorithms on the UCI Adult Income Dataset (also known as the Census Income dataset) to predict whether an individual's income exceeds $50K per year based on census data.

## 📊 Objective

To train, evaluate, and compare the performance of multiple machine learning classifiers using comprehensive metrics, cross-validation, and visualizations.

## 🧠 Models Implemented

- K-Nearest Neighbors (KNN)
- Decision Tree
- Random Forest
- AdaBoost
- Stacking Classifier

## 🛠️ Project Structure

1. **Data Preparation**
   - Load and preprocess the dataset
   - Handle missing values
   - Encode categorical variables
   - Feature scaling (if required)
   - Train-test split (80% training / 20% testing)

2. **Model Training**
   - Train each model with multiple hyperparameter configurations

3. **Evaluation Metrics**
   - Confusion Matrix
   - Precision, Recall, F1 Score
   - ROC Curve and AUC
   - Cross-Validation (5-fold)
   - Accuracy and Error Metrics

4. **Visualization**
   - ROC Curves
   - Accuracy and Error plots for different hyperparameter configurations

## 📂 Deliverables

Each model includes:

- Evaluation reports (Confusion Matrix, Precision, Recall, F1 Score, AUC)
- Cross-Validation performance
- Hyperparameter tuning results
- Accuracy/Error metric visualizations

## 🔍 Model Highlights

### 1. K-Nearest Neighbors (KNN)
- Evaluated k = [1, 3, 5, 7, 9]
- Identified the best-performing value of k based on cross-validated metrics

### 2. Decision Tree
- Explored different tree depths and `min_samples_leaf` values
- Selected optimal configuration using evaluation metrics

### 3. Random Forest
- Tuned number of trees (`n_estimators`) and tree depth
- Reported the best-performing ensemble configuration

### 4. AdaBoost
- Tested multiple base estimators and numbers of boosting rounds
- Chose the best model based on AUC and F1 Score

### 5. Stacking
- Combined KNN, Decision Tree, and Random Forest as base learners
- Used Logistic Regression as meta-learner
- Evaluated stacked model’s overall performance

## 📌 Dataset

- **Source:** [UCI Machine Learning Repository - Adult Dataset](https://archive.ics.uci.edu/ml/datasets/adult)
- Contains demographic data and income labels

## 🧪 Libraries Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn

## 🚀 How to Run

```bash
# Clone the repository
git clone https://github.com/your-username/income-classification-ml-models.git
cd income-classification-ml-models

# (Optional) Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the notebook or script
