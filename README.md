# Titanic_survival_prediction_project


<img src="https://static1.squarespace.com/static/5006453fe4b09ef2252ba068/t/5090b249e4b047ba54dfd258/1351660113175/TItanic-Survival-Infographic.jpg?format=1500w">

## Overview
The Titanic Survival Prediction project aims to predict the survival of passengers on the Titanic using machine learning algorithms. This project leverages various features of the Titanic dataset to build a predictive model that can determine the likelihood of a passenger surviving the disaster.

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [DatasetFeatures](#Datasetfeatures)
- [Installation](#installation)
- [Usage](#usage)
- [Models Used](#models-used)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Dataset
The dataset used in this project is the Titanic dataset from Kaggle. It contains information about the passengers on the Titanic, including their demographics, ticket information, and whether they survived the disaster.

### Dataset Features:
- **PassengerId**: Unique ID for each passenger
- **Survived**: Survival (0 = No, 1 = Yes)
- **Pclass**: Ticket class (1 = 1st, 2 = 2nd, 3 = 3rd)
- **Name**: Name of the passenger
- **Sex**: Sex
- **Age**: Age in years
- **SibSp**: Number of siblings/spouses aboard the Titanic
- **Parch**: Number of parents/children aboard the Titanic
- **Ticket**: Ticket number
- **Fare**: Passenger fare
- **Cabin**: Cabin number
- **Embarked**: Port of Embarkation (C = Cherbourg, Q = Queenstown, S = Southampton)

## Installation
To run this project, you need to have Python installed on your system along with the following libraries:
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Jupyter Notebook

You can install the required libraries using pip:
```sh
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
```
## Usage:
### 1.Clone the repository:
```sh
git clone https://github.com/JVENKATAPHANISAI/Titanic-survival-prediction-project.git
```
- This command copies the project directly to  your local machine.

###  2.Navigate to the project directory:
```sh
cd TitanicSurvivalPrediction
```
- This changes your current directory to the project folder.

### 3.Install the required libraries (if not already installed):
```sh
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
```
- This ensures all necessary Python libraries are installed.
### 4.Open the Jupyter Notebook:
```sh
jupyter notebook TitanicClassification.ipynb
```
- This starts the Jupyter Notebook server and opens the notebook in your web browser.

### Run the notebook cells:  
- In your web browser, you will see the Jupyter interface with a list of files. Click on TitanicClassification.ipynb to open it.
-Execute each cell in the notebook sequentially to perform the data analysis, train the machine learning models, and generate predictions.

## Models Used:
The following machine learning models were used in this project:

- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)

## Results
- The performance of each model was evaluated using accuracy, precision, recall, and F1 score. - The Random Forest Classifier provided the best results with an accuracy of X%.

## Contributing
- Contributions are welcome! Please feel free to submit a Pull Request.

## License
- This project is licensed under the MIT License.

## Acknowledgements
- Kaggle for providing the Titanic dataset
- Scikit-learn documentation
- Matplotlib and Seaborn for data visualization
- Image Source
