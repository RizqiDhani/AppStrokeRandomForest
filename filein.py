import streamlit as st 
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder

st.title('Streamlit Example')

st.write("""
# Explore different classifiers and datasets
Which one is the best?
""")

# Sidebar for dataset selection
dataset_option = st.sidebar.radio("Select Dataset", ('Built-in Datasets', 'Upload your own dataset'))

if dataset_option == 'Built-in Datasets':
    dataset_name = st.sidebar.selectbox(
        'Select Dataset',
        ('Iris', 'Breast Cancer', 'Wine')
    )

    st.write(f"## {dataset_name} Dataset")

    def get_dataset(name):
        data = None
        if name == 'Iris':
            data = datasets.load_iris()
        elif name == 'Wine':
            data = datasets.load_wine()
        else:
            data = datasets.load_breast_cancer()
        X = data.data
        y = data.target
        return X, y

    X, y = get_dataset(dataset_name)
else:
    uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        
        # Convert categorical columns to numerical using Label Encoding
        for col in df.columns[:-1]:  # Iterate all columns except the target
            if df[col].dtype == 'object':  # Check if column is categorical
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col])

        st.write(df.head())
        
        # Assuming the last column is the target variable
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values

        # Convert target column if it's not numeric
        if y.dtype == 'object':
            le_target = LabelEncoder()
            y = le_target.fit_transform(y)

        st.write('Shape of dataset:', X.shape)
        st.write('Number of classes:', len(np.unique(y)))

classifier_name = st.sidebar.selectbox(
    'Select classifier',
    ('KNN', 'SVM', 'Random Forest', 'Logistic Regression')
)

def add_parameter_ui(clf_name):
    params = dict()
    if clf_name == 'SVM':
        C = st.sidebar.slider('C', 0.01, 10.0)
        params['C'] = C
    elif clf_name == 'KNN':
        K = st.sidebar.slider('K', 1, 15)
        params['K'] = K
    elif clf_name == 'Random Forest':
        max_depth = st.sidebar.slider('max_depth', 2, 15)
        params['max_depth'] = max_depth
        n_estimators = st.sidebar.slider('n_estimators', 1, 100)
        params['n_estimators'] = n_estimators
    elif clf_name == 'Logistic Regression':
        C = st.sidebar.slider('C', 0.01, 10.0)
        params['C'] = C
    return params

params = add_parameter_ui(classifier_name)

def get_classifier(clf_name, params):
    clf = None
    if clf_name == 'SVM':
        clf = SVC(C=params['C'])
    elif clf_name == 'KNN':
        clf = KNeighborsClassifier(n_neighbors=params['K'])
    elif clf_name == 'Random Forest':
        clf = RandomForestClassifier(n_estimators=params['n_estimators'], 
                                     max_depth=params['max_depth'], random_state=1234)
    elif clf_name == 'Logistic Regression':
        clf = LogisticRegression(C=params['C'], max_iter=1000)
    return clf

clf = get_classifier(classifier_name, params)

# CLASSIFICATION
if dataset_option == 'Built-in Datasets' or (dataset_option == 'Upload your own dataset' and uploaded_file is not None):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)

    st.write(f'Classifier = {classifier_name}')
    st.write(f'Accuracy =', acc)

    # Display Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
    ax.set_xlabel("Predicted labels")
    ax.set_ylabel("True labels")
    ax.set_title("Confusion Matrix")
    st.pyplot(fig)

    # PLOT DATASET
    if X.shape[1] >= 2:  # Ensure there are at least 2 features
        pca = PCA(2)
        X_projected = pca.fit_transform(X)

        x1 = X_projected[:, 0]
        x2 = X_projected[:, 1]

        fig = plt.figure()
        plt.scatter(x1, x2,
                    c=y, alpha=0.8,
                    cmap='viridis')

        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.colorbar()

        st.pyplot(fig)
    else:
        st.write("Not enough features for PCA visualization.")
