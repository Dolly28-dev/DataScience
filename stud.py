import streamlit as st 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
st.title("Students Performance Prediction Using Machine Learning")
st.write("This application performs data cleaning and visualization on the Student Performance dataset.")
df = pd.read_csv("student_performance_dataset.csv")

#Data Cleaning
st.subheader("Dataset Information")
st.write("Rows :", df.shape[0])
st.write("Columns :", df.shape[1])
st.write(df.dtypes)

st.subheader("Dataset Preview")
st.dataframe(df.head())
st.dataframe(df.tail())

st.subheader("Dataset Shape")
st.write(df.shape)

st.subheader("Dataset Columns")
st.write(df.columns)

st.subheader("Missing Values")
st.write(df.isnull().sum())

st.subheader("Duplicate Values")
st.write(df.duplicated().sum())

st.subheader("Removing Duplicate Values")
df = df.drop_duplicates()
st.write("Duplicate values removed successfully")

st.write("New Dataset Shape:")
st.write(df.shape)

st.write("Total Missing Values:")
st.write(df.isnull().sum().sum())
num_cols = df.select_dtypes(include=["int64","float64"]).columns

for col in num_cols:
    df[col] = df[col].fillna(df[col].median())
cat_cols = df.select_dtypes(include="object").columns

for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])
st.subheader("Missing Values After Cleaning")

st.write(df.isnull().sum())
st.subheader("Data Types After Cleaning")
st.write(df.dtypes)

st.subheader("Removing Extra Spaces from Text Columns")

df['Gender'] = df['Gender'].str.strip()
df['Extracurricular'] = df['Extracurricular'].str.strip()
df['Internet_Access'] = df['Internet_Access'].str.strip()
df['Parent_Education'] = df['Parent_Education'].str.strip()

st.write("Extra spaces removed successfully")
st.dataframe(df.head())

st.subheader("Gender Unique Values")
df["Gender"] = df["Gender"].replace({
    "male": "Male",
    "FEMALE": "Female"
})
st.write(df["Gender"].unique())

df["Internet_Access"] = df["Internet_Access"].str.strip()
df["Internet_Access"] = df["Internet_Access"].replace({
    "yes": "Yes",
    "NO": "No"
})
st.write(df["Internet_Access"].unique())

df["Extracurricular"] = df["Extracurricular"].str.strip()
df["Extracurricular"] = df["Extracurricular"].replace({
    "yes": "Yes",
    "NO": "No"
})
st.write(df["Extracurricular"].unique())

df["Parent_Education"] = df["Parent_Education"].str.strip()
df["Parent_Education"] = df["Parent_Education"].replace({
    "graduate": "Graduate"
})
st.write(df["Parent_Education"].unique())

st.subheader("Attendance Cleaning")
st.write("Maximum Attendance Before Cleaning:")
st.write(df["Attendance"].max())
df.loc[df["Attendance"] > 100, "Attendance"] = df["Attendance"].median()
st.write("Maximum Attendance After Cleaning:")
st.write(df["Attendance"].max())

st.subheader("Previous Marks Cleaning")
st.write("Minimum Previous Marks Before Cleaning:")
st.write(df["Previous_Marks"].min())
df.loc[df["Previous_Marks"] < 0, "Previous_Marks"] = df["Previous_Marks"].median()
st.write("Minimum Previous Marks After Cleaning:")
st.write(df["Previous_Marks"].min())

st.subheader("Dataset Information")
st.write(df.dtypes)

st.subheader("Gender Count")
st.write(df["Gender"].value_counts())

st.subheader("Parent Education Count")
st.write(df["Parent_Education"].value_counts())

st.subheader("Internet Access Count")
st.write(df["Internet_Access"].value_counts())

st.subheader("Final Exam Marks")
st.write("Mean:", df["Final_Exam_Marks"].mean())
st.write("Maximum:", df["Final_Exam_Marks"].max())
st.write("Minimum:", df["Final_Exam_Marks"].min())

st.subheader("Age Range")
st.write("Minimum Age:", df["Age"].min())
st.write("Maximum Age:", df["Age"].max())

st.subheader("Study Hours Range")
st.write("Minimum:", df["Study_Hours"].min())
st.write("Maximum:", df["Study_Hours"].max())

st.subheader("Assignments Completed Range")
st.write("Minimum:", df["Assignments_Completed"].min())
st.write("Maximum:", df["Assignments_Completed"].max())

st.subheader("Internal Marks Range")
st.write("Minimum:", df["Internal_Marks"].min())
st.write("Maximum:", df["Internal_Marks"].max())

st.subheader("Final Exam Marks Range")
st.write("Minimum:", df["Final_Exam_Marks"].min())
st.write("Maximum:", df["Final_Exam_Marks"].max())

st.subheader("Final Dataset Summary")
st.write(df.describe())
clean_csv = df.to_csv(index=False)

st.download_button(
    label="Download Clean Dataset",
    data=clean_csv,
    file_name="student_performance_cleanedsteamlit.csv",
    mime="text/csv"
)

#Visualization

st.subheader("Gender Distribution")
gender_count = df["Gender"].value_counts()
fig, ax = plt.subplots()
ax.bar(gender_count.index, gender_count.values)
ax.set_title("Gender Distribution")
ax.set_xlabel("Gender")
ax.set_ylabel("Count")
st.pyplot(fig)

st.subheader("Internet Access Distribution")
internet = df["Internet_Access"].value_counts()
fig, ax = plt.subplots()
ax.pie(internet.values,
       labels=internet.index,
       autopct="%1.1f%%",
       startangle=90)
ax.set_title("Internet Access")
st.pyplot(fig)

st.subheader("Parent Education Distribution")
parent = df["Parent_Education"].value_counts()
fig, ax = plt.subplots()
ax.barh(parent.index, parent.values)
ax.set_title("Parent Education Distribution")
ax.set_xlabel("Count")
ax.set_ylabel("Parent Education")
st.pyplot(fig)

st.subheader("Study Hours Distribution")
fig, ax = plt.subplots()
ax.hist(df["Study_Hours"], bins=10)
ax.set_title("Study Hours Distribution")
ax.set_xlabel("Study Hours")
ax.set_ylabel("Frequency")
st.pyplot(fig)

st.subheader("Study Hours and Final Exam Marks Comparison")
x = np.arange(len(df))
width = 0.4
fig, ax = plt.subplots(figsize=(15,6))
ax.bar(x - width/2,df["Study_Hours"],width,color="blue",label="Study Hours")
ax.bar(x + width/2,df["Final_Exam_Marks"],width,color="red",label="Final Exam Marks")
ax.set_title("Study Hours vs Final Exam Marks")
ax.set_xlabel("Students")
ax.set_ylabel("Values")
ax.legend()
st.pyplot(fig)

st.subheader("Attendance Trend")
fig, ax = plt.subplots()
ax.plot(df["Attendance"])
ax.set_title("Attendance Trend")
ax.set_xlabel("Students")
ax.set_ylabel("Attendance")
st.pyplot(fig) 

st.subheader("Previous Marks and Final Exam Marks")
fig, ax = plt.subplots(figsize=(10,6))
ax.scatter(range(len(df)),df["Previous_Marks"],color="blue",marker="o",s=60,label="Previous Marks")
ax.scatter(range(len(df)),df["Final_Exam_Marks"],color="red",marker="*",s=50,label="Final Exam Marks")
ax.set_title("Previous Marks and Final Exam Marks")
ax.set_xlabel("Student Index")
ax.set_ylabel("Marks")
ax.legend()
st.pyplot(fig)

st.subheader("Final Exam Marks Box Plot")
fig, ax = plt.subplots()
ax.boxplot(df["Final_Exam_Marks"])
ax.set_title("Final Exam Marks")
st.pyplot(fig)

st.subheader("Internal Marks Area Chart")
fig, ax = plt.subplots()
ax.fill_between(range(len(df)), df["Internal_Marks"])
ax.set_title("Internal Marks")
ax.set_xlabel("Students")
ax.set_ylabel("Marks")
st.pyplot(fig)

st.header("Final Exam Marks Distribution")
fig, ax = plt.subplots(figsize=(8,5))
sns.histplot(data=df,x="Final_Exam_Marks",bins=10,kde=True,ax=ax)
ax.set_title("Final Exam Marks Distribution")
ax.set_xlabel("Final Exam Marks")
ax.set_ylabel("Frequency")
st.pyplot(fig)

st.header("Study Hours vs Final Exam Marks")
fig, ax = plt.subplots(figsize=(8,5))
sns.scatterplot(data=df,x="Study_Hours",y="Final_Exam_Marks",hue="Gender",ax=ax)
ax.set_title("Study Hours vs Final Exam Marks")
ax.set_xlabel("Study Hours")
ax.set_ylabel("Final Exam Marks")
st.pyplot(fig)

#Machine Learning
st.header("Machine Learning")

#Logistic Regression
le = LabelEncoder()
df["Gender"] = le.fit_transform(df["Gender"])
df["Extracurricular"] = le.fit_transform(df["Extracurricular"])
df["Internet_Access"] = le.fit_transform(df["Internet_Access"])
df["Parent_Education"] = le.fit_transform(df["Parent_Education"])
st.subheader("Encoded Dataset")
st.dataframe(df.head())

st.subheader("Create Target Variable")
df["Performance"] = (df["Final_Exam_Marks"] >= 50).astype(int)
st.write(df[["Final_Exam_Marks", "Performance"]].head())
st.write(df[["Final_Exam_Marks", "Performance"]].tail())

st.subheader("Select Features and Target")
X = df.drop(["Student_ID", "Final_Exam_Marks", "Performance"], axis=1)
y = df["Performance"]
st.write("Features (X)")
st.write(X.head())
st.write(X.tail())
st.write("Target (y)")
st.write(y.head())
st.write(y.tail())

st.subheader("Train Test Split")
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)
st.write("Training Data Shape:", X_train.shape)
st.write("Testing Data Shape:", X_test.shape)

st.subheader("Train Logistic Regression Model")
model = LogisticRegression(max_iter=1000,class_weight="balanced")
model.fit(X_train, y_train)
st.success("Model Trained Successfully")

st.subheader("Prediction")
y_pred = model.predict(X_test)
st.write("Predicted Values:")
st.write(y_pred)

st.header("Logistic Regression")
st.subheader("Model Accuracy")
lr_accuracy = accuracy_score(y_test, y_pred)
st.write("Accuracy:", round(lr_accuracy * 100, 2), "%")

st.subheader("Confusion Matrix")
cm = confusion_matrix(y_test, y_pred)
st.write(cm)

st.subheader("Classification Report")
report = classification_report(y_test, y_pred)
st.text(report)

st.subheader("Actual vs Predicted")
result = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": y_pred
})
st.dataframe(result)

#Decision Tree

st.header("Decision Tree")
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)

dt_pred = dt_model.predict(X_test)

st.subheader("Decision Tree Accuracy")
dt_accuracy = accuracy_score(y_test, dt_pred)
st.write("Accuracy:",round(dt_accuracy*100,2),"%")

st.subheader("Dicision Tree Confusion")
dt_cm = confusion_matrix(y_test, dt_pred)
st.write(dt_cm)

st.subheader("Decision Tree Classification Report")
st.text(classification_report(y_test, dt_pred))

st.subheader("Decision Tree Actual vs Prediction")
dt_result = pd.DataFrame({"Actual":y_test.values, "Prediction":dt_pred})
st.dataframe(dt_result)

#SVM

st.header("Support Vector Machine (SVM)")

svm = SVC()

svm.fit(X_train, y_train)

y_pred_svm = svm.predict(X_test)

svm_accuracy = accuracy_score(y_test, y_pred_svm)

st.subheader("SVM Accuracy")
st.success(f"Accuracy: {svm_accuracy*100:.2f} %")

st.subheader("SVM Confusion Matrix")
st.write(confusion_matrix(y_test, y_pred_svm))

st.subheader("SVM Classification Report")
st.text(classification_report(y_test, y_pred_svm))

st.subheader("SVM Actual vs Prediction")
svm_result = pd.DataFrame({"Actual": y_test.values,"Prediction": y_pred_svm})
st.dataframe(svm_result)

#Compare
st.header("Algorithm Comparison")
comparison = pd.DataFrame({
    "Algorithm": [
        "Logistic Regression",
        "Decision Tree",
        "SVM"
    ],
    "Accuracy (%)": [
        lr_accuracy * 100,
        dt_accuracy * 100,
        svm_accuracy * 100,
    ]
})

st.dataframe(comparison)

# Student Performance Prediction

st.header("Predict Student Performance")

gender = st.selectbox("Gender", ["Male", "Female"])
age = st.number_input("Age", min_value=15, max_value=30, value=20)
study_hours = st.number_input("Study Hours", min_value=0, max_value=15, value=5)
attendance = st.number_input("Attendance", min_value=0, max_value=100, value=80)
previous_marks = st.number_input("Previous Marks", min_value=0, max_value=100, value=60)
assignments = st.number_input("Assignments Completed", min_value=0, max_value=20, value=10)
internal_marks = st.number_input("Internal Marks", min_value=0, max_value=100, value=50)
extra = st.selectbox("Extracurricular", ["Yes", "No"])
internet = st.selectbox("Internet Access", ["Yes", "No"])
parent = st.selectbox("Parent Education", ["Graduate", "Postgraduate", "School"])

# Encoding
gender = 1 if gender == "Male" else 0
extra = 1 if extra == "Yes" else 0
internet = 1 if internet == "Yes" else 0

if parent == "Graduate":
    parent = 0
elif parent == "Postgraduate":
    parent = 1
else:
    parent = 2

if st.button("Predict Performance"):

    input_data = pd.DataFrame({
        "Gender":[gender],
        "Age":[age],
        "Study_Hours":[study_hours],
        "Attendance":[attendance],
        "Previous_Marks":[previous_marks],
        "Assignments_Completed":[assignments],
        "Internal_Marks":[internal_marks],
        "Extracurricular":[extra],
        "Internet_Access":[internet],
        "Parent_Education":[parent]
    })

    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.success("Prediction: Good Performance (Pass)")
        st.write("The student is likely to score 50 or more marks in the final exam.")
    else:
        st.error("Prediction: Poor Performance (Fail)")
        st.write("The student is likely to score below 50 marks in the final exam.")

