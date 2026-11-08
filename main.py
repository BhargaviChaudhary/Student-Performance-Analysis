
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score


# ==========================================
# 1. LOAD AND UNDERSTAND THE DATASET
# ==========================================

data = pd.read_csv("data/student-mat.csv", sep=";")

print("\nFirst 5 Rows:")
print(data.head())

print("\nDataset Information:")
data.info()

print("\nDataset Shape:")
print(data.shape)


# ==========================================
# 2. DATA QUALITY CHECKS
# ==========================================

print("\nStatistical Summary:")
print(data.describe())

print("\nMissing Values:")
print(data.isnull().sum())

print("\nDuplicate Records:")
print(data.duplicated().sum())


# ==========================================
# 3. STUDY TIME VS FINAL GRADE
# ==========================================

print("\nAverage Final Grade by Study Time:")

study_analysis = data.groupby("studytime")["G3"].mean()
print(study_analysis)

plt.figure(figsize=(8, 5))

sns.barplot(
    x=study_analysis.index,
    y=study_analysis.values
)

plt.title("Study Time vs Average Final Grade")
plt.xlabel("Study Time Category")
plt.ylabel("Average Final Grade")

plt.tight_layout()
plt.savefig("visualizations/study_time_vs_grade.png")
plt.show()


# ==========================================
# 4. SCHOOL ABSENCES VS FINAL GRADES
# ==========================================

plt.figure(figsize=(8, 5))

sns.scatterplot(
    x=data["absences"],
    y=data["G3"]
)

plt.title("School Absences vs Final Grades")
plt.xlabel("Number of Absences")
plt.ylabel("Final Grade")

plt.grid(True)
plt.tight_layout()
plt.savefig("visualizations/absences_vs_grade.png")
plt.show()


# ==========================================
# 5. FIRST PERIOD GRADE VS FINAL GRADE
# ==========================================

plt.figure(figsize=(8, 5))

sns.scatterplot(
    x=data["G1"],
    y=data["G3"]
)

plt.title("First Period Grade vs Final Grade")
plt.xlabel("First Period Grade (G1)")
plt.ylabel("Final Grade (G3)")

plt.grid(True)
plt.tight_layout()
plt.savefig("visualizations/g1_vs_g3.png")
plt.show()


# ==========================================
# 6. CORRELATION ANALYSIS
# ==========================================

print("\nCorrelation between Grades:")

grade_correlation = data[["G1", "G2", "G3"]].corr()
print(grade_correlation)


# ==========================================
# 7. MACHINE LEARNING - GRADE PREDICTION
# ==========================================

# Features used for prediction
X = data[["G1", "G2", "studytime", "absences", "failures", "age"]]

# Target variable
y = data["G3"]

# Split dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create and train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict final grades
predictions = model.predict(X_test)

# Evaluate model performance
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("\n========== MACHINE LEARNING RESULTS ==========")
print("Mean Absolute Error (MAE):", round(mae, 2))
print("R2 Score:", round(r2, 3))
print("==============================================")


# ==========================================
# 8. SECOND MODEL - WITHOUT PREVIOUS GRADES
# ==========================================

# Select features without G1 and G2
X_new = data[["studytime", "absences", "failures", "age"]]

# Target variable
y_new = data["G3"]

# Split the data
X_train_new, X_test_new, y_train_new, y_test_new = train_test_split(
    X_new, y_new, test_size=0.2, random_state=42
)

# Create and train the second model
model_new = LinearRegression()
model_new.fit(X_train_new, y_train_new)

# Make predictions
predictions_new = model_new.predict(X_test_new)

# Evaluate the second model
mae_new = mean_absolute_error(y_test_new, predictions_new)
r2_new = r2_score(y_test_new, predictions_new)

print("\n========== SECOND MODEL RESULTS ==========")
print("Mean Absolute Error:", round(mae_new, 2))
print("R2 Score:", round(r2_new, 3))
print("==========================================")


# ==========================================
# 9. MODEL COMPARISON VISUALIZATION
# ==========================================

model_names = ["Model 1\n(With G1 and G2)", "Model 2\n(Without G1 and G2)"]

mae_values = [mae, mae_new]
r2_values = [r2, r2_new]

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# MAE Comparison
axes[0].bar(model_names, mae_values)
axes[0].set_title("Mean Absolute Error Comparison")
axes[0].set_ylabel("MAE (Lower is Better)")

for i, value in enumerate(mae_values):
    axes[0].text(i, value + 0.05, f"{value:.2f}", ha="center")

# R2 Score Comparison
axes[1].bar(model_names, r2_values)
axes[1].set_title("R2 Score Comparison")
axes[1].set_ylabel("R2 Score (Higher is Better)")
axes[1].set_ylim(0, 1)

for i, value in enumerate(r2_values):
    axes[1].text(i, value + 0.02, f"{value:.3f}", ha="center")

plt.suptitle("Student Grade Prediction - Model Comparison")
plt.tight_layout()

plt.savefig("visualizations/model_comparison.png")
plt.show()


# ==========================================
# 10. PREDICT FINAL GRADE FOR A NEW STUDENT
# ==========================================

# Example student information
new_student = pd.DataFrame({
    "G1": [12],
    "G2": [14],
    "studytime": [3],
    "absences": [2],
    "failures": [0],
    "age": [17]
})

# Predict final grade using Model 1
predicted_grade = model.predict(new_student)

print("\n========== NEW STUDENT PREDICTION ==========")
print("Predicted Final Grade:", round(predicted_grade[0], 2))
print("============================================")
