# Student Performance Analysis & Prediction

## Project Overview
This project explores student academic performance using Python, data analysis, data visualization, and machine learning.

The goal is to understand how factors such as study time, absences, previous grades, and past failures relate to students' final mathematics grades. It also compares two machine learning models and demonstrates a sample grade prediction.

## Dataset
- **Source:** [UCI Student Performance Dataset](https://archive.ics.uci.edu/dataset/320/student+performance)
- **Subject:** Mathematics
- **Records:** 395 students
- **Features:** 33 columns

## Technologies Used
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

## Project Workflow

1. Loaded and inspected the student performance dataset.
2. Checked missing values and duplicate records.
3. Explored relationships between study time, absences, and grades.
4. Created visualizations to understand patterns in the data.
5. Performed correlation analysis on previous and final grades.
6. Built two Linear Regression models.
7. Compared model performance using MAE and R² score.
8. Tested a sample prediction for a new student.

## Visualizations

The project includes four visualizations:

1. Study Time vs Final Grade
2. Absences vs Final Grade
3. Previous Grade (G1) vs Final Grade (G3)
4. Machine Learning Model Comparison

All visualization images are available in the `visualizations/` folder.

## Machine Learning Results

Two Linear Regression models were developed and compared.

| Evaluation Metric | Model 1: With G1 and G2 | Model 2: Without G1 and G2 |
|---|---:|---:|
| Mean Absolute Error (MAE) | 1.33 | 3.54 |
| R² Score | 0.782 | 0.045 |

**Observation:** Model 1 performed better because previous grades (G1 and G2) provide useful information for estimating the final grade (G3).

## Sample Prediction

A sample student was given the following details:

- G1: 12
- G2: 14
- Study Time: 3
- Absences: 2
- Failures: 0
- Age: 17

**Predicted Final Grade: 13.59 / 20**

## Project Structure

```text
Student-Performance-Analysis/
│
├── data/
│   └── student-mat.csv
│
├── visualizations/
│   ├── study_time_vs_grade.png
│   ├── absences_vs_grade.png
│   ├── g1_vs_g3.png
│   └── model_comparison.png
│
├── src/
├── main.py
├── README.md
└── .gitignore
```

## How to Run the Project

1. Clone or download this repository.
2. Open the project folder in VS Code.
3. Create and activate a Python virtual environment.
4. Install the required libraries:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

5. Run the program:

```bash
python main.py
```

## Key Learnings

- Working with a real-world dataset.
- Data inspection and exploratory data analysis.
- Data visualization using Python.
- Understanding correlation between variables.
- Building and evaluating Linear Regression models.
- Making predictions using trained machine learning models.

## Limitations

This project uses a single mathematics dataset with 395 student records. Its predictions reflect patterns in this dataset and should not be treated as reliable assessments of individual students or as proof of cause-and-effect relationships.

## Author

**Bhargavi Chaudhary**

B.Tech Computer Science Engineering Student
