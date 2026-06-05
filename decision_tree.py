import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# Load the dataset
df = pd.read_csv('Student_performance_data_streamlined.csv')

# Remove unnecessary columns
drop_columns = [
    'StudentID',
    'Gender_Label',
    'Ethnicity_Label',
    'ParentalEducation_Label',
    'ParentalSupport_Label',
    'GradeClass_Label'
]
df= df.drop(columns=drop_columns)

# Features and target variable
X = df.drop('GradeClass', axis=1)
y = df['GradeClass']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training data:", X_train.shape)
print("Testing data:", X_test.shape)

# Experiment with different depth values
Max_depth_values = [3, 5, 7, 10, None]
accuracies = []

print("\nDecision Tree Results:")
print("-" * 50)

for depth in Max_depth_values:
    model = DecisionTreeClassifier(max_depth=depth, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    accuracies.append(accuracy)

    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')

    f1 = f1_score(y_test, y_pred, average='weighted')

    print(f"\nMax Depth = {depth}")
    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")

# Find the best depth based on accuracy
best_depth = Max_depth_values[accuracies.index(max(accuracies))]

print(f"\nBest Max Depth: {best_depth} with Accuracy: {max(accuracies):.4f}")

# Train the best model
best_model = DecisionTreeClassifier(max_depth=best_depth, random_state=42)

best_model.fit(X_train, y_train)

y_pred = best_model.predict(X_test)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=best_model.classes_)

disp.plot()
plt.title("Decision Tree Confusion Matrix")
plt.savefig("decision_tree_confusion_matrix.png")
plt.show()

# Accuracy Vs Max Depth
plt.figure(figsize=(8, 5))
plt.plot(Max_depth_values, accuracies, marker='o')
plt.xlabel('Max Depth')
plt.ylabel('Accuracy')
plt.title('Decision Tree Accuracy vs Max Depth')
plt.grid(True)
plt.savefig("decision_tree_accuracy.png")
plt.show()

# Decision Tree Visualization

plt.figure(figsize=(20,10))

plot_tree(
    best_model,
    filled=True,
    feature_names=X.columns,
    class_names=[str(x) for x in best_model.classes_]
)

plt.title("Decision Tree Structure")

plt.savefig("decision_tree_structure.png")

plt.show()
