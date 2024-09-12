from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt

import seaborn as sns

if __name__ == '__main__':
    data = load_wine()


    X = data.data
    y = data.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    clf = DecisionTreeClassifier()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    confusion = confusion_matrix(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)

    plt.figure(figsize=(18, 6))
    sns.heatmap(confusion, annot=True, fmt="d", cmap="Greens", cbar=False)
    plt.xlabel("Valores Preditos")
    plt.ylabel("Valores Reais")
    plt.title(f"Matriz de Confusão (Precisão: {accuracy:.2f})")
    plt.savefig("confusion_tree.png")

