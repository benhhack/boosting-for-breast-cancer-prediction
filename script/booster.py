import argparse
import pandas as pd

def main():
    parser = argparse.ArgumentParser(description="Train an booster model")
    parser.add_argument("--data", type=str, required=True, help="Path to a dataset")

    args = parser.parse_args()


    # getting the dataset and setting features and target
    dataset = pd.read_csv(f"{args.data}")

    dataset["diagnosis"] = dataset["diagnosis"].map({"M": 1, "B":0})

    features = list(dataset.columns)
    features.remove("diagnosis")
    X, y = dataset[features].values, dataset["diagnosis"].values

    # Splitting the dataset into the Training set and Test set
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # XGBoost training
    from xgboost import XGBClassifier
    classifier = XGBClassifier()
    classifier.fit(X_train, y_train)



    # applying k-fold cross validation
    print("--- Cross Validation ---")
    from sklearn.model_selection import cross_val_score
    accuracies = cross_val_score(estimator=classifier, X=X_train, y=y_train, cv=10)
    print("Accuracy: {:.2f} %".format(accuracies.mean() * 100))
    print("Standard Deviation: {:.2f} %".format(accuracies.std() * 100))

    # confusion matrix creation
    from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
    y_pred = classifier.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    print("F1 Score: {:.3f}".format(f1_score(y_test, y_pred)))
    accuracy_score(y_test, y_pred)
    print("Test Accuracy {:.4f}".format(accuracy_score(y_test, y_pred)))

if __name__ == "__main__":
    main()
