.. code:: ipython3

    # Import necessary libraries
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report, confusion_matrix
    
    # Load the Iris dataset
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
    names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
    dataset = pd.read_csv(url, names=names)
    
    # Data preprocessing
    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, 4].values
    
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    
    # Feature scaling
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    
    # Train the model
    classifier = LogisticRegression(random_state=0)
    classifier.fit(X_train, y_train)
    
    # Predict the test set results
    y_pred = classifier.predict(X_test)
    
    # Evaluate the model
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

.. code:: ipython3

    # Data representation with head and tail of the Iris dataset
    print("Head of the Iris dataset:")
    print(dataset.head())
    print("\nTail of the Iris dataset:")
    print(dataset.tail())

.. code:: ipython3

    # Histogram for Sepal Length
    plt.figure(figsize=(10, 7))
    x = dataset["sepal-length"]
    plt.hist(x, bins=20, color="green")
    plt.title("Sepal Length in cm")
    plt.xlabel("Sepal Length (cm)")
    plt.ylabel("Count")
    plt.show()

.. code:: ipython3

    # Histogram for Sepal Width
    plt.figure(figsize=(10, 7))
    x = dataset["sepal-width"]
    plt.hist(x, bins=20, color="green")
    plt.title("Sepal Width in cm")
    plt.xlabel("Sepal Width (cm)")
    plt.ylabel("Count")
    plt.show()

.. code:: ipython3

    # Histogram for Petal Length
    plt.figure(figsize=(10, 7))
    x = dataset["petal-length"]
    plt.hist(x, bins=20, color="green")
    plt.title("Petal Length in cm")
    plt.xlabel("Petal Length (cm)")
    plt.ylabel("Count")
    plt.show()

.. code:: ipython3

    # Histogram for Petal Width
    plt.figure(figsize=(10, 7))
    x = dataset["petal-width"]
    plt.hist(x, bins=20, color="green")
    plt.title("Petal Width in cm")
    plt.xlabel("Petal Width (cm)")
    plt.ylabel("Count")
    plt.show()

.. code:: ipython3

    # Calculate the accuracy of the model
    accuracy = classifier.score(X_test, y_test)
    print(f"Model Accuracy: {accuracy}")

.. code:: ipython3

    # Visualize the data with sepal features included in pair plot
    sns.pairplot(dataset, hue='class', vars=['sepal-length', 'sepal-width', 'petal-length', 'petal-width'])
    plt.show()

.. code:: ipython3

    # Plot confusion matrix with sepal features included
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

.. code:: ipython3

    # Check for outliers in the dataset
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=dataset.drop(columns='class'))
    plt.title('Outliers in the Iris Dataset')
    plt.show()
