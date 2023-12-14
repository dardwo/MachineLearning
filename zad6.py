import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

def prepare_data(data):
    features = data['X'].values.reshape(-1, 1)
    target = data['y'].values
    return features, target

def create_model(degree):
    return make_pipeline(PolynomialFeatures(degree), LinearRegression())

def plot_results(X, y, label):
    plt.plot(X, y, label=label)

def scatter_plot(X, y, label):
    plt.scatter(X, y, label=label)

def show_plot(title, xlabel, ylabel):
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.show()

def main():
    path = 'data6.tsv'
    data = pd.read_csv(path, sep='\t', header=None, names=['X', 'y'])

    features, target = prepare_data(data)

    deg = [1, 2, 5]
    plt.figure(figsize=(12, 6))

    for degree in deg:
        model = create_model(degree)
        model.fit(features, target)

        X_plot = np.linspace(min(features), max(features), 100)
        y_plot =model.predict(X_plot.reshape(-1, 1))

        plot_results(X_plot, y_plot, label=f'Degree {degree}')

    scatter_plot(features, target, label='Points')
    show_plot('Polynomial Regression for degrees 1, 2 and 5', 'X', 'y')

if __name__ == "__main__":
    main()
