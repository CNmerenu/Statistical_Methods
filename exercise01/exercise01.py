"""
This file solves the first exercise in statistical methods.
To run this file, follow these steps:
1. Make sure every dependency is installed via `pip install -r requirements.txt`
   You can use the `--user` flag to install these packages for your account only.
2. Make sure the `soilrespiration.csv` is in the same directory as this file.
3. Change into the directory of this file and run `python exercise01.py`
"""
import matplotlib.pyplot as plt
import numpy as np
from pandas import read_csv


def linear_estimation(x, β1, β2):
    """
    Estimates the value of the linear function with parameters β1, β2 and the argument x.
    β1, β2 ...  scalar values
    x ... vector or scalar
    """
    return β1 + β2 * x


def estimate_params_with_grid(x, y, β1_range, β2_range, cost, model=linear_estimation):
    """
    x ... vector with input data
    y ... vector with output data
    β1_range ... list with possible values for β1
    β2_range ... list with possible values for β2
    cost ... function that computes the cost of an estimation
    model
    """
    β1_min = β2_min = None
    c_min = np.infty

    for β1 in β1_range:
        for β2 in β2_range:
            y_ = model(x, β1, β2)
            c = cost(y, y_)
            if c < c_min:
                c_min = c
                β1_min = β1
                β2_min = β2

    return β1_min, β2_min


def square_error(y, y_):
    """
    Computes the squared error of an estimation y_ off the real data y.
    """
    return ((y - y_) ** 2).sum()


def abs_error(y, y_):
    """
    Computes the absolute error of an estimation y_ off the real data y.
    """
    return np.abs(y - y_).sum()


def quartic_error(y, y_):
    """
    Computes the quartic error of an estimation y_ off the real data y.
    """
    return ((y - y_) ** 4).sum()


if __name__ == "__main__":
    # read the data and display it
    soil_resp_dataframe = read_csv("soilrespiration.csv")
    plt.scatter(soil_resp_dataframe.temp, soil_resp_dataframe.resp, color="k")
    plt.xlabel("Temperatur")
    plt.ylabel("log. Bodenatmung")
    legend = []  # following estimations add their title to this list

    # Task 1
    legend.append("quadratisch")
    β1_estim, β2_estim = estimate_params_with_grid(
        x=soil_resp_dataframe.temp,
        y=soil_resp_dataframe.resp,
        β1_range=np.linspace(-1, 0, num=100),
        β2_range=np.linspace(0, 1, num=100),
        cost=square_error,
    )
    print(f"β1≈{round(β1_estim, 3)}, β2≈{round(β2_estim, 3)}")
    plt.plot(
        soil_resp_dataframe.temp,
        linear_estimation(soil_resp_dataframe.temp, β1_estim, β2_estim),
    )

    # Task 2
    X = np.column_stack(
        (np.ones(soil_resp_dataframe.temp.size), soil_resp_dataframe.temp)
    )
    β_estim = np.array([β1_estim, β2_estim])

    # @ is the symbol for matrix multiplication
    Xβ = X @ β_estim
    XX = X.T @ X
    XX_inv = np.linalg.inv(XX)

    # Task 3
    legend.append("absolut")
    β1_estim, β2_estim = estimate_params_with_grid(
        x=soil_resp_dataframe.temp,
        y=soil_resp_dataframe.resp,
        β1_range=np.linspace(-1, 0, num=100),
        β2_range=np.linspace(0, 1, num=100),
        cost=abs_error,
    )
    plt.plot(
        soil_resp_dataframe.temp,
        linear_estimation(soil_resp_dataframe.temp, β1_estim, β2_estim),
    )
    print(f"β1≈{round(β1_estim, 3)}, β2≈{round(β2_estim, 3)}")

    legend.append("quartisch")
    β1_estim, β2_estim = estimate_params_with_grid(
        x=soil_resp_dataframe.temp,
        y=soil_resp_dataframe.resp,
        β1_range=np.linspace(-1, 0, num=100),
        β2_range=np.linspace(0, 1, num=100),
        cost=quartic_error,
    )
    plt.plot(
        soil_resp_dataframe.temp,
        linear_estimation(soil_resp_dataframe.temp, β1_estim, β2_estim),
    )
    print(f"β1≈{round(β1_estim, 3)}, β2≈{round(β2_estim, 3)}")

    legend.append("Rohdaten")
    plt.legend(legend)
    plt.show()
