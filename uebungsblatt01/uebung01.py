from pandas import read_csv
import numpy as np


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


if __name__ == "__main__":
    # Task 1
    soil_resp_dataframe = read_csv("soilrespiration.csv")
    β1_estim, β2_estim = estimate_params_with_grid(
        x=soil_resp_dataframe.temp,
        y=soil_resp_dataframe.resp,
        β1_range=np.linspace(-1, 0, num=100),
        β2_range=np.linspace(0, 1, num=100),
        cost=square_error,
    )
    print(f"β1≈{round(β1_estim, 3)}, β2≈{round(β2_estim, 3)}")

    # Task 2
    X = np.column_stack(
        (np.ones(soil_resp_dataframe.temp.size), soil_resp_dataframe.temp)
    )
    β_estim = np.array([β1_estim, β2_estim])

    # @ is the symbol for matrix multiplication
    Xβ = X @ β_estim
    XX = X.T @ X
    XX_inv = np.linalg.inv(XX)
