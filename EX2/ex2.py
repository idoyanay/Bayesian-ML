import numpy as np
from matplotlib import pyplot as plt
from typing import Callable
import argparse



def polynomial_basis_functions(degree: int) -> Callable:
    """
    Create a function that calculates the polynomial basis functions up to (and including) a degree
    :param degree: the maximal degree of the polynomial basis functions
    :return: a function that receives as input an array of values X of length N and returns the design matrix of the
             polynomial basis functions, a numpy array of shape [N, degree+1]
    """
    def pbf(x: np.ndarray):
        x = np.asarray(x)
        return np.column_stack([(x)**k for k in range(degree + 1)])
    return pbf


def fourier_basis_functions(num_freqs: int) -> Callable:
    """
    Create a function that calculates the fourier basis functions up to a certain frequency
    :param num_freqs: the number of frequencies to use
    :return: a function that receives as input an array of values X of length N and returns the design matrix of the
             Fourier basis functions, a numpy array of shape [N, 2*num_freqs + 1]
    """
    def fbf(x: np.ndarray):
        x = np.asarray(x)
        cos_ = np.column_stack([np.cos(2*np.pi/24 * k * x) for k in range(num_freqs + 1)])
        sin_ = np.column_stack([np.sin(2*np.pi/24 * k * x) for k in range(1, num_freqs + 1)])
        return np.concatenate((cos_, sin_), axis=1)
    return fbf


def spline_basis_functions(knots: np.ndarray) -> Callable:
    """
    Create a function that calculates the cubic regression spline basis functions around a set of knots
    :param knots: an array of knots that should be used by the spline
    :return: a function that receives as input an array of values X of length N and returns the design matrix of the
             cubic regression spline basis functions, a numpy array of shape [N, len(knots)+4]
    """
    def csbf(x: np.ndarray):
        x = np.asarray(x)
        no_knots = np.column_stack([x**k for k in range(4)])
        knots_matrix = np.column_stack([np.maximum(0, x - k)**3 for k in knots])
        csbf_matrix = np.concatenate((no_knots, knots_matrix), axis=1)
        return csbf_matrix
    return csbf


def learn_prior(hours: np.ndarray, temps: np.ndarray, basis_func: Callable) -> tuple:
    """
    Learn a Gaussian prior using historic data
    :param hours: an array of vectors to be used as the 'X' data
    :param temps: a matrix of average daily temperatures in November, as loaded from 'jerus_daytemps.npy', with shape
                  [# years, # hours]
    :param basis_func: a function that returns the design matrix of the basis functions to be used
    :return: the mean and covariance of the learned covariance - the mean is an array with length dim while the
             covariance is a matrix with shape [dim, dim], where dim is the number of basis functions used
    """
    thetas = []
    # iterate over all past years
    for i, t in enumerate(temps):
        ln = LinearRegression(basis_func).fit(hours, t)
        thetas.append(ln.coefs_)

    thetas = np.array(thetas)

    # take mean over parameters learned each year for the mean of the prior
    mu = np.mean(thetas, axis=0)
    # calculate empirical covariance over parameters learned each year for the covariance of the prior
    cov = (thetas - mu[None, :]).T @ (thetas - mu[None, :]) / thetas.shape[0]
    return mu, cov


class BayesianLinearRegression:
    def __init__(self, theta_mean: np.ndarray, theta_cov: np.ndarray, sig: float, basis_functions: Callable):
        """
        Initializes a Bayesian linear regression model
        :param theta_mean:          the mean of the prior
        :param theta_cov:           the covariance of the prior
        :param sig:                 the signal noise to use when fitting the model
        :param basis_functions:     a function that receives data points as inputs and returns a design matrix
        """
        self.theta_mean = theta_mean
        self.theta_cov = theta_cov
        self.sig = sig
        self.basis_functions = basis_functions


    def fit(self, X: np.ndarray, y: np.ndarray) -> 'BayesianLinearRegression':
        """
        Find the model's posterior using the training data X
        :param X: the training data
        :param y: the true regression values for the samples X
        :return: the fitted model
        """
        H = self.basis_functions(X)
        self.sigma_posterior = np.linalg.inv(np.linalg.inv(self.theta_cov) + 1/self.sig**2 * H.T @ H)
        self.mean_posterior = self.sigma_posterior @ (H.T @ y * 1/self.sig**2 + np.linalg.inv(self.theta_cov) @ self.theta_mean)
        return self


    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts the regression values of X with the trained model using MMSE
        :param X: the samples to predict
        :return: the predictions for X
        """
        H = self.basis_functions(X)
        return H @ self.mean_posterior ## the mean is the MMSE estimator



    def fit_predict(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Find the model's posterior and return the predicted values for X using MMSE
        :param X: the training data
        :param y: the true regression values for the samples X
        :return: the predictions of the model for the samples X
        """
        self.fit(X, y)
        return self.predict(X)

    def predict_std(self, X: np.ndarray) -> np.ndarray:
        """
        Calculates the model's standard deviation around the mean prediction for the values of X
        :param X: the samples around which to calculate the standard deviation
        :return: a numpy array with the standard deviations (same shape as X)
        """
        H = self.basis_functions(X)
        var = H @ self.sigma_posterior @ H.T + self.sig**2 * np.eye(X.shape[0])
        return np.sqrt(np.diag(var))


    def posterior_sample(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts the regression values of X with the trained model and sampling from the posterior
        :param X: the samples to predict
        :return: the predictions for X
        """
        H = self.basis_functions(X)
        theta_samples = np.random.multivariate_normal(self.mean_posterior, self.sigma_posterior)
        return H @ theta_samples


class LinearRegression:

    def __init__(self, basis_functions: Callable):
        """
        Initializes a linear regression model
        :param basis_functions:     a function that receives data points as inputs and returns a design matrix
        """
        self.basis_functions = basis_functions

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LinearRegression':
        """
        Fit the model to the training data X
        :param X: the training data
        :param y: the true regression values for the samples X
        :return: the fitted model
        """
        H = self.basis_functions(X)
        self.coefs_ = np.linalg.pinv(H) @ y
        return self


    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts the regression values of X with the trained model
        :param X: the samples to predict
        :return: the predictions for X
        """
        H = self.basis_functions(X)
        return H @ self.coefs_

    def fit_predict(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Fit the model and return the predicted values for X
        :param X: the training data
        :param y: the true regression values for the samples X
        :return: the predictions of the model for the samples X
        """
        self.fit(X, y)
        return self.predict(X)

def plot_linear_estimations(test_hours, test, d, train_hours, train, nov16_hours, nov16):
    ln = LinearRegression(polynomial_basis_functions(d)).fit(train_hours, train)
    title = f"Linear Regression Predictions with d={d}\n MSE: {np.mean((test - ln.predict(test_hours))**2):.2f}"
    plt.figure()
    plt.plot(nov16_hours, nov16, 'k.', label='true values')
    plt.plot(test_hours, ln.predict(test_hours), 'r-', lw=2, label='test predictions')
    plt.plot(train_hours, ln.predict(train_hours), 'g-', lw=2, label='train predictions')
    plt.title(title)
    plt.xlabel('hour')
    plt.ylabel('temperature [C]')
    plt.legend()
    plt.show()

def plot_prior(x, title, mu_prior, cov_prior, basis_functions):
    H = basis_functions(x)
    prior_mean = H @ mu_prior
    prior_std = np.sqrt(np.diagonal(H @ cov_prior @ H.T))
    plt.figure()
    plt.fill_between(x, prior_mean - prior_std, prior_mean + prior_std, alpha=.5, label='prior confidence interval')
    for i in range(5):
        theta_sample = np.random.multivariate_normal(mu_prior, cov_prior)
        prior_sample = H @ theta_sample
        plt.plot(x, prior_sample)
    plt.plot(x, prior_mean, 'k', lw=2, label='prior mean')
    plt.title(title)
    plt.xlabel('hour')
    plt.ylabel('temperature [C]')
    plt.xlim(0, 24)
    plt.legend()
    plt.show()


def plot_bayesian_estimations(blr, nov16_hours, nov16, test_hours, test, train_hours, train, title, basis_functions):
    # Get predictions and standard deviations for all nov16_hours
    all_est = blr.predict(nov16_hours)
    all_std = blr.predict_std(nov16_hours)

    plt.figure()

    # Plot confidence interval for all hours
    plt.fill_between(nov16_hours, all_est - all_std, all_est + all_std, alpha=.5, label='confidence interval')

    # Plot sample estimations from posterior
    for i in range(5):
        theta_sample = np.random.multivariate_normal(blr.mean_posterior, blr.sigma_posterior)
        est_sample = basis_functions(nov16_hours) @ theta_sample
        plt.plot(nov16_hours, est_sample, alpha=0.7, lw=1.5)

    # Create masks for train and test indices
    train_mask = np.isin(nov16_hours, train_hours)
    test_mask = np.isin(nov16_hours, test_hours)

    # Plot true values in different colors for train and test
    plt.plot(nov16_hours[train_mask], nov16[train_mask], 'b.', label='train true values', markersize=8)
    plt.plot(nov16_hours[test_mask], nov16[test_mask], 'g.', label='test true values', markersize=8)

    # Plot predictions for all hours
    plt.plot(nov16_hours, all_est, 'r-', lw=2, label='predictions')

    plt.title(title)
    plt.xlabel('hour')
    plt.ylabel('temperature [C]')
    plt.legend()
    plt.show()


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument( "-l", "--linear", action="store_true", help="run linear regression part")
    parser.add_argument( "-p", "--polynomial", action="store_true", help="run Bayesian regression part")
    parser.add_argument( "-P", "--prior", action="store_true", help="run prior estimation part")
    parser.add_argument( "-f", "--fourier", action="store_true", help="run Fourier basis part")
    parser.add_argument( "-c", "--cubic", action="store_true", help="run Cubic basis part")

    args = parser.parse_args()

    # If no flags - run everything
    if not (args.linear or args.polynomial or args.prior or args.fourier or args.cubic):
        args.all_parts = True
    else:
        args.all_parts = False

    return args


def main():
    # get the args from the command line
    args = parse_args()

    # load the data for November 16 2024
    nov16 = np.load('nov162024.npy')
    nov16_hours = np.arange(0, 24, .5)

    train = nov16[:len(nov16)//2]
    train_hours = nov16_hours[:len(nov16)//2]
    test = nov16[len(nov16)//2:]
    test_hours = nov16_hours[len(nov16)//2:]

    # setup the model parameters
    degrees = [3, 7]

    # ----------------------------------------- Classical Linear Regression
    if args.linear or args.all_parts:
        for d in degrees:
            plot_linear_estimations(test_hours, test, d, train_hours, train, nov16_hours, nov16)

    # ----------------------------------------- Bayesian Linear Regression

    # load the historic data
    temps = np.load('jerus_daytemps.npy').astype(np.float64)
    hours = np.array([2, 5, 8, 11, 14, 17, 20, 23]).astype(np.float64)
    x = np.arange(0, 24, .1)

    # setup the model parameters
    sigma = 0.5
    degrees = [3, 7]  # polynomial basis functions degrees

    # ---------------------- polynomial basis functions
    if (args.prior and args.polynomial) or args.all_parts:
        for deg in degrees:
            pbf = polynomial_basis_functions(deg)
            mu, cov = learn_prior(hours, temps, pbf)
            plot_prior(x, f'Prior with Polynomial Basis Functions of Degree {deg}', mu, cov, pbf)

    if args.polynomial or args.all_parts:
        for deg in degrees:
            pbf = polynomial_basis_functions(deg)
            mu, cov = learn_prior(hours, temps, pbf)
            blr = BayesianLinearRegression(mu, cov, sigma, pbf).fit(train_hours, train)
            test_est = blr.predict(test_hours)
            test_std = blr.predict_std(test_hours)
            title = f'Bayesian Polynomial Linear Regression Predictions with d={deg}\n MSE: {np.mean((test - test_est)**2):.2f}'
            plot_bayesian_estimations(blr, nov16_hours, nov16, test_hours, test, train_hours, train, title, pbf)


    # ---------------------- Fourier basis functions
    freqs = [1,2,3]
    if (args.prior and args.fourier) or args.all_parts:
        for ind, K in enumerate(freqs):
            rbf = fourier_basis_functions(K)
            mu, cov = learn_prior(hours, temps, rbf)
            title = f'Prior with Fourier Basis Functions with frequency {K}'
            plot_prior(x, title, mu, cov, rbf)

    if args.fourier or args.all_parts:
        for ind, K in enumerate(freqs):
            rbf = fourier_basis_functions(K)
            mu, cov = learn_prior(hours, temps, rbf)
            blr = BayesianLinearRegression(mu, cov, sigma, rbf).fit(train_hours, train)
            test_est = blr.predict(test_hours)
            test_std = blr.predict_std(test_hours)
            title = f'Bayesian Fourier Linear Regression Predictions with K={K}\n MSE: {np.mean((test - test_est)**2):.2f}'
            plot_bayesian_estimations(blr, nov16_hours, nov16, test_hours, test, train_hours, train, title, rbf)


    # ---------------------- cubic regression splines
    knots = [np.array([12]), np.array([8, 16]), np.array([6, 12, 18])]
    if (args.prior and args.cubic) or args.all_parts:
        for ind, k in enumerate(knots):
            spline = spline_basis_functions(k)
            mu, cov = learn_prior(hours, temps, spline)
            title = f'Prior with Cubic Spline Basis Functions with knots at {k}'
            plot_prior(x, title, mu, cov, spline)

    if args.cubic or args.all_parts:
        for ind, k in enumerate(knots):
            spline = spline_basis_functions(k)
            mu, cov = learn_prior(hours, temps, spline)
            blr = BayesianLinearRegression(mu, cov, sigma, spline).fit(train_hours, train)
            test_est = blr.predict(test_hours)
            test_std = blr.predict_std(test_hours)
            title = f'Bayesian Cubic Spline Linear Regression Predictions with knots at {k}\n MSE: {np.mean((test - test_est)**2):.2f}'
            plot_bayesian_estimations(blr, nov16_hours, nov16, test_hours, test, train_hours, train, title, spline)


if __name__ == '__main__':
    main()
