import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

from support.lib_config import CustomLogger

logger = CustomLogger(command=False)

class ProbabilityDistributions:
    def __init__(self):
        pass

    # Standard Log normal Distribution
    def lognormal_pdf(self, x, mu, sigma):
        return stats.lognorm.pdf(x, sigma, scale=np.exp(mu))

    # Customized Log normal Distribution (with additional α parameter)
    def custom_lognormal_pdf(self, h, mu, sigma, alpha):
        if h <= mu:
            return 0  # Prevent log issues
        term1 = 1 / ((h - mu) * sigma * np.sqrt(2 * np.pi))
        term2 = np.exp(-((np.log((h - mu) / alpha)) ** 2) / (2 * sigma ** 2))
        return term1 * term2

    # Normal Distribution
    def normal_pdf(self, x, mean, std):
        return stats.norm.pdf(x, loc=mean, scale=std)

    # Cauchy Distribution
    def cauchy_pdf(self, x, x0, gamma):
        return stats.cauchy.pdf(x, loc=x0, scale=gamma)

    # Chi-Square Distribution
    def chi2_pdf(self, x, df):
        return stats.chi2.pdf(x, df)

    # Exponential Distribution
    def exponential_pdf(self, x, lambda_):
        return stats.expon.pdf(x, scale=1 / lambda_)

    # Exponential Power Distribution
    def exp_power_pdf(self, x, beta):
        return stats.exponpow.pdf(x, beta)

    # Gamma Distribution
    def gamma_pdf(self, x, shape, scale):
        return stats.gamma.pdf(x, shape, scale=scale)

    # Power Law Distribution
    def powerlaw_pdf(self, x, alpha):
        return stats.powerlaw.pdf(x, alpha)

    # Rayleigh Distribution
    def rayleigh_pdf(self, x, sigma):
        return stats.rayleigh.pdf(x, sigma)

    # Uniform Distribution
    def uniform_pdf(self, x, a, b):
        return stats.uniform.pdf(x, loc=a, scale=b - a)

    # Gaussian Mixture Model PDF
    def gmm_pdf(self, x, weights, mus, sigmas):
        if not (len(weights) == len(mus) == len(sigmas)):
            raise ValueError("weights, mus, and sigmas must have the same length")

        pdf_val = 0
        for w, mu, sigma in zip(weights, mus, sigmas):
            pdf_val += w * stats.norm.pdf(x, loc=mu, scale=sigma)
        return pdf_val

    # Wrapper function to get probability for any distribution
    def get_pdf(self, dist_name, x, **params):
        distributions = {
            "lognormal": self.lognormal_pdf,
            "custom_lognormal": self.custom_lognormal_pdf,
            "normal": self.normal_pdf,
            "cauchy": self.cauchy_pdf,
            "chi2": self.chi2_pdf,
            "exponential": self.exponential_pdf,
            "exp_power": self.exp_power_pdf,
            "gamma": self.gamma_pdf,
            "powerlaw": self.powerlaw_pdf,
            "rayleigh": self.rayleigh_pdf,
            "uniform": self.uniform_pdf,
            "gmm":self.gmm_pdf
        }

        if dist_name not in distributions:
            raise ValueError(f"Distribution '{dist_name}' not supported.")

        return distributions[dist_name](x, **params)

    # Standard Lognormal Distribution (Inverse Transform Sampling)
    def lognormal_inv_transform(self, u, mu, sigma):
        return np.exp(mu + sigma * stats.norm.ppf(u))

    # Customized Lognormal Distribution (Fixed Inverse Transform Sampling)
    def custom_lognormal_inv_transform(self, u, mu, sigma, alpha):
        return mu + alpha * np.exp(sigma * stats.norm.ppf(u))

    # Normal Distribution (Inverse Transform Sampling)
    def normal_inv_transform(self, u, mean, std):
        return stats.norm.ppf(u, loc=mean, scale=std)

    # Cauchy Distribution (Inverse Transform Sampling)
    def cauchy_inv_transform(self, u, x0, gamma):
        return x0 + gamma * np.tan(np.pi * (u - 0.5))

    # Chi-Square Distribution (Inverse Transform Sampling)
    def chi2_inv_transform(self, u, df):
        return stats.chi2.ppf(u, df)

    # Exponential Distribution (Inverse Transform Sampling)
    def exponential_inv_transform(self, u, lambda_):
        return -np.log(1 - u) / lambda_

    # Exponential Power Distribution (Inverse Transform Sampling)
    def exp_power_inv_transform(self, u, beta):
        return (-np.log(1 - u)) ** (1 / beta)

    # Gamma Distribution (Inverse Transform Sampling)
    def gamma_inv_transform(self, u, shape, scale):
        return stats.gamma.ppf(u, shape, scale=scale)

    # Power Law Distribution (Inverse Transform Sampling)
    def powerlaw_inv_transform(self, u, alpha):
        return (1 - u) ** (-1 / alpha)

    # Rayleigh Distribution (Inverse Transform Sampling)
    def rayleigh_inv_transform(self, u, sigma):
        return sigma * np.sqrt(-2 * np.log(1 - u))

    # Uniform Distribution (Inverse Transform Sampling)
    def uniform_inv_transform(self, u, a, b):
        return a + (b - a) * u

    # Gaussian Mixture Model Inverse Transform Sampling (numerical)
    def gmm_inv_transform(self, u, weights, mus, sigmas):
        # Generate a component index based on weights
        component = np.random.choice(len(weights), p=weights)
        # Sample from the selected Gaussian component
        return stats.norm.ppf(u, loc=mus[component], scale=sigmas[component])

    # Wrapper function to get inverse transform for any distribution
    def get_inv_transform(self, dist_name, u, **params):
        distributions = {
            "lognormal": self.lognormal_inv_transform,
            "custom_lognormal": self.custom_lognormal_inv_transform,
            "normal": self.normal_inv_transform,
            "cauchy": self.cauchy_inv_transform,
            "chi2": self.chi2_inv_transform,
            "exponential": self.exponential_inv_transform,
            "exp_power": self.exp_power_inv_transform,
            "gamma": self.gamma_inv_transform,
            "powerlaw": self.powerlaw_inv_transform,
            "rayleigh": self.rayleigh_inv_transform,
            "uniform": self.uniform_inv_transform,
            "gmm": self.gmm_inv_transform
        }

        if dist_name not in distributions:
            raise ValueError(f"Distribution '{dist_name}' not supported.")

        return distributions[dist_name](u, **params)

    # Function to generate a single random sample using Inverse Transform Sampling
    def monte_carlo_single_sample(self, dist_name, **params):
        u = np.random.rand()  # Generate a single uniform random variable
        output = self.get_inv_transform(dist_name, u, **params)
        logger.commandline(f'Random U: {u}-> output {output}')  # Debugging
        return output

    # Function to plot the probability density function (PDF)
    def plot_distribution(self, dist_name, x_range, **params):
        x_values = np.linspace(x_range[0], x_range[1], 1000)
        y_values = np.array([self.get_pdf(dist_name, x, **params) for x in x_values])

        plt.figure(figsize=(8, 5))
        plt.plot(x_values, y_values, label=f"{dist_name} PDF", color='red', linewidth=2)

        # Labels and title
        plt.xlabel('Value')
        plt.ylabel('Probability Density')
        plt.title(f'{dist_name.capitalize()} Distribution')
        plt.legend()
        plt.grid(True)

        # Show the plot
        plt.show()

    def plot_cdf(self, dist_name, x_range ,**params):
        y_values = np.linspace(0, 1, 1000)
        x_values = np.array([self.get_inv_transform(dist_name, y, **params) for y in y_values])

        plt.figure(figsize=(8, 5))
        plt.plot(x_values, y_values, label=f"{dist_name} CDF", color='blue', linewidth=2)

        # Labels and title
        plt.xlabel('Value')
        plt.xlim([x_range[0],x_range[1]])
        plt.ylabel('Cumulative Probability')
        plt.title(f'{dist_name.capitalize()} Cumulative Distribution')
        plt.legend()
        plt.grid(True)

        # Show the plot
        plt.show()

    def checkDistConfig(self, config):
        if len(config) != 3:
            raise ValueError(
                "The tuple must contain 3 elements: distribution name, range, and parameters dictionary.")
        dist_name, x_range, params = config
        # Checking First Element
        if not isinstance(dist_name, str):
            raise ValueError("The first element must be the distribution name (string).")
        # Checking Second Element
        if isinstance(x_range, tuple):
            if len(x_range) != 2:
                raise ValueError(f"If second element is a tuple, it must have exactly two values. Format: (0,24), not{x_range}")
        elif isinstance(x_range, float):
            if not (0 <= x_range <= 1):
                raise ValueError(f"If second element is a probability, it must be between 0 and 1. {x_range} is out of bound")
        else:
            raise ValueError("Second Element must be either a tuple of two values (0,24) or a float(probability) between 0 and 1.")


        if not isinstance(params, dict):
            raise ValueError("The third element must be a dictionary containing the distribution parameters.")

        return config


# Example Usage
if __name__ == "__main__":
    dist = ProbabilityDistributions()

    # # Parameters for Custom Lognormal
    # mu, sigma, alpha = 4.73, 0.59, 3.86
    #
    # config = ('custom_lognormal', (0, 24), {'mu': mu, 'sigma': sigma, 'alpha': alpha})
    # logger.commandline(dist.checkDistConfig(config))
    # dist.plot_distribution(config[0], config[1], **config[2])
    #
    # config = ('cauchy', (0, 24), {'x0': 16.91, 'gamma':  1.69})
    # logger.commandline(dist.checkDistConfig(config))
    # dist.plot_distribution(config[0], config[1], **config[2])
    #
    # config = ('custom_lognormal', 0.8, {'mu': mu, 'sigma': sigma, 'alpha': alpha})
    # logger.commandline(dist.checkDistConfig(config))
    # logger.commandline(f'time: {dist.get_inv_transform(config[0], config[1], **config[2])}')
    #
    # random_val = np.random.rand()
    # logger.commandline(random_val)
    # logger.commandline(f'time = {dist.get_inv_transform('custom_lognormal',0.9,mu =mu,sigma= sigma,alpha= alpha)}')

    # Define a 2-component GMM
    weights = [0.5614978104566135, 0.3695059743958918, 0.06899621514749468]
    mus = [676.9426706757755, 176.58624037744755,1202.1041509230677]
    sigmas = np.sqrt([17947.784823394413,3727.1153369595468,18131.64471874752])

    config = ('gmm', (0, 1440), {'weights': weights, 'mus': mus, 'sigmas': sigmas})

    time = dist.get_inv_transform(config[0], config[1], **config[2])
    logger.commandline(f'new plug times {time}')

    # logger.commandline(dist.checkDistConfig(config))
    # dist.plot_distribution(config[0], config[1], **config[2])
    u_values = np.random.rand(1000)
    samples = [dist.monte_carlo_single_sample(config[0],**config[2]) for u in u_values]
    plt.figure(figsize=(8, 5))
    plt.hist(samples, bins=50, density=True, alpha=0.6, color='skyblue')
    plt.title("Histogram of GMM Samples")
    plt.xlabel("Value")
    plt.ylabel("Density")
    plt.grid(True)
    plt.show()


    # num_samples = 1000
    # u_values = np.random.rand(num_samples)  # Generate uniform random values
    # samples = [dist.custom_lognormal_inv_transform(u, mu, sigma, alpha) for u in u_values]
    #
    # # Compute PDF values for generated samples
    # pdf_values = [dist.custom_lognormal_pdf(s, mu, sigma, alpha) for s in samples]
    #
    # # Plot histogram of generated samples
    # plt.figure(figsize=(8, 5))
    # plt.hist(samples, bins=50, density=True, alpha=0.6, color='b', label="Generated Samples")
    #
    # # Plot theoretical PDF for comparison
    # x_range = np.linspace(mu + 0.01, max(samples), 1000)
    # y_values = [dist.custom_lognormal_pdf(x, mu, sigma, alpha) for x in x_range]
    # plt.plot(x_range, y_values, label="Theoretical PDF", color='red', linewidth=2)
    #
    # # Labels and title
    # plt.xlabel('Value')
    # plt.ylabel('Probability Density')
    # plt.title('Custom Lognormal Distribution: Inverse Transform Test')
    # plt.legend()
    # plt.grid(True)
    #
    # # Show the plot
    # plt.show()




