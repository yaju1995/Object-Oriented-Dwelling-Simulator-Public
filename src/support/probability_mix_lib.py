from sklearn.mixture import GaussianMixture
import numpy as np


class ProbabilityDistributionMix:
    def __init__(self, dist_name: str, u: list, params: dict):
        self.dist_name = dist_name
        self.u = u
        self.params = params
        self.model = self.setup_model()

    def setup_model(self):
        distributions = {
            "gaussian_mix": self.gaussian_mix_model,
        }

        if self.dist_name not in distributions:
            raise ValueError(f"Distribution '{self.dist_name}' not supported.")

        return distributions[self.dist_name](**self.params)

    def gaussian_mix_model(self, n_components, means, covariances, weights):
        gmm = GaussianMixture(n_components=n_components, covariance_type="full")

        # Fake fitting step (needed to initialize internals)
        dummy_data = np.random.randn(100 * n_components, 1)
        gmm.fit(dummy_data)

        # Now override internal params
        gmm.weights_ = np.array(weights)
        gmm.means_ = np.array(means).reshape(n_components, 1)
        gmm.covariances_ = np.array(covariances).reshape(n_components, 1, 1)

        # Compute precision matrices from covariances
        gmm.precisions_cholesky_ = np.array([
            [[1. / np.sqrt(cov[0][0])]] for cov in gmm.covariances_
        ])

        return gmm

    def sample(self):
        return self.model.sample(len(self.u))[0].flatten()  # shape (n_samples, 1) → (n_samples,)

    def pdf(self, x):
        return np.exp(self.model.score_samples(np.array(x).reshape(-1, 1)))  # PDF via log-density


def setupGMMModel(n_components, means, covariances, weights):
    # Create and fit GaussianMixture model
    gmm_model = GaussianMixture(n_components=n_components)  # Precision = inverse covariance
    gmm_model.fit(np.random.randn(100, 1))  # Dummy data for initialization
    gmm_model.means_ = means
    gmm_model.covariances_ = covariances
    gmm_model.weights_ = weights

    return gmm_model
