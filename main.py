import numpy as np
import matplotlib.pyplot as pyplot
from scipy.stats import norm
from scipy.special import factorial

"""
This code implements an exact simulation of Merton's jump-diffusion model at fixed-dates,
as presented in Paul Glasserman's "Monte Carlo Methods in Financial Engineering" (2003,
Chapter 3.5).
The model accounts for both continuous diffusion and discrete jumps in asset prices.

References:
Glasserman, Paul. "Monte Carlo Methods in Financial Engineering," 2003, Springer-Verlag,
Chapter 3.5, "Processes with Jumps," pages 134-142.
"""


class MertonJumpDiffusionModel:
    def __init__(self, s0, r, mu, lambda_, sigma, a, b, t, steps) -> None:
        self.s0 = s0  # initial stock price
        self.r = r  # risk-free rate
        self.mu = mu  # drift
        self.lambda_ = lambda_  # poisson jump intensity
        self.sigma = sigma  # volatility
        self.a = a  # mean jump value
        self.b = b  # standard deviation of jump
        self.t = t  # time horizon
        self.steps = steps  # time steps
        self.dt = t / steps  # delta t, or t_k+1 - t_k

    def simulate_terminal_prices(self, num_sims):
        """
        Simulating terminal prices as described by Glasserman in Chapter 3.5
        "Processes with Jumps"

        The steps:
        1. Generate Z ~ N(0,1)
        2. Generate N ~ Poisson(lambda(t_{i+1} - t_{i}));
           if N = 0, set M = 0 go to (4).
        3. Generate log Y_1, ... , log Y_N and set M = sum_{i=1}^{N} log Y_i
        4. Set X(t_{i+1}) = X({t_i}) + drift term + diffusion term
        """

        Z = np.random.normal(
            0, 1, size=num_sims
        )  # standard random normal variable
        W_T = Z * np.sqrt(self.t)  # brownian motion
        N = np.random.poisson(
            self.lambda_ * self.t, size=num_sims
        )  # random poisson variable

        sum_Y = np.zeros(num_sims)
        positive_N_indicies = (
            N > 0
        )  # no need to check for N < 0, since it won't happen
        N_positive = N[positive_N_indicies]
        sum_Y[positive_N_indicies] = np.random.normal(
            loc=self.a * N_positive, scale=np.sqrt(N_positive) * self.b
        )  # standard deviation for the sum of N log-normal variables

        risk_neutral_drift = (
            self.r
            - 0.5 * self.sigma**2
            - self.lambda_ * (np.exp(self.a + 0.5 * self.b**2) - 1)
        )
        drift = risk_neutral_drift * self.t
        diffusion = self.sigma * W_T + sum_Y
        ln_St = np.log(self.s0) + drift + diffusion
        S_t = np.exp(ln_St)

        return S_t

    def black_scholes_price(self, S, K, T, r, sigma, option_type="call"):
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        if option_type == "call":
            return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        else:  # put option
            return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

    def closed_form_price(self, strike, option_type="call", max_jumps=50):
        """
        closed-form solution of a European option
        under Merton's Jump Diffusion Model.

        formula adapted from:
        https://quant.stackexchange.com/questions/33560/formula-for-merton-jump-diffusion-call-price
        """

        price = 0.0

        for k in range(max_jumps + 1):
            poisson_weight = (
                np.exp(-self.lambda_ * self.t)
                * (self.lambda_ * self.t) ** k
                / factorial(k)
            )
            r_k = (
                self.r
                - self.lambda_ * self.a
                + (k * np.log(1 + self.a)) / self.t
            )
            sigma_k = np.sqrt(self.sigma**2 + k * self.b**2 / self.t)
            bs_price = self.black_scholes_price(
                self.s0, strike, self.t, r_k, sigma_k, option_type=option_type
            )
            price += poisson_weight * bs_price

        return price

    def plot_option_estimate(
        self, model, N_values, M, strike_price, option_type="call"
    ):
        """
        For a given number of simulation paths, N,
        we repeat the valuation of the option M times.
        These M price estimates each are an average of
        N simulated option payoffs.
        """

        option_prices = []
        confidence_intervals = []
        relative_errors = []

        if option_type == "call":
            closed_form_price = model.closed_form_price(
                strike_price, option_type="call"
            )
        else:
            closed_form_price = model.closed_form_price(
                strike_price, option_type="put"
            )

        print(
            f"Closed-form price for a European {option_type} option: {closed_form_price}"
        )

        for N in N_values:
            price_estimates = np.zeros(M)

            # repeats the valuation of the option M times
            # using N simulation paths
            for m in range(M):
                terminal_prices = model.simulate_terminal_prices(N)
                if option_type == "call":
                    payoffs = np.maximum(terminal_prices - strike_price, 0)
                else:
                    payoffs = np.maximum(strike_price - terminal_prices, 0)
                price_estimates[m] = np.exp(-model.r * model.t) * np.mean(
                    payoffs
                )

            avg_price = np.mean(price_estimates)
            std_error = np.std(price_estimates, ddof=1) / np.sqrt(
                M
            )  # ddof = 1, default for R

            relative_error = (avg_price - closed_form_price) / closed_form_price

            print(
                f"N = {N:>6}: "
                f"Avg Price = {avg_price:.4f}, "
                f"Std Error = {std_error:.6f}, "
                f"Relative Error = {relative_error:.6f}"
            )

            ci_lower = avg_price - 1.96 * std_error
            ci_upper = avg_price + 1.96 * std_error

            option_prices.append(avg_price)
            confidence_intervals.append((ci_lower, ci_upper))
            relative_errors.append(relative_error)

        # plot the estimates
        fig, (option_estimate_plot, relative_error_plot) = pyplot.subplots(
            2, 1, figsize=(12, 16)
        )

        confidence_intervals = np.array(confidence_intervals)
        option_prices = np.array(option_prices)

        option_estimate_plot.plot(
            N_values, option_prices, label="Average option price", color="blue"
        )
        option_estimate_plot.fill_between(
            N_values,
            confidence_intervals[:, 0],
            confidence_intervals[:, 1],
            color="lightblue",
            alpha=0.45,
            label="95% CI",
        )
        option_estimate_plot.axhline(
            y=closed_form_price,
            color="r",
            linestyle="--",
            label="Closed-form price",
        )

        relative_error_plot.plot(
            N_values,
            relative_errors,
            label="Relative error",
            color="green",
            marker="o",
        )
        relative_error_plot.axhline(
            y=0, color="r", linestyle="--", label="Zero error"
        )

        param_text = (
            "Model Parameters:\n"
            f"S0 = {model.s0:.2f}\n"
            f"K = {strike_price:.2f}\n"
            f"r = {model.r:.2%}\n"
            f"μ = {model.mu:.2%}\n"
            f"σ = {model.sigma:.2f}\n"
            f"λ = {model.lambda_:.2f}\n"
            f"mu_j = {model.a:.2f}\n"
            f"sigma_j = {model.b:.2f}\n"
            f"T = {model.t:.1f} year(s)\n"
            f"Option type: {option_type}\n"
            f"Closed-form price: {closed_form_price:.4f}\n"
            f"M (fixed): {M}"
        )

        bbox_props = dict(
            boxstyle="round,pad=0.5",
            facecolor="white",
            alpha=0.8,
            edgecolor="gray",
        )

        option_estimate_plot.text(
            0.02,
            0.98,
            param_text,
            transform=option_estimate_plot.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=bbox_props,
            family="monospace",
        )

        for plot in (option_estimate_plot, relative_error_plot):
            plot.set_xscale("log")
            plot.set_xlabel("Number of simulated paths (N)")
            plot.grid(True)
            plot.legend()

        option_estimate_plot.set_ylabel("Option price")
        option_estimate_plot.set_title(
            "Option price estimate with confidence_intervals"
        )

        relative_error_plot.set_ylabel("Relative error")
        relative_error_plot.set_title("Relative error vs closed-form solution")

        pyplot.tight_layout()

        pyplot.savefig("options_estimate.png", dpi=600, bbox_inches="tight")
        pyplot.close()


def main():
    # Model parameters
    s0 = 100  # initial stock price
    r = 0.05  # risk-free rate
    mu = 0.05  # drift
    lambda_ = 1  # poisson jump intensity
    sigma = 0.2  # volatility
    a = -0.05  # mean jump value
    b = 0.1  # standard deviation of jump
    t = 1  # time horizon
    steps = 252  # time steps (approximating trading days in a year)

    strike_price = 100
    option_type = "call"

    N_values = [10_000, 50_000, 100_000, 500_000]  # number of simulation paths
    M = 1000  # number of repetitions for each N

    print("Simulating using Merton's Jump Diffusion Model")

    model = MertonJumpDiffusionModel(s0, r, mu, lambda_, sigma, a, b, t, steps)

    # plot option estimates
    model.plot_option_estimate(model, N_values, M, strike_price, option_type)

    print("Simulation completed.")


if __name__ == "__main__":
    main()
