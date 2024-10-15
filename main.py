import numpy as np
from scipy.stats import norm
from scipy.special import factorial
import matplotlib.pyplot as pyplot


class MertonJumpDiffusionModel:
    def __init__(self, s0, r, mu, lambda_, sigma, a, b, t, steps, strike):
        self.s0 = s0  # initial stock price
        self.r = r  # risk-free rate
        self.mu = mu  # drift
        self.lambda_ = lambda_  # poisson jump intensity
        self.sigma = sigma  # volatility
        self.a = a  # mean jump value
        self.b = b  # standard deviation of jump
        self.t = t  # time horizon
        self.steps = steps  # time steps
        self.strike = strike
        self.dt = t / steps  # delta t, or t_k+1 - t_k

        self.nu = b
        self.m = np.exp(a + 0.5 * b**2)

    def simulate_terminal_prices(self, num_simulations):
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
            0, 1, num_simulations
        )  # standard random normal variable
        W_T = Z * np.sqrt(self.t)  # brownian motion
        N = np.random.poisson(
            self.lambda_ * self.t, size=num_simulations
        )  # random poisson variable

        sum_Y = np.zeros(num_simulations)
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

    def nested_monte_carlo_option_price(
        self, option_type="call", M=100, N=1000
    ):
        price_estimates = np.zeros(M)

        for m in range(M):
            terminal_prices = self.simulate_terminal_prices(N)

            if option_type == "call":
                payoffs = np.maximum(terminal_prices - self.strike, 0)
            elif option_type == "put":
                payoffs = np.maximum(self.strike - terminal_prices, 0)
            else:
                raise ValueError("Option type must be 'call' or 'put'")

            price_estimates[m] = np.exp(-self.r * self.t) * np.mean(payoffs)

        return price_estimates

    def black_scholes_price(self, S, K, T, r, sigma, option_type="call"):
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        if option_type == "call":
            price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        elif option_type == "put":
            price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        else:
            raise ValueError("Option type must be 'call' or 'put'")

        return price

    def closed_form_price(self, option_type="call", n_terms=100):
        """
        Adapted from
        Robert C. Merton,
        Option pricing when underlying stock returns are discontinuous,
        Journal of Financial Economics,
        Volume 3, Issues 1–2,
        1976,
        Pages 125-144,
        """

        lambda_prime = self.lambda_ * self.m
        price = 0

        for n in range(n_terms):
            sigma_n = np.sqrt(self.sigma**2 + n * self.nu**2 / self.t)
            r_n = (
                self.r
                - self.lambda_ * (self.m - 1)
                + n * np.log(self.m) / self.t
            )

            bs_price = self.black_scholes_price(
                self.s0,
                self.strike,
                self.t,
                r_n,
                sigma_n,
                option_type=option_type,
            )

            term = (
                np.exp(-lambda_prime * self.t)
                * (lambda_prime * self.t) ** n
                / factorial(n)
                * bs_price
            )
            price += term

        return price


def generate_convergence_graphs(model, M, N_values):
    call_means = []
    call_stds = []
    put_means = []
    put_stds = []

    for n in N_values:
        call_prices = model.nested_monte_carlo_option_price(
            option_type="call", M=M, N=n
        )
        put_prices = model.nested_monte_carlo_option_price(
            option_type="put", M=M, N=n
        )

        call_means.append(np.mean(call_prices))
        call_stds.append(np.std(call_prices))
        put_means.append(np.mean(put_prices))
        put_stds.append(np.std(put_prices))

    closed_form_call = model.closed_form_price(option_type="call")
    closed_form_put = model.closed_form_price(option_type="put")

    fig, (ax1, ax2) = pyplot.subplots(1, 2, figsize=(15, 6))

    # fill between confidence intervals (commented out)
    # ax1.plot(N_values, call_means, 'o-')
    # ax1.fill_between(N_values, np.array(call_means) - np.array(call_stds), np.array(call_means) + np.array(call_stds), alpha=0.3)
    # ax2.plot(N_values, put_means, 'o-')
    # ax2.fill_between(N_values, np.array(put_means) - np.array(put_stds), np.array(put_means) + np.array(put_stds), alpha=0.3)

    ax1.errorbar(N_values, call_means, yerr=call_stds, fmt="o-", capsize=5)
    ax1.axhline(
        y=closed_form_call,
        color="r",
        linestyle="--",
        label="Closed-form solution",
    )
    ax1.set_xscale("log")
    ax1.set_xlabel("Number of simulations (N)")
    ax1.set_ylabel("Call Option Price")
    ax1.set_title("Convergence of Call Option Price")
    ax1.legend()

    ax2.errorbar(N_values, put_means, yerr=put_stds, fmt="o-", capsize=5)
    ax2.axhline(
        y=closed_form_put,
        color="r",
        linestyle="--",
        label="Closed-form solution",
    )
    ax2.set_xscale("log")
    ax2.set_xlabel("Number of simulations (N)")
    ax2.set_ylabel("Put Option Price")
    ax2.set_title("Convergence of Put Option Price")
    ax2.legend()

    params_text = (
        f"Parameters:\n"
        f"S0 = {model.s0}\n"
        f"K = {model.strike}\n"
        f"r = {model.r}\n"
        f"μ = {model.mu}\n"
        f"λ = {model.lambda_}\n"
        f"σ = {model.sigma}\n"
        f"a = {model.a}\n"
        f"b = {model.b}\n"
        f"T = {model.t}\n"
        f"M = {M}"
    )

    fig.text(
        0.4,
        0.30,
        params_text,
        verticalalignment="center",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    pyplot.tight_layout()
    pyplot.savefig("options-convergence.png", dpi=600)


def main():
    model = MertonJumpDiffusionModel(
        s0=100,
        r=0.03,
        mu=0.03,
        lambda_=1,
        sigma=0.4,
        a=0.05,
        b=0.1,
        t=1,
        steps=252,
        strike=100,
    )

    M = 100
    N = [
        500,
        1000,
        2000,
        4000,
        8000,
        16000,
        32000,
        64000,
        128000,
    ]  # Number of simulations for each estimate

    for n in N:
        call_prices = model.nested_monte_carlo_option_price(
            option_type="call", M=M, N=n
        )
        put_prices = model.nested_monte_carlo_option_price(
            option_type="put", M=M, N=n
        )

        print(
            f"European Call Option Price: Mean = {np.mean(call_prices):.4f}, Std = {np.std(call_prices):.4f}"
        )
        print(
            f"European Put Option Price: Mean = {np.mean(put_prices):.4f}, Std = {np.std(put_prices):.4f}"
        )

        call_price = model.closed_form_price(option_type="call")
        put_price = model.closed_form_price(option_type="put")

        print(f"European Call Option Price: {call_price:.4f}")
        print(f"European Put Option Price: {put_price:.4f}")

        print("---------------------")

        generate_convergence_graphs(model, M, N)


if __name__ == "__main__":
    main()
