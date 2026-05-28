import numpy as np
from scipy.special import gammaln, kve, digamma
from scipy.optimize import brentq

LOG_2PI = np.log(2.0 * np.pi)
LOG_PI = np.log(np.pi)

NPAR = {"norm": 2, "t": 2, "nig": 4}


def norm_logpdf(x, mu, sigma):
    """log N(x; mu, sigma^2)."""
    return -0.5 * LOG_2PI - np.log(sigma) - 0.5 * ((x - mu) / sigma) ** 2


def t_logpdf(x, mu, s, nu):
    r"""log density of the Student-t with location mu, scale s, dof nu.

    pdf is given by the following equation. See https://en.wikipedia.org/wiki/Student%27s_t-distribution specifically
    the section on non-standardized Student's t distribution.
    \[
        p(x) = \frac{\Gamma\left(\frac{\nu+1}{2}\right)}{\Gamma\left(\frac{\nu}{2}\right) s \sqrt{\pi \nu}} \left(1 + \frac{1}{\nu} \left(\frac{x - \mu}{s}\right)^2 \right)^{-\frac{\nu + 1}{2}}
    \]

    \[
    \log p(x) = \log \Gamma \left(\frac{\nu+1}{2}\right) - \log \Gamma \left(\frac{\nu}{2}\right) - \log s - \frac{1}{2}\log (\pi \nu) - \left(\frac{\nu +1}{2}\right) \log \left(1 + \frac{1}{\nu}\left(\frac{x-\mu}{s}\right)^2\right)
    \]
    """

    d2 = ((x - mu) / s) ** 2
    return (gammaln((nu + 1) / 2) - gammaln(nu / 2)
            - 0.5 * np.log(nu * np.pi) - np.log(s)
            - (nu + 1) / 2 * np.log1p(d2 / nu))


def nig_logpdf(x, mu, delta, alpha, beta):
    r"""
    log density of NIG(alpha, beta, mu, delta)

    \[
    g(x) = \frac{\alpha \delta K_1(\alpha r)}{\pi r} \exp(\delta \gamma + \beta (x-\mu))
    \]

    where
    \[
    \gamma = \sqrt{\alpha^2 - \beta^2}, \qquad  r = \sqrt{\delta^2 + (x-\mu)^2}
    \]

    \[
    \log g(x) = \log \alpha + \log \delta + \log K_1 (\alpha r) - \log(\pi r) + \delta \gamma + \beta(x-\mu)
    \]
    See https://en.wikipedia.org/wiki/Normal-inverse_Gaussian_distribution for the distribution along with
    some information about the parameters

    """
    gamma = np.sqrt(alpha ** 2 - beta ** 2)
    r = np.sqrt(delta ** 2 + (x - mu) ** 2)

    r"""
    See Scipy kve for more information, but for numerical precision we do
        kve(v, z) = kv(v, z) * exp(z) iff kv(v, z) = kve(v, z) * exp(-z)
         taking logs gives the equation bellow
    """

    logK1 = np.log(kve(1, alpha * r)) - alpha * r

    return (np.log(alpha) + np.log(delta) - LOG_PI - np.log(r)
            + delta * gamma + beta * (x - mu) + logK1)


def nig_moments(mu, delta, alpha, beta):
    r""" (mean, sd, skewness, excess kurtosis)
    See https://en.wikipedia.org/wiki/Normal-inverse_Gaussian_distribution for moments

    For $Y \sim \text{NIG}(\alpha, \beta, \mu, \delta)$:

    \[E[Y] = \mu + \frac{\delta \beta}{\gamma}\]

    \[\operatorname{Var}[Y] = \frac{\delta \alpha^2}{\gamma^3}\]

    \[E\left[ \left( \frac{Y - E[Y]}{\sqrt{\operatorname{Var}[Y]}} \right)^3 \right] = \frac{3 \beta}{\alpha \sqrt{\delta \gamma}}\]

    \[\operatorname{Kurt}[Y] - 3 = \frac{3 \left( 1 + 4 \frac{\beta^2}{\alpha^2} \right)}{\delta \gamma}\]
    """
    gamma = np.sqrt(alpha ** 2 - beta ** 2)
    mean = mu + delta * beta / gamma
    var = delta * alpha ** 2 / gamma ** 3
    skew = 3.0 * beta / (alpha * np.sqrt(delta * gamma))
    exkurt = 3.0 * (1.0 + 4.0 * beta ** 2 / alpha ** 2) / (delta * gamma)
    return mean, np.sqrt(var), skew, exkurt


def emission_moments(kind, p):
    r"""
    Implied (mean, sd, skewness, excess kurtosis) of a state.

    Gaussian: $(\mu,\ \sigma,\ 0,\ 0)$.

    Student-t: mean $\mu$, and

    \[
        \mathrm{sd} = s\sqrt{\tfrac{\nu}{\nu-2}}\ (\nu>2), \qquad
        \mathrm{skew} = 0, \qquad
        \mathrm{exkurt} = \tfrac{6}{\nu-4}\ (\nu>4).
    \]

    NIG: see `nig_moments`.
    """
    if kind == "norm":
        return p["mu"], p["sigma"], 0.0, 0.0
    if kind == "t":
        nu = p["nu"]
        sd = p["s"] * np.sqrt(nu / (nu - 2)) if nu > 2 else np.inf
        ek = 6.0 / (nu - 4) if nu > 4 else np.inf
        return p["mu"], sd, 0.0, ek
    return nig_moments(p["mu"], p["delta"], p["alpha"], p["beta"])



# M steps for the emissions
def gaussian_mstep(x, w):
    r"""
    
    M-step for a Student-t emission from 
        Bulla, Jan. (2009). "Hidden Markov models with t components. 
        Increased persistence and other aspects", p. 31.

    He uses $\gamma$ for the weight of observations seeing as the NIG distribution also has a $\gamma$ parameter we use $w$ instead.
    
    \[
        \mu = \frac{\sum_t w_t \cdot  x_t}{\sum_t w_t}, \qquad
        \sigma^2 = \frac{\sum_t w_t \cdot (x_t-\mu)^2}{\sum_t w_t}.
    \]
    """

    W = w.sum() + 1e-300 # W != 0 
    mu = (w * x).sum() / W
    var = (w * (x - mu) ** 2).sum() / W
    return {"mu": mu, "sigma": np.sqrt(max(var, 1e-12))}


def student_t_mstep(x, w, p):
    r"""
    M-step for a Student-t emission from 
        Bulla, J. (2009). "Hidden Markov models with t components.
        Increased persistence and other aspects", p. 31.

    Bulla gives the re-estimation formulae for one t-distributed state of
    an HMM; this is their univariate specialisation.

    \[
        u^{(k)}(t) = \frac{\nu^{(k)} + 1}{\nu^{(k)} + (x_t-\mu^{(k)})^2 / (s^{(k)})^2}.
    \]

    \[
        \mu^{(k+1)} = \frac{\sum_{t=1} w_t\cdot u^{(k)}(t)\cdot x_t}{\sum_{t=1} w_t\cdot u^{(k)}(t)},
        \qquad
         s^{2\,(k+1)} = \frac{\sum_{t=1} w_t\cdot u^{(k)}(t)\cdot (x_t-\mu^{(k+1)})^2}{\sum_{t=1} w_t}.
    \]

    $\nu^{(k+1)}$ is the unique root of the equation bellow

    \[
        -\psi\!\Big(\tfrac{\nu}{2}\Big)
        + \log\!\Big(\tfrac{\nu}{2}\Big) + 1
        + \frac{1}{\sum_{t} w_t}
          \sum_{t=1}^{T} w_t\Big(\log u^{(k)}(t) - u^{(k)}(t)\Big)
        + \psi\!\Big(\tfrac{\nu^{(k)}+1}{2}\Big)
        - \log\!\Big(\tfrac{\nu^{(k)}+1}{2}\Big)
        \;=\; 0.
    \]
    
    
    """
    # initial values
    mu, s, nu = p["mu"], p["s"], p["nu"]
    W = w.sum() + 1e-300

    # Mahalanobis distance 
    delta = ((x - mu) / s) ** 2
    
    # aux function
    u = (nu + 1.0) / (nu + delta)

    # final equation p. 31.
    wu = w * u
    mu_n = (wu * x).sum() / (wu.sum() + 1e-300)
    
    # first equation p. 32.
    # NOTE: univariate distribution so we get the equation bellow 
    s2 = (wu * (x - mu_n) ** 2).sum() / W


    # nu^(k+1): unique root the equation on p. 32.
    # as was discussed in Bulla 2009 we can fix the degrees of freedom
    # ********* uncomment to estimate \nu ***********
    # C = (w * (np.log(u) - u)).sum() / W
    # corr = digamma((nu + 1) / 2) - np.log((nu + 1) / 2)   # psi - log term at nu^(k), p=1
    # lhs = lambda v: -digamma(v / 2) + np.log(v / 2) + 1.0 + C + corr
    # try:
    #     nu_n = brentq(lhs, 2.001, 200.0)
    # except ValueError:
    #     nu_n = 2.001 if lhs(2.001) < 0 else 200.0
    # #***********************************************
    
    # NOTE:  remove this after uncommenting above to estimate nu_n
    nu_n = nu
    
    return {"mu": mu_n, "s": np.sqrt(max(s2, 1e-8)), "nu": nu_n}


def nig_mstep(x, w, p):
    r"""
    M-step for a NIG HMM emission

    Based on the EM algorithm of Dimitris Karlis (2002), "An EM type
    algorithm for maximum likelihood estimation of the normal-inverse
    Gaussian distribution"

    Conditional expectations of the Inverse-Gaussian
    sufficient statistics z_t and 1/z_t. 
    
    Let $r_t = \sqrt(\delta^2 + (x_t-\mu)^2)$ and $\omega_t = \alpha r_t$ then 
    \[
        s_t = E[z_t\mid x_t] = \frac{r_t}{\alpha}\,
              \frac{K_0(\omega_t)}{K_1(\omega_t)},
    \]
    \[
        \eta_t = E[z_t^{-1}\mid x_t] = \frac{\alpha}{r_t}\,
                 \frac{K_0(\omega_t)}{K_1(\omega_t)} + \frac{2}{r_t^2}.
    \]

    M-step (Karlis p. 5-6):
    \[
        \bar s = \frac{1}{\Gamma}\sum_t w_t s_t, \qquad
        \hat\Phi = \frac{\Gamma}{\sum_t w_t \eta_t - \Gamma/\bar s},
    \]
    
    \[
        \delta^\prime = \sqrt{\hat\Phi}, \qquad \zeta^\prime = \delta^\prime /\bar s,
    \]
    \[
        \beta^\prime = \frac{\sum_t w_t x_t \eta_t - \bar x \sum_t w_t \eta_t}
                            {\Gamma - \bar s \sum_t w_t \eta_t},
        \qquad
        \mu^\prime = \bar x - \beta^\prime \bar s,
    \]
    \[
        \alpha^\prime = \sqrt{{\beta^\prime}^2 + {\zeta^\prime}^2}.
    \]
    
    NOTE: that in the paper he uses $w_i$ vs we use instead $\eta_t$ due to $w_t$ being associated with weight.
    """
    mu, delta, alpha = p["mu"], p["delta"], p["alpha"]
    W = w.sum() + 1e-300
    x_bar = (w * x).sum() / W

    # from the paper's notation $\phi^{(k)}(x) = 1 + [(x - \mu^{(k)})/\delta^(k)]^2$ so
    # $r = \sqrt{\phi(x)}\delta = \sqrt{\delta^2 + (x-\mu )^2}$ the '^k' is omitted
    r = np.sqrt(delta ** 2 + (x - mu) ** 2)
    # from the paper's notation $\delta \alpha \phi = r \alpha = \omega $ our notation on the right
    omega = alpha * r

    # K_0(omega)/K_1(omega)
    ratio = kve(0, omega) / kve(1, omega)
    s = (r / alpha) * ratio
    # paper's notation also this follows from recursive equation for 
    # modified Basel function
    eta = (alpha / r) * ratio + 2.0 / r ** 2

    # save computation
    A = (w * s).sum()
    C = (w * eta).sum() 
         
    s_bar = A / W
    
    # bottom has W/s_bar factored out
    Phi = W / max(C - W / s_bar, 1e-12)
    
    delta_n = np.sqrt(Phi) 
    
    zeta = delta_n / s_bar 

    #W - \bar s \cdot C === n - \bar s \sum_{i=1}^n w_i in paper
    den = W - s_bar * C
    Sxe = (w * x * eta).sum()
    
    beta_n = (Sxe - x_bar * C) / (den if abs(den) > 1e-12 else 1e-12)
    
    mu_n = x_bar - beta_n * s_bar
    alpha_n = np.sqrt(beta_n ** 2 + zeta ** 2)
    
    return {"mu": mu_n, "delta": delta_n, "alpha": alpha_n, "beta": beta_n}


def emission_mstep(kind, x, w, p):
    if kind == "norm":
        return gaussian_mstep(x, w)
    if kind == "t":
        return student_t_mstep(x, w, p)
    return nig_mstep(x, w, p)


def stationary(P):
    r"""
    Stationary distribution of the Markov chain

    \[
        \pi = \pi \Pi, \qquad \sum_{j=1}^m \pi_j = 1.
    \]
    """
    m = len(P)
    A = np.vstack([P.T - np.eye(m), np.ones(m)])
    b = np.append(np.zeros(m), 1.0)
    return np.linalg.lstsq(A, b, rcond=None)[0]


def emission_logdens(x, kinds, params):
    r"""
    Yen-Chi Chen Lecture 9 University of Washington Course: STAT 516 
    src https://faculty.washington.edu/yenchic/24A_stat516/Lec9_HMM.pdf
    
    With $b_j(x_t)$ the emission density of state $j$

    \[
        (\log B)_{t,j} = \log b_j(x_t) = \log p(x_t \mid S_t = j),
    \]
    
    """
    T, m = len(x), len(kinds)
    log_B = np.empty((T, m))
    for j in range(m):
        p = params[j]
        if kinds[j] == "norm":
            log_B[:, j] = norm_logpdf(x, p["mu"], p["sigma"])
        elif kinds[j] == "t":
            log_B[:, j] = t_logpdf(x, p["mu"], p["s"], p["nu"])
        else:
            log_B[:, j] = nig_logpdf(x, p["mu"], p["delta"],
   p["alpha"], p["beta"])
    return log_B


def forward_backward(log_p, P, pi):
    r"""
    forward-backward recursions following 
    Yen-Chi Chen Lecture 9 University of Washington Course: STAT 516 
    src https://faculty.washington.edu/yenchic/24A_stat516/Lec9_HMM.pdf
    
    With $b_j(x_t)$ the emission density of state $j$

    \[
    \alpha_1(j) = \pi_j b_j(x_1), \qquad
    \alpha_t(j) = b_j(x_t)\sum_{i=1}^m \alpha_{t-1}(i)P_{ij}.
    \]

    The backward pass is

    \[
    \beta_T(i) = 1, \qquad
    \beta_t(i) = \sum_{j=1}^m P_{ij}b_j(x_{t+1})\beta_{t+1}(j).
    \]

    The notation used in Yen-Chi Chen's slides is a bit different, but this is exactly the equations on page 7.
    Wikipedia calls this "Temporary Variables"  https://en.wikipedia.org/wiki/Baum%E2%80%93Welch_algorithm.
    The \gamma is the same the \xi is exactly P(X_t= j , X_{t-1}=i \mid \bf y ; \theta^{(k)}) from page 7.

   \[
    \gamma_t(j) = \frac{\alpha_t(j)\beta_t(j)}
                       {\sum_k \alpha_t(k)\beta_t(k)},
    \qquad
    \xi_t(i,j) = \frac{\alpha_t(i)P_{ij}b_j(x_{t+1})\beta_{t+1}(j)}
                      {\sum_{k,\ell} \alpha_t(k)P_{k\ell}e_\ell(x_{t+1})\beta_{t+1}(\ell)}.
    \]

    and the observed-data log-likelihood is

    \[
    \log L = \log P(x_{1:T}) = \log \sum_{j=1}^m \alpha_T(j).
    \]
    """
    T, m = log_p.shape


    offset = log_p.max(axis=1)
    B = np.exp(log_p - offset[:, None])

    # forward pass
    alpha = np.empty((T, m))
    c = np.empty(T)
    #\alpha_1(j) = \pi_j b_j(x_1) 
    alpha[0] = pi * B[0]
    c[0] = alpha[0].sum() #reweight for stability
    alpha[0] /= c[0]
    #  \alpha_t(j) = b_j(x_t)\sum_{i=1}^m \alpha_{t-1}(i)P_{ij}
    for t in range(1, T):
        alpha[t] = (alpha[t - 1] @ P) * B[t]
        c[t] = alpha[t].sum()
        alpha[t] /= c[t]

    # backward pass
    beta = np.empty((T, m))
    #\beta_T(i) = 1
    beta[-1] = 1.0
    #\beta_t(i) = \sum_{j=1}^m P_{ij}b_j(x_{t+1})\beta_{t+1}(j)
    for t in range(T - 2, -1, -1):
        beta[t] = (P @ (B[t + 1] * beta[t + 1])) / c[t + 1]

    # "Temporary Variables"
    gamma = alpha * beta
    # \gamma_t(j) = \frac{\alpha_t(j)\beta_t(j)}{\sum_k \alpha_t(k)\beta_t(k)}
    gamma /= gamma.sum(axis=1, keepdims=True)

    xi = np.zeros((m, m))
    
    #    \xi_t(i,j) = \frac{\alpha_t(i)P_{ij}b_j(x_{t+1})\beta_{t+1}(j)}
    #                  {\sum_{k,\ell} \alpha_t(k)P_{k\ell}e_\ell(x_{t+1})\beta_{t+1}(\ell)}
    for t in range(T - 1):
        xi += (alpha[t][:, None] * P) * (B[t + 1] * beta[t + 1])[None, :] / c[t + 1]

    loglik = np.log(c).sum() + offset.sum()
    return gamma, xi, loglik

def init_params(x, m, kinds, rng):
    a = np.abs(x - np.median(x))
    qs = np.quantile(a, np.linspace(0, 1, m + 1))
    qs[0], qs[-1] = -np.inf, np.inf
    lab = np.clip(np.digitize(a, qs[1:-1]), 0, m - 1)
    order = np.argsort([x[lab == k].std() for k in range(m)])

    params = []
    for k in range(m):
        xk = x[lab == order[k]]
        mu = xk.mean() + rng.normal(0, 0.02)
        sd = max(xk.std(), 0.05) * np.exp(rng.normal(0, 0.1))
        if kinds[k] == "norm":
            params.append({"mu": mu, "sigma": sd})
        elif kinds[k] == "t":
            params.append({"mu": mu, "s": sd * np.sqrt(6 / 8), "nu": 8.0})
        else:
            ek = 1.0 + abs(rng.normal(0, 0.3))
            da = 3.0 / ek
            delta = sd * np.sqrt(da)
            alpha = np.sqrt(da) / sd
            params.append({"mu": mu, "delta": delta, "alpha": alpha,
                           "beta": rng.normal(0, 0.05 * alpha)})

    P = np.full((m, m), 0.1 / (m - 1))
    np.fill_diagonal(P, 0.9)
    return P, params


def baum_welch(x, m, kinds, max_iter=400, tol=1e-7, n_init=6, seed=0):
    r"""
    Implementation of Baum-Welch algorithm 

    Each iteration runs `forward_backward` (E-step) then the M-step
    
    \[
        P_{ij} = \frac{\sum_{t=1}^{T-1}\xi_t(i,j)}
                      {\sum_{t=1}^{T-1}\sum_{k=1}^m \xi_t(i,k)},
        \qquad
        \pi_j = \gamma_1(j),
    \]
    NOTE: $\gamma_t(i) = \sum_{j} \xi_t(i,j)$ which is where we get the equation above 


    Every state's emission parameters refit by its weighted M-step
(`emission_mstep`, weights $\gamma_t(j)$). 

    Several random restarts are run and the highest-likelihood fit is kept
    states are then relabelled by implied standard deviation so state 1
    is low-volatility.  We also return BIC here the number of parameters is 
    k= m(m-1) + (m-1) =\sum_j dist_j 
    where m(m-1) is the matrix as this is a two state markov process 
    m(m-1)= 2(2-1)=2 
    
    and the initial distribution is m-1=2-1=1
    and lastly the number of parameters for **both** 
    states of the distribution
    
    """
    rng = np.random.default_rng(seed)
    best = None

    for _ in range(n_init):
        P, params = init_params(x, m, kinds, rng)
        pi = stationary(P)
        ll_old, ll, it = -np.inf, -np.inf, 0

        for it in range(1, max_iter + 1):
            log_P = emission_logdens(x, kinds, params)
            if not np.all(np.isfinite(log_P)):
                ll = -np.inf
                break

            # E step
            gamma, xi, ll = forward_backward(log_P, P, pi)
            if not np.isfinite(ll):
                break

            # M step
            P = xi / (xi.sum(axis=1, keepdims=True) + 1e-300)
            pi = gamma[0].copy()
            # weighted M for params
            # Use expected sufficient statistic weighted by https://faculty.washington.edu/yenchic/24A_stat516/Lec9_HMM.pdf 
            # p. 7.
            params = [emission_mstep(kinds[j], x, gamma[:, j], params[j]) for j in range(m)]

            if abs(ll - ll_old) < tol * (1.0 + abs(ll_old)):
                break
            ll_old = ll

        if np.isfinite(ll) and (best is None or ll > best["loglik"]):
            best = dict(loglik=ll, P=P.copy(), pi=pi.copy(),
                        params=[dict(p) for p in params],
                        kinds=list(kinds), m=m, n_iter=it)

    #relabel states by implied SD: state 1 = low-vol
    sd = [emission_moments(best["kinds"][j], best["params"][j])[1]
          for j in range(m)]
    order = np.argsort(sd)
    best["P"] = best["P"][np.ix_(order, order)]
    best["pi"] = best["pi"][order]
    best["params"] = [best["params"][j] for j in order]
    best["kinds"] = [best["kinds"][j] for j in order]

    
    n_par = m * (m - 1) + (m - 1) + sum(NPAR[k] for k in best["kinds"])
    best["n_par"] = n_par
    best["bic"] = -2.0 * best["loglik"] + n_par * np.log(len(x))
    
    return best


def decode(res, x):
    r"""
    assign each $t$ to the most likely state under its marginal posterior,

    \[
        \hat S_t = \arg\max_j \gamma_t(j).
    \]
    """
    log_B = emission_logdens(x, res["kinds"], res["params"])
    gamma, _, _ = forward_backward(log_B, res["P"], res["pi"])
    return gamma.argmax(axis=1), gamma


def sojourn(states, m):
    r"""
    Expected 'sojourn'
     Bulla, Jan. (2009). "Hidden Markov models with t components. 
        Increased persistence and other aspects", p. 31.

    """
    runs = {k: [] for k in range(m)}
    cur, length = states[0], 1
    for v in states[1:]:
        if v == cur:
            length += 1
        else:
            runs[cur].append(length)
            cur, length = v, 1
    runs[cur].append(length)
    return {k: (np.mean(v) if v else 0.0) for k, v in runs.items()}



def draw(kind, p, rng):
    r"""
    Sample distribution, used for the simulation in ACF.
    
    Using variance-mean mixture with inverse gaussian see
    https://en.wikipedia.org/wiki/Normal-inverse_Gaussian_distribution
    \[
        W \sim \text{IG}(\delta,\zeta), \qquad
        X = \mu + \beta W + \sqrt{W}\,Z, \quad Z \sim N(0,1),
    \]

    with $\zeta = \sqrt{\alpha^2-\beta^2}$.
    """
    if kind == "norm":
        return p["mu"] + p["sigma"] * rng.standard_normal()
    if kind == "t":
        return p["mu"] + p["s"] * rng.standard_t(p["nu"])
    zeta = np.sqrt(p["alpha"] ** 2 - p["beta"] ** 2)
    W = rng.wald(p["delta"] / zeta, p["delta"] ** 2)
    return p["mu"] + p["beta"] * W + np.sqrt(W) * rng.standard_normal()


def simulate(spec, T, seed=0):
    r"""
    Simulate an HMM path of length $T$:

    \[
        S_1 \sim \pi, \qquad S_{t+1}\mid S_t \sim P_{S_t,\cdot}, \qquad
        x_t \sim b_{S_t}.
    \]

    spec is a dict with keys m, kinds, P, params (optional pi).
    """
    rng = np.random.default_rng(seed)
    P = np.asarray(spec["P"], float)
    pi = spec.get("pi")
    
    if pi is None:
        pi = stationary(P)
    s = rng.choice(spec["m"], p=pi)
    out = np.empty(T)
    
    for t in range(T):
        out[t] = draw(spec["kinds"][s], spec["params"][s], rng)
        s = rng.choice(spec["m"], p=P[s])
    return out


def acf(x, lags):
    r"""
    Sample autocorrelation at the given lags:

    \[
        \rho_k = \frac{\sum_t (x_t-\bar x)(x_{t+k}-\bar x)}
                       {\sum_t (x_t-\bar x)^2}.
    \]
    """
    x = x - x.mean()
    v = (x * x).mean()
    return np.array([(x[:-k] * x[k:]).mean() / v for k in lags])
