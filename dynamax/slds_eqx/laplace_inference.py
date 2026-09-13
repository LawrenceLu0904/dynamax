import jax
import jax.numpy as jnp
import jax.random as jr

from jax import jit, vmap, lax
from tensorflow_probability.substrates import jax as tfp

import equinox as eqx

tfd = tfp.distributions
tfb = tfp.bijectors
MVN = tfd.MultivariateNormalFullCovariance
MVNDiag = tfd.MultivariateNormalDiag

from dynamax import hidden_markov_model as hmm

import optax
import jax.scipy.optimize
from jax import hessian, value_and_grad, jacfwd, jacrev
from dynamax.linear_gaussian_ssm.info_inference import block_tridiag_mvn_expectations
from dynamax.slds_eqx.utils import block_tridiag_mvn_sample
from functools import partial



def laplace_approximation(log_prob,
                          initial_distribution,
                          dynamics_distribution,
                          emission_distribution,
                          initial_states,
                          emissions,
                          method="BFGS",
                          adam_learning_rate=1e-2,
                          num_iters=10):
    """
    Laplace approximation to the posterior distribution for state space models
    with continuous latent states.

    log_prob: states, emissions -> log prob (scalar)
    initial_distribution: initial state -> log prob (scalar)
    dynamics_distribution: time, curr_state, next_state -> log prob (scalar)
    emission_distribution: time, curr_state, curr_emission -> log prob (scalar)
    x0 (array, (num_timesteps, latent_dim)): Initial guess of state mode.
    data (array, (num_timesteps, obs_dim)): Observation data.
    method (str, optional): Optimization method to use. Choices are
        ["L-BFGS", "BFGS", "Adam"]. Defaults to "L-BFGS".
    learning_rate (float, optional): [description]. Defaults to 1e-3.
    num_iters (int, optional): Only used when optimization method is "Adam."
        Specifies the number of update iterations. Defaults to 50.

    """
    def _compute_laplace_mean(initial_states):
        """Find the mode of the log joint probability for the Laplace approximation.
        """

        scale = initial_states.size
        dim = initial_states.shape[-1]

        if method == "BFGS" or "L-BFGS":
            # scipy minimize expects x to be shape (n,) so we flatten / unflatten
            def _objective(x_flattened):
                x = x_flattened.reshape(-1, dim)
                return -1 * jnp.sum(log_prob(x, emissions)) / scale

            optimize_results = jax.scipy.optimize.minimize(
                _objective,
                initial_states.ravel(),
                method="bfgs" if method == "BFGS" else "l-bfgs-experimental-do-not-rely-on-this",
                options=dict(maxiter=num_iters))

            # NOTE: optimize_results.status ==> 3 ("zoom failed") although it seems to be finding a max?
            x_mode = optimize_results.x.reshape(-1, dim)  # reshape back to (T, D)

        elif method == "Adam":

            params = initial_states
            _objective = lambda x: -1 * jnp.sum(log_prob(x, emissions)) / scale
            optimizer = optax.adam(adam_learning_rate)
            opt_state = optimizer.init(params)

            @jit
            def step(params, opt_state):
                loss_value, grads = value_and_grad(_objective)()
                updates, opt_state = optimizer.update(grads, opt_state, params)
                params = optax.apply_updates(params, updates)
                return params, opt_state, loss_value

            # TODO: Replace with a scan
            for i in range(num_iters):
                params, opt_state, loss_value = step(params, opt_state)
            x_mode = params

        else:
            raise ValueError(f"method = {method} is not recognized. Should be one of ['Adam', 'BFGS']")

        return x_mode

    def _compute_laplace_precision_blocks(states):
        """Get the negative Hessian at the given states for the Laplace approximation.
        """
        # initial distribution
        J_init = -1 * hessian(initial_distribution)(states[0])

        # dynamics
        f = dynamics_distribution
        ts = jnp.arange(len(states))
        J_11 = -1 * vmap(hessian(f, argnums=1))(ts[:-1], states[:-1], states[1:])
        J_22 = -1 * vmap(hessian(f, argnums=2))(ts[:-1], states[:-1], states[1:])
        J_21 = -1 * vmap(jacfwd(jacrev(f, argnums=2), argnums=1))(ts[:-1], states[:-1], states[1:])

        # emissions
        f = emission_distribution
        J_obs = -1 * vmap(hessian(f, argnums=1))(ts, states, emissions)

        # debug only if this flag is set
        # if jax.config.jax_disable_jit:
        #     assert not np.any(np.isnan(J_init)), "nans in J_init"
        #     assert not np.any(np.isnan(J_11)), "nans in J_11"
        #     assert not np.any(np.isnan(J_22)), "nans in J_22"
        #     assert not np.any(np.isnan(J_21)), "nans in J_21"
        #     assert not np.any(np.isnan(J_obs)), "nans in J_obs"

        # combine into diagonal and lower diagonal blocks
        J_diag = J_obs
        J_diag = J_diag.at[0].add(J_init)
        J_diag = J_diag.at[:-1].add(J_11)
        J_diag = J_diag.at[1:].add(J_22)
        J_lower_diag = J_21
        return J_diag, J_lower_diag


    # Find the mean and precision of the Laplace approximation
    mu = _compute_laplace_mean(initial_states)

    # The precision is given by the negative hessian at the mode
    J_diag, J_lower_diag = _compute_laplace_precision_blocks(mu)

    # Compute the linear potential by multiplying a block tridiagonal matrix with a vector
    # We represent the block tridiag matrix with the (T, D, D) array of diagonal blocks
    # and the (T-1, D, D) array of lower diagonal blocks. The vector is represented
    # as a (T, D) array.
    f = vmap(jnp.matmul)
    h = f(J_diag, mu) # (T, D)
    h = h.at[1:].add(f(J_lower_diag, mu[:-1]))
    h = h.at[:-1].add(f(jnp.swapaxes(J_lower_diag, -1, -2), mu[1:]))

    log_normalizer, Ex, ExxT, ExxnT = block_tridiag_mvn_expectations(J_diag, J_lower_diag, h)

    # Returns log_normalizer, Ex, ExxT, ExxnT, and posterior params for sampling
    return log_normalizer, Ex, ExxT, ExxnT, J_diag, J_lower_diag, h


def fit_laplace_em(slds, key, emissions, initial_zs, initial_xs,
                    num_iters=100, n_discrete_samples=1, freeze_z = False, freeze_params = False,
                    project_fn = None, m_step_lr = 1e-3, m_step_iters = 10,
                    closed_form_dynamics = False):
    """
    closed_form_dynamics: if True, the dynamics A_k, b_k and Q_k are replaced by
    their exact maximiser after every Adam M-step -- weighted least squares from
    the E-step's second moments, see _update_dynamics_closed_form. C, d, R and P
    keep the Adam update. False keeps the original all-Adam M-step.

    m_step_lr / m_step_iters control the Adam M-step. They matter more than they
    look: Adam moves each parameter by at most about the learning rate per step,
    so a parameter can travel no further than roughly
        num_iters * m_step_iters * m_step_lr
    from its initialisation over the whole fit. With the old defaults
    (10 EM iters x 10 Adam steps x 1e-3) that ceiling is 0.1, which silently
    caps every parameter. If a fitted value sits exactly at that bound, the fit
    ran out of optimiser rather than converging.

    Estimate the parameters of the SLDS and an approximate posterior distr.
    over latent states using Laplace EM. Specifically, the approximate
    posterior factors over discrete and continuous latent states. The
    discrete state posterior is a discrete chain graph, and the continuous
    posterior is a linear Gaussian chain. We estimate the continuous posterior
    using a Laplace approximation, which is appropriate when the likelihood
    is log concave in the continuous states.
    """
    K = slds.num_states
    D = slds.latent_dim
    N = slds.emission_dim
    ys = emissions
    B, T, _ = ys.shape

    def _update_discrete_states(slds, key, J_diag, J_lower_diag, h):
        """
        Update the discrete states to the coordinate-wise maximum using the
        Viterbi algorithm.
        """
        # sample xs from q(x)
        key, *skeys = jr.split(key, n_discrete_samples+1)
        vmap_block_tridiag_mvn_sample = vmap(block_tridiag_mvn_sample, in_axes=(0, None, None, None))
        x_samples = vmap_block_tridiag_mvn_sample(jnp.array(skeys), J_diag, J_lower_diag, h)

        pi0 = jnp.mean(jnp.array(
            [slds.pi0
                for x in x_samples]), axis=0)

        # TODO: eventually, transition matrix will depend on x
        P = jnp.mean(jnp.array(
            [slds.transition_matrix
                for x in x_samples]), axis=0)

        def _dynamics_likelihood(xs):
            f0 = lambda z: slds.init_continuous_state_distn(z).log_prob(xs[0])
            f = lambda z: vmap(lambda zn, x, xn: slds.dynamics_distn(zn, x).log_prob(xn), in_axes=(None, 0, 0))(z, xs[:-1], xs[1:]) # T-1
            return jnp.vstack([
                vmap(f0)(jnp.arange(K)),
                vmap(f)(jnp.arange(K)).T   # (T-1, K)
            ])

        log_likes = jnp.mean(vmap(_dynamics_likelihood)(x_samples), axis=0)

        return hmm.inference.hmm_smoother(pi0, P, log_likes), x_samples[0]

    def _update_continuous_states(slds, ys, zs, xs):

        # Define log prob functions that close over zs
        log_prob = lambda xs, ys: slds.log_prob(ys, zs, xs)

        # TODO : change these to slds object distributions
        # TODO : marginalize over q(z)
        initial_distribution = lambda x0: slds.init_continuous_state_distn(zs[0]).log_prob(x0)
        dynamics_distribution = lambda t, xt, xtp1: slds.dynamics_distn(zs[t+1], xt).log_prob(xtp1)
        emission_distribution = lambda t, xt, yt: slds.emission_distn(xt).log_prob(yt)
        log_normalizer, Ex, ExxT, ExxnT, J_diag, J_lower_diag, h = \
            laplace_approximation(log_prob,
                                initial_distribution,
                                dynamics_distribution,
                                emission_distribution,
                                jnp.zeros_like(xs),
                                ys,
                                method="L-BFGS",
                                num_iters=100)

        return Ex, ExxT, ExxnT, J_diag, J_lower_diag, h

    def _update_dynamics_closed_form(slds, zs, Ex, ExxT, ExxnT, ridge=1e-4):
        r"""
        Exact M-step for the dynamics A_k, b_k and Q_k, as ssm does it.

        Each state's dynamics x_{t+1} = A_k x_t + b_k + noise is a weighted least
        squares problem over the transitions that state governs. The state that
        governs x_t -> x_{t+1} is z_{t+1}, as in models.py::log_prob. With
        u_t = [x_t; 1] and sums over those transitions:

            [A_k  b_k] = ( sum_t E[x_{t+1} u_t^T] ) ( sum_t E[u_t u_t^T] )^-1
            Q_k        = diag( sum_t E[x_{t+1} x_{t+1}^T]
                               - [A_k  b_k] ( sum_t E[x_{t+1} u_t^T] )^T ) / N_k

        The expectations use the Laplace posterior's second moments rather than
        the outer product of its mean: x is inferred, E[x x^T] = x x^T + Cov, and
        leaving out Cov biases A_k.

        ExxnT[t] is E[x_{t+1} x_t^T]. The docstring of
        block_tridiag_mvn_expectations describes its transpose; this was checked
        against the exact covariance of a small block-tridiagonal system, and
        using the transpose makes the fit diverge.
        """
        K = slds.num_states
        D = slds.latent_dim

        # weight[k, b, t] = 1 where state k governs transition t of sequence b
        weight_list = []
        for k in range(K):
            weight_list.append((zs[:, 1:] == k).astype(jnp.float32))
        weight = jnp.stack(weight_list)

        # per state, summed over its transitions
        S_xx_prev = jnp.einsum("kbt,btij->kij", weight, ExxT[:, :-1, :, :])  # E[x_t x_t^T]
        S_xx_next = jnp.einsum("kbt,btij->kij", weight, ExxT[:, 1:, :, :])   # E[x_{t+1} x_{t+1}^T]
        S_next_prev = jnp.einsum("kbt,btij->kij", weight, ExxnT)             # E[x_{t+1} x_t^T]
        S_x_prev = jnp.einsum("kbt,bti->ki", weight, Ex[:, :-1, :])          # E[x_t]
        S_x_next = jnp.einsum("kbt,bti->ki", weight, Ex[:, 1:, :])           # E[x_{t+1}]
        N = jnp.einsum("kbt->k", weight)                                     # number of transitions

        def _solve_one_state(s_xx_prev, s_xx_next, s_next_prev, s_x_prev, s_x_next, n):
            """A_k, b_k and Q_k of one state, from its summed moments"""
            # sum_t E[u_t u_t^T], with u_t = [x_t; 1]
            uu = jnp.block([[s_xx_prev, s_x_prev[:, None]],
                            [s_x_prev[None, :], n[None, None]]])
            uu = uu + ridge * jnp.eye(D + 1)
            # sum_t E[x_{t+1} u_t^T]
            xu = jnp.concatenate([s_next_prev, s_x_next[:, None]], axis=1)
            Ab = jnp.linalg.solve(uu.T, xu.T).T        # xu @ inverse(uu)
            A_k = Ab[:, :D]
            b_k = Ab[:, D]
            residual = s_xx_next - Ab @ xu.T
            # floored well above the Softplus offset (1e-5), where its inverse diverges
            q_k = jnp.clip(jnp.diag(residual) / jnp.maximum(n, 1.0), 1e-4, None)
            return A_k, b_k, q_k

        As, bs, qs = vmap(_solve_one_state)(S_xx_prev, S_xx_next, S_next_prev,
                                            S_x_prev, S_x_next, N)

        # a state with too few transitions to solve for keeps its dynamics
        has_data = (N > D + 1)[:, None, None]
        As = jnp.where(has_data, As, slds.dynamics_matrices)
        bs = jnp.where(has_data[:, :, 0], bs, slds.dynamics_biases)
        qs_before = tfb.Softplus(low=1e-5).forward(slds.dynamics_diag_logvars)
        qs = jnp.where(has_data[:, :, 0], qs, qs_before)
        log_qs = tfb.Softplus(low=1e-5).inverse(qs)

        def _dynamics_matrices(model):
            """where the A_k live in the model, for eqx.tree_at"""
            return model.dynamics_matrices

        def _dynamics_biases(model):
            """where the b_k live in the model, for eqx.tree_at"""
            return model.dynamics_biases

        def _dynamics_diag_logvars(model):
            """where the Q_k live in the model (unconstrained), for eqx.tree_at"""
            return model.dynamics_diag_logvars

        slds = eqx.tree_at(_dynamics_matrices, slds, As)
        slds = eqx.tree_at(_dynamics_biases, slds, bs)
        slds = eqx.tree_at(_dynamics_diag_logvars, slds, log_qs)
        return slds

    def _update_params(slds, ys, zs, xs, lr=1e-3, num_iters=10):

        def _objective(slds):
            return -1.0 * jnp.sum(vmap(slds.log_prob)(ys, zs, xs)) / ys.size

        optimizer = optax.adam(lr)
        opt_state = optimizer.init(slds)

        def _step(carry, args):
            slds, opt_state = carry
            grads = jax.grad(_objective)(slds)
            updates, opt_state = optimizer.update(grads, opt_state)
            slds = optax.apply_updates(slds, updates)
            return (slds, opt_state), None

        (slds, _), _ = lax.scan(_step, (slds, opt_state), None, length=num_iters)

        return slds

    def _step(carry, step_size):
        zs, xs, slds, key = carry
        # lp = vmap(slds.log_prob)(ys, zs, xs).sum()
        Ex, ExxT, ExxnT, J_diag, J_lower_diag, h = lax.map(lambda args : _update_continuous_states(slds, *args), (ys, zs, xs))
        xs = Ex # redefine xs as mean

        # For ECoG SLDS, Conditionally skip the z update:
        if not freeze_z:
            key, skey = jr.split(key)
            post, x_samples = vmap(partial(_update_discrete_states, slds))(jr.split(skey, B), J_diag, J_lower_diag, h)
            zs = jnp.argmax(post.smoothed_probs, axis=-1)

        # Conditionally skip the M-step(ECoG SLDS)
        if not freeze_params:
            slds = _update_params(slds, ys, zs, xs, lr=m_step_lr, num_iters=m_step_iters)
            # Adam has just moved every parameter. The dynamics are then set to
            # their exact maximiser, which replaces Adam's step on them only
            if closed_form_dynamics:
                slds = _update_dynamics_closed_form(slds, zs, Ex, ExxT, ExxnT)

        lp = vmap(slds.log_prob)(ys, zs, xs).sum()
        return (zs, xs, slds, key), lp

    # ECoG SLDS: a Python loop rather than lax.scan, so the constraint can be
    # applied between EM iterations and progress can be shown
    from tqdm.auto import trange

    carry = (initial_zs, initial_xs, slds, key)
    lps_list = []
    for i in trange(num_iters, desc="Laplace-EM"):
        carry, lp = _step(carry, 1.0)

        # a constrained M-step: maximize, then project back onto the constraint
        if project_fn is not None:
            zs_c, xs_c, slds_c, key_c = carry
            slds_c = project_fn(slds_c)
            carry = (zs_c, xs_c, slds_c, key_c)

        lps_list.append(lp)

    zs, xs, slds, key = carry
    lps = jnp.array(lps_list)
    return slds, lps, zs, xs, key
