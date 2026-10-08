"""
Pseudo_DFL Model Module
=======================
Network, smoothed repair decisions and Fenchel-Young (FY) loss for the DFL
repair policy.

Contents:
    RepairNetwork             : MLP mapping the state to repair values θ
    perturbed_repair          : smoothed decision  r_ε[k] = σ(θ[k]/ε) · mask[k]
    perturbed_repair_budgeted : Monte-Carlo perturb-and-MAP over the 0/1
                                knapsack (per-stage repair budget)
    FenchelYoungLoss          : loss with gradient  ∇_θ L = r_ε(θ) - r*
    hard_repair               : unperturbed threshold decision
    load_repair_network       : rebuild a trained network from a checkpoint

Without a repair budget the repair problem at each stage t separates over the
components:
    r(θ) = argmax_{r ∈ {0,1}^n, r[k]≤mask[k]} θ^T r
         = {1 if θ[k]>0 and mask[k]=1, else 0}  for each k independently.

With i.i.d. logistic noise of scale ε on θ (Gumbel-max over the two choices of
each component) the perturbed solution has a closed form:
    r_ε[k] = σ(θ[k]/ε) · mask[k]
With a budget the stage problem is a 0/1 knapsack (knapsack.py) and r_ε is
estimated from Gaussian perturbations.

FY pseudo-loss, with r* the target repairs of the training labels. r_ε is held
constant, so autograd never differentiates through it:
    L = (r_ε.detach() − r*) · θ
    ∇_θ L = r_ε − r*,  which is the FY gradient
"""

import numpy as np

try:
    import torch
    import torch.nn as nn
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


def _check_torch():
    if not _TORCH_AVAILABLE:
        raise ImportError(
            "PyTorch is required for Pseudo_DFL. Install with:\n"
            "  pip install torch"
        )


# ---------------------------------------------------------------------------
# RepairNetwork
# ---------------------------------------------------------------------------

class RepairNetwork(nn.Module):
    """
    MLP mapping state features → repair values θ.

    Input  x ∈ R^{input_dim}    = [a_gen, a_line, t/T]  (availabilities, stage fraction)
    Output θ ∈ R^{n_components} = [θ_gen, θ_line]

    θ[k] is read as the net value of repairing component k. Without a budget a
    failed component is repaired iff θ[k] > 0; with a budget the knapsack
    chooses among the failed components with θ[k] > 0.
    """

    def __init__(self, input_dim, n_components, hidden_dims=(64, 128, 64),
                 dropout=0.1):
        """
        Parameters
        ----------
        input_dim    : int   dimension of the state feature vector (n_gen + n_lines + 1)
        n_components : int   number of components (n_gen + n_lines)
        hidden_dims  : tuple hidden layer widths
        dropout      : float dropout probability after each hidden layer (0 = none)
        """
        _check_torch()
        super().__init__()

        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, n_components))   # output: θ (unbounded)

        self.net = nn.Sequential(*layers)

        # Xavier-uniform weights, zero biases
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        Parameters
        ----------
        x : Tensor, shape (..., input_dim)

        Returns
        -------
        theta : Tensor, shape (..., n_components)
        """
        return self.net(x)


# ---------------------------------------------------------------------------
# Perturbed repair operators (closed-form sigmoid; Monte-Carlo under a budget)
# ---------------------------------------------------------------------------

def perturbed_repair(theta, mask, epsilon=0.1):
    """
    Smoothed repair decision under logistic perturbation (no budget).

    For the separable binary repair problem, the expected solution under
    i.i.d. logistic noise with scale ε is
        r_ε[k] = σ(θ[k] / ε) · mask[k]
    where σ is the sigmoid and mask[k] = 1 iff component k is failed.

    Parameters
    ----------
    theta   : Tensor, shape (..., n_components), repair values
    mask    : Tensor, shape (..., n_components), 1 = failed, 0 = operational
    epsilon : float, perturbation scale (larger ε gives smoother decisions)

    Returns
    -------
    r_soft : Tensor, shape (..., n_components), values in [0, 1]
    """
    _check_torch()
    return torch.sigmoid(theta / epsilon) * mask


def perturbed_repair_budgeted(theta, mask, weights, capacity,
                              epsilon=0.1, mc_samples=8, rng=None):
    """Monte-Carlo perturb-and-MAP smoothing under a per-stage repair budget.

        r_ε = E_Z[ argmax_{r in Y}  (θ + ε Z) · r ],
        Y = { r : r <= mask,  β·r <= B,  r ∈ {0,1} },   Z ~ N(0, I),

    estimated with ``mc_samples`` Gaussian draws and the exact knapsack oracle
    (knapsack.budgeted_argmax). ``weights`` are the repair costs β and
    ``capacity`` is the budget B. When ``capacity`` is None there is no budget
    and the function returns perturbed_repair(theta, mask, epsilon).

    theta and mask have shape (n,) or (batch, n). The estimate is computed in
    NumPy and returned as a tensor without gradient. ``rng`` is a NumPy
    Generator for the draws; None draws from the global np.random state.

    The unbudgeted closed form is the logistic (sigmoid) map, while this
    estimator uses Gaussian noise. If the budget never binds, every draw
    returns the threshold set of θ + εZ and the estimate tends to Φ(θ/ε)·mask
    as mc_samples → ∞, not to the sigmoid.
    """
    from knapsack import budgeted_argmax, budget_disabled
    if budget_disabled(capacity):
        return perturbed_repair(theta, mask, epsilon)

    th = theta.detach().cpu().numpy()
    mk = mask.detach().cpu().numpy()
    w  = np.asarray(weights, dtype=float)
    single = (th.ndim == 1)
    if single:
        th = th[None, :]
        mk = mk[None, :]
    bsz, n = th.shape
    draw = (rng.standard_normal if rng is not None else np.random.standard_normal)

    acc = np.zeros((bsz, n))
    for b in range(bsz):
        for _ in range(mc_samples):
            z = draw(n)
            acc[b] += budgeted_argmax(th[b] + epsilon * z, mk[b], w, capacity)
    acc /= mc_samples

    out = torch.as_tensor(acc, dtype=theta.dtype, device=theta.device)
    return out[0] if single else out


# ---------------------------------------------------------------------------
# Fenchel-Young loss
# ---------------------------------------------------------------------------

class FenchelYoungLoss(nn.Module):
    """
    Fenchel-Young loss for the DFL repair network.

    ∇_θ L_FY(θ, r*) = r_ε(θ) − r*

    Implemented as the pseudo-loss (r_ε − r*) · θ with r_ε computed under
    no_grad, so autograd returns exactly the FY gradient and never
    differentiates through r_ε. The pseudo-loss has the FY gradient but not
    the FY loss value (it can be negative); the train and validation losses
    reported during training are pseudo-loss values.

    With capacity=None, r_ε is the closed-form sigmoid (perturbed_repair).
    Otherwise r_ε is the Monte-Carlo estimate over the knapsack
    (perturbed_repair_budgeted) with repair costs ``weights`` and
    ``mc_samples`` draws, and the gradient is a Monte-Carlo estimate.
    """

    def __init__(self, epsilon=0.1, weights=None, capacity=None, mc_samples=8):
        _check_torch()
        super().__init__()
        self.epsilon    = epsilon
        # Budget settings: capacity=None selects the closed-form surrogate,
        # otherwise the budgeted Monte-Carlo perturb-and-MAP is used.
        self.weights    = weights
        self.capacity   = capacity
        self.mc_samples = mc_samples

    def forward(self, theta, r_star, mask):
        """
        Parameters
        ----------
        theta  : Tensor, shape (batch, n_components)
        r_star : Tensor, shape (batch, n_components), target repairs (binary)
        mask   : Tensor, shape (batch, n_components), 1 = failed component

        Returns
        -------
        loss   : scalar Tensor, batch mean of the pseudo-loss (r_soft − r_star) · θ
        r_soft : Tensor (batch, n_components), smoothed decisions r_ε
        """
        with torch.no_grad():
            if self.capacity is None:
                r_soft = perturbed_repair(theta, mask, self.epsilon)
            else:
                r_soft = perturbed_repair_budgeted(
                    theta, mask, self.weights, self.capacity,
                    self.epsilon, self.mc_samples)

        # r_soft carries no gradient, so d(loss)/dθ = (r_soft - r_star) / batch
        loss = ((r_soft - r_star) * theta).sum(dim=-1).mean()

        return loss, r_soft


# ---------------------------------------------------------------------------
# Hard inference decision
# ---------------------------------------------------------------------------

def hard_repair(theta, mask):
    """
    Hard repair decision without a budget (threshold rule).

    r[k] = 1  iff  θ[k] > 0  and  mask[k] = 1

    The rollout (PseudoDFLRollout) calls knapsack.budgeted_argmax instead,
    which gives this same decision when there is no budget.

    Parameters
    ----------
    theta : array-like or Tensor, shape (..., n_components)
    mask  : array-like or Tensor, shape (..., n_components)

    Returns
    -------
    r : np.ndarray  shape (..., n_components), dtype float
    """
    if _TORCH_AVAILABLE and isinstance(theta, torch.Tensor):
        theta = theta.detach().cpu().numpy()
    if _TORCH_AVAILABLE and isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()
    theta = np.asarray(theta, dtype=float)
    mask  = np.asarray(mask,  dtype=float)
    return ((theta > 0).astype(float) * mask)


# ---------------------------------------------------------------------------
# Load a saved model
# ---------------------------------------------------------------------------

def load_repair_network(path, input_dim, n_components,
                        hidden_dims=(64, 128, 64), dropout=0.1):
    """Load a saved RepairNetwork from a checkpoint file.

    hidden_dims must match the trained network (checkpoints do not store the
    hidden widths). The dropout rate does not matter, since Dropout has no
    weights, but dropout = 0 leaves out the Dropout layers and shifts the layer
    indices in the state dict: dropout must be zero here iff it was zero in
    training.
    Returns (network in eval mode, epsilon stored in the checkpoint or 0.1).
    """
    _check_torch()
    net = RepairNetwork(input_dim, n_components, hidden_dims, dropout)
    ckpt = torch.load(path, map_location='cpu')
    net.load_state_dict(ckpt['model_state'])
    net.eval()
    return net, ckpt.get('epsilon', 0.1)


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    _check_torch()
    import torch

    input_dim    = 8   # toy (5-bus) network: 2 gen + 5 line + 1 stage_frac
    n_components = 7   # 2 gen + 5 line

    torch.manual_seed(0)   # reproducible test outputs
    net = RepairNetwork(input_dim, n_components)
    print("RepairNetwork architecture:")
    print(net)

    x = torch.rand(32, input_dim)
    theta = net(x)
    print(f"\nBatch forward: x.shape={x.shape} → theta.shape={theta.shape}")

    mask     = torch.randint(0, 2, (32, n_components)).float()
    r_star   = torch.zeros(32, n_components)

    fy_loss = FenchelYoungLoss(epsilon=0.1)
    loss, r_soft = fy_loss(theta, r_star, mask)
    print(f"FY loss: {loss.item():.4f}")

    r_hard = hard_repair(theta, mask)
    repair_rate = r_hard.mean()
    print(f"Hard repair rate: {repair_rate:.3f}  (about 0.25 for random θ and a half-full mask)")
