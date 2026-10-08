"""
Pseudo_DFL Training Module
==========================
Trains RepairNetwork with the Fenchel-Young loss.

Training loop:
  1. Flatten the solved scenarios into (N, input_dim), (N, n_comp), (N, n_comp)
     arrays of features, target repairs and masks (one row per scenario and stage)
  2. Random train/validation split of the rows (not of whole scenarios)
  3. Mini-batch Adam with weight decay, gradient-norm clipping and
     ReduceLROnPlateau on the validation loss
  4. Early stopping on the validation loss; at the end the weights of the best
     validation epoch are restored (save() writes them to disk)
  5. Every epoch also records the validation repair ratio (repairs ordered by
     the threshold rule / target repairs)

Usage:
    trainer = PseudoDFLTrainer(network, params)
    trainer.train(dataset, n_epochs=200)
    trainer.save('model.pt')
"""

import numpy as np
from pathlib import Path

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

from Pseudo_DFL_model import RepairNetwork, FenchelYoungLoss
from Pseudo_DFL_data import flatten_dataset
from config import (
    NN_HIDDEN_DIMS, NN_DROPOUT, FY_EPSILON, FY_MC_SAMPLES,
    TRAIN_N_EPOCHS, TRAIN_BATCH_SIZE, TRAIN_LR,
    TRAIN_WEIGHT_DECAY, TRAIN_VAL_SPLIT, TRAIN_GRAD_CLIP,
    TRAIN_SCHED_PATIENCE, TRAIN_SCHED_FACTOR, TRAIN_EARLY_STOP_PATIENCE,
    DEFAULT_TRAIN_SEED, DEFAULT_INIT_SEED,
)


def _require_torch():
    if not _TORCH_AVAILABLE:
        raise ImportError("PyTorch required. Install with:  pip install torch")


class PseudoDFLTrainer:
    """Trains a RepairNetwork with the Fenchel-Young loss."""

    def __init__(self, network, params, epsilon=FY_EPSILON,
                 hidden_dims=NN_HIDDEN_DIMS, dropout=NN_DROPOUT, seed=DEFAULT_TRAIN_SEED,
                 init_seed=DEFAULT_INIT_SEED):
        """
        Parameters
        ----------
        network     : NetworkData
        params      : ModelParameters (repair_budget_per_stage selects the loss
                      surrogate; the repair costs are the knapsack weights)
        epsilon     : float   FY perturbation strength
        hidden_dims : tuple   hidden layer widths for RepairNetwork
        dropout     : float   dropout rate
        seed        : int     default seed of train(): train/val split, batch
                              order and dropout
        init_seed   : int     RNG seed for the network weight initialization
        """
        _require_torch()
        self.network_data = network
        self.params       = params
        self.epsilon      = epsilon
        self.seed         = seed

        n_gen   = network.n_generators
        n_lines = network.n_lines
        n_comp  = n_gen + n_lines
        in_dim  = n_gen + n_lines + 1   # a_gen, a_line, stage_frac

        self.in_dim  = in_dim
        self.n_comp  = n_comp

        torch.manual_seed(init_seed)
        self.model     = RepairNetwork(in_dim, n_comp, hidden_dims, dropout)
        # With a repair budget (repair_budget_per_stage not None) the loss uses
        # the budgeted perturb-and-MAP surrogate, otherwise the closed-form
        # sigmoid. Repair-cost weights are ordered [gen, line] like the network
        # output.
        _budget  = getattr(params, 'repair_budget_per_stage', None)
        _weights = np.concatenate([np.asarray(params.gen_repair_cost,  float),
                                   np.asarray(params.line_repair_cost, float)])
        self.criterion = FenchelYoungLoss(epsilon, weights=_weights,
                                          capacity=_budget,
                                          mc_samples=FY_MC_SAMPLES)

        self.train_losses = []
        self.val_losses   = []
        self.val_accs     = []   # validation repair ratio per epoch

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, dataset, n_epochs=TRAIN_N_EPOCHS, batch_size=TRAIN_BATCH_SIZE,
              lr=TRAIN_LR, val_split=TRAIN_VAL_SPLIT, verbose=True, seed=None,
              early_stop_patience=TRAIN_EARLY_STOP_PATIENCE,
              weight_decay=TRAIN_WEIGHT_DECAY, sched_patience=TRAIN_SCHED_PATIENCE,
              sched_factor=TRAIN_SCHED_FACTOR, grad_clip=TRAIN_GRAD_CLIP):
        """
        Train the model.

        Parameters
        ----------
        dataset              : dict from Pseudo_DFL_data.generate_training_data
        n_epochs             : int     maximum number of epochs
        batch_size           : int
        lr                   : float   initial Adam learning rate
        val_split            : float   fraction of samples held out for validation
        verbose              : bool
        seed                 : int or None  seed of the split, batch order and
                               dropout; None uses self.seed
        early_stop_patience  : int     halt if val loss has not improved for this many
                               epochs (0 or None disables early stopping)
        weight_decay         : float   Adam weight decay
        sched_patience       : int     ReduceLROnPlateau patience (epochs)
        sched_factor         : float   ReduceLROnPlateau learning-rate multiplier
        grad_clip            : float   max gradient norm (clip_grad_norm_)
        """
        _require_torch()

        # ---- Flatten dataset (solved scenarios only) -------------------
        features, r_target, mask = flatten_dataset(
            dataset, only_solved=True, only_repairable=False
        )

        N = len(features)
        if N == 0:
            raise ValueError("No valid training samples found in dataset.")

        # ---- Split train / validation ----------------------------------
        _split_seed = seed if seed is not None else self.seed
        rng = np.random.default_rng(_split_seed)
        idx = rng.permutation(N)
        # At least 5 validation samples when N >= 25, at least 1 below that
        # (N = 25 is where the default 20% split gives 5).
        n_val   = max(5, int(N * val_split)) if N >= 25 else max(1, int(N * val_split))
        n_train = N - n_val
        tr_idx, va_idx = idx[:n_train], idx[n_train:]

        def to_t(arr, rows):
            return torch.FloatTensor(arr[rows])

        Xtr, Rtr, Mtr = to_t(features, tr_idx), to_t(r_target, tr_idx), to_t(mask, tr_idx)
        Xva, Rva, Mva = to_t(features, va_idx), to_t(r_target, va_idx), to_t(mask, va_idx)

        train_ds = TensorDataset(Xtr, Rtr, Mtr)
        # Global torch RNG seeded for deterministic dropout; a separate DataLoader
        # Generator keeps the shuffle order independent of any other RNG use before
        # the first batch. Under a repair budget the Monte-Carlo perturbations in
        # the loss draw from NumPy's global RNG, which is not seeded here; unless
        # the caller seeds it, budgeted training does not repeat exactly.
        torch.manual_seed(_split_seed)
        _dl_gen = torch.Generator()
        _dl_gen.manual_seed(_split_seed)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                                  generator=_dl_gen)

        # ---- Optimizer + scheduler ------------------------------------
        optimizer  = torch.optim.Adam(self.model.parameters(), lr=lr,
                                      weight_decay=weight_decay)
        scheduler  = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=sched_patience, factor=sched_factor
        )

        if verbose:
            print(f"  Training:   {n_train} samples  |  "
                  f"Validation: {n_val} samples")
            print(f"  Epochs: {n_epochs}  |  Batch: {batch_size}  "
                  f"|  LR: {lr}  |  ε: {self.epsilon}")
            print()

        best_val  = float('inf')
        best_state = None
        no_improve = 0

        for epoch in range(1, n_epochs + 1):

            # -- Training pass --
            self.model.train()
            ep_loss = 0.0
            for Xb, Rb, Mb in train_loader:
                optimizer.zero_grad()
                theta      = self.model(Xb)
                loss, _    = self.criterion(theta, Rb, Mb)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=grad_clip)
                optimizer.step()
                ep_loss += loss.item() * len(Xb)
            ep_loss /= n_train

            # -- Validation pass --
            self.model.eval()
            with torch.no_grad():
                theta_va        = self.model(Xva)
                val_loss, r_soft = self.criterion(theta_va, Rva, Mva)
                val_loss         = val_loss.item()

                # Repair ratio: number of repairs ordered by the threshold rule
                # (θ > 0 on failed components, budget ignored) divided by the
                # number of target repairs, over the whole validation set. It
                # compares counts and is not an accuracy.
                hard = (theta_va > 0).float() * Mva
                opt_repairs = Rva.sum().item()
                dfl_repairs = hard.sum().item()
                repair_ratio = dfl_repairs / opt_repairs if opt_repairs > 0 else float('nan')

            self.train_losses.append(ep_loss)
            self.val_losses.append(val_loss)
            self.val_accs.append(repair_ratio)

            scheduler.step(val_loss)

            if val_loss < best_val:
                best_val   = val_loss
                best_state = {k: v.clone() for k, v in
                              self.model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1

            if verbose and (epoch % max(1, n_epochs // 10) == 0):
                lr_now = optimizer.param_groups[0]['lr']
                print(f"  Epoch {epoch:4d}/{n_epochs}: "
                      f"FY_train={ep_loss:.4f}  FY_val={val_loss:.4f}  "
                      f"repair_ratio={repair_ratio:.3f}  lr={lr_now:.2e}")

            if early_stop_patience and no_improve >= early_stop_patience:
                if verbose:
                    print(f"  Early stopping at epoch {epoch} "
                          f"(no val improvement for {early_stop_patience} epochs)")
                break

        # Restore the weights of the epoch with the lowest validation loss
        if best_state is not None:
            self.model.load_state_dict(best_state)

        if verbose:
            print(f"\n  Best FY_val: {best_val:.4f}")

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path):
        """Save model weights, training history, epsilon, in_dim and n_comp to a .pt file.

        The hidden layer widths are not stored; reloading needs the same hidden_dims.
        """
        _require_torch()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'model_state':   self.model.state_dict(),
            'train_losses':  self.train_losses,
            'val_losses':    self.val_losses,
            'val_accs':      self.val_accs,
            'epsilon':       self.epsilon,
            'in_dim':        self.in_dim,
            'n_comp':        self.n_comp,
        }, path)
        print(f"  Model saved → {path}")

    def load(self, path):
        """Load model weights, training history and epsilon from a .pt checkpoint."""
        _require_torch()
        ckpt = torch.load(path, map_location='cpu')
        self.model.load_state_dict(ckpt['model_state'])
        self.train_losses = ckpt.get('train_losses', [])
        self.val_losses   = ckpt.get('val_losses',   [])
        self.val_accs     = ckpt.get('val_accs',     [])
        self.epsilon      = ckpt.get('epsilon', self.epsilon)
        print(f"  Model loaded ← {path}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def print_summary(self):
        """Print the best validation loss, the repair ratio at that epoch and the
        final training loss."""
        if not self.train_losses:
            print("  No training history available.")
            return
        best_ep  = int(np.argmin(self.val_losses))
        best_ratio = self.val_accs[best_ep] if self.val_accs else float('nan')
        print(f"  Best FY_val   : {self.val_losses[best_ep]:.4f} "
              f"(epoch {best_ep+1})")
        print(f"  Best repair_ratio: {best_ratio:.3f}")
        print(f"  Final FY_train: {self.train_losses[-1]:.4f}")


# ---------------------------------------------------------------------------

def create_trainer(network, params, epsilon=FY_EPSILON,
                   hidden_dims=NN_HIDDEN_DIMS, dropout=NN_DROPOUT, seed=DEFAULT_TRAIN_SEED,
                   init_seed=DEFAULT_INIT_SEED):
    """Build a PseudoDFLTrainer (same arguments as its constructor)."""
    return PseudoDFLTrainer(network, params, epsilon, hidden_dims, dropout, seed=seed,
                            init_seed=init_seed)


if __name__ == "__main__":
    _require_torch()
    from network_data import create_network
    from model_parameters import create_model_parameters
    from Pseudo_DFL_data import generate_training_data

    net    = create_network()
    params = create_model_parameters(net, seed=42)

    print("Generating small training dataset …")
    dataset, _ = generate_training_data(
        net, params, n_scenarios=20, solver_name=None, seed=42, verbose=True
    )

    print("\nTraining RepairNetwork …")
    trainer = create_trainer(net, params, epsilon=0.1)
    trainer.train(dataset, n_epochs=30, batch_size=32, verbose=True)
    trainer.print_summary()
