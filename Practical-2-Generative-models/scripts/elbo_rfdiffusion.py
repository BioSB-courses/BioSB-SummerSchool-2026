"""
Estimate how "likely" a generated RFdiffusion backbone is under the trained
model, via Monte Carlo sampling over diffusion timesteps t.

Usage:
    conda activate protein-design
    python scripts/elbo_rfdiffusion.py \
        inference.input_pdb=out_unconditional/out_rfdiffusion/design_0.pdb

This writes `design_0.elbo.json` next to the input pdb with two scores:

    denoising_loss  -- cheap, uniform-timestep-weighted proxy (Ca MSE +
                       rotation geodesic distance^2 averaged over sampled t).
                       Good for ranking designs against each other.
    weighted_elbo   -- proper variational bound estimate (nats), comparing
                       posterior means weighted by each timestep's
                       1/(2*sigma_t^2)-type coefficient.
"""

import json
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig

from rfdiffusion.inference.utils import get_mu_xt_x0, get_next_frames, parse_pdb, sampler_selector
from rfdiffusion.util import rigid_from_3_points


# --------------------------------------------------------------------------
# SO(3) tangent-space helper (self-contained)
# --------------------------------------------------------------------------

def rotation_geodesic_distance(R_a, R_b):
    """Angle (radians) between rotations R_a, R_b: shape (..., 3, 3) -> (...,)"""
    cos_theta = ((torch.matmul(R_a.transpose(-1, -2), R_b)
                  .diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2).clamp(-1 + 1e-7, 1 - 1e-7)
    return torch.acos(cos_theta)


def frames_from_bb_crds(crds):
    """crds: (L, 3, 3) N/Ca/C coordinates -> (L, 3, 3) rotation matrices."""
    N = crds[None, :, 0, :]
    Ca = crds[None, :, 1, :]
    C = crds[None, :, 2, :]
    R, _ = rigid_from_3_points(N, Ca, C)
    return R.squeeze(0)


def _reset_self_conditioning(sampler, L):
    """Zero out sampler.prev_pred so the next sample_step() call scores t in
    isolation (self-conditioning off), instead of crashing (first call ever)
    or reusing an unrelated timestep's prediction (subsequent calls)."""
    sampler.prev_pred = torch.zeros(1, L, 3, 3, device=sampler.device)


def _run_sample_step(sampler, t, x_t, seq_onehot):
    """One reverse-diffusion model call at timestep t, in isolation.

    x_t: (L, 27, 3) or (L, 14, 3) coordinates -- sliced to the 14 atoms
    sample_step expects. seq_onehot: (L, 22) one-hot sequence.
    Returns px0: (L, 14, 3), on CPU.

    final_step=t (not 1) makes sample_step take its `else` branch and skip
    computing x_t_1 via denoiser.get_next_pose(), which we don't use anyway:
    it internally calls the IGSO3 diffuser's g(t), which differentiates
    sigma(t)^2 via torch.autograd.grad -- a closed-form derivative trick,
    not a training gradient, but real gradient tracking (no torch.no_grad())
    must therefore stay enabled for the whole call.
    """
    L = x_t.shape[0]
    _reset_self_conditioning(sampler, L)
    px0, _, _, _ = sampler.sample_step(
        t=int(t), x_t=x_t[:, :14, :], seq_init=seq_onehot, final_step=int(t),
    )
    return px0.detach().cpu()


# --------------------------------------------------------------------------
# Option 1: loss-based proxy
# --------------------------------------------------------------------------

def estimate_denoising_loss(x0_fullatom, seq, atom_mask, sampler, n_t_samples=64, T=None):
    """
    x0_fullatom: (L, 14 or 27, 3) tensor -- coordinates of the generated structure
    seq:         (L,) integer sequence tensor, as parse_pdb / RFdiffusion expects
    atom_mask:   (L, 14 or 27) bool tensor of present atoms
    sampler:     an rfdiffusion.inference.model_runners.Sampler (or subclass)
                 instance, already through sample_init(), same object
                 run_inference.py builds

    Returns: scalar, average per-timestep loss (translation MSE + mean
    rotation geodesic distance^2). Lower = more typical under the model.
    Use this to RANK generated samples -- not a calibrated log-likelihood.
    """
    diffuser = sampler.diffuser
    T = T or diffuser.T
    L = x0_fullatom.shape[0]
    diffusion_mask = torch.zeros(L, dtype=torch.bool)  # nothing fixed: whole thing generated
    seq_onehot = torch.nn.functional.one_hot(seq, num_classes=22).float()

    ts = np.random.randint(1, T + 1, size=n_t_samples)
    trans_losses, rot_losses = [], []

    for t in ts:
        fa_stack, _ = diffuser.diffuse_pose(
            x0_fullatom, seq, atom_mask,
            diffusion_mask=diffusion_mask, t_list=[int(t)],
        )
        x_t = fa_stack[0]  # (L, 27, 3)

        px0 = _run_sample_step(sampler, t, x_t, seq_onehot)  # (L, 14, 3)

        trans_losses.append(
            torch.mean((px0[:, 1, :] - x0_fullatom[:, 1, :]) ** 2).item()  # Ca MSE
        )

        R_true = frames_from_bb_crds(x0_fullatom[:, :3, :])
        R_pred = frames_from_bb_crds(px0[:, :3, :])
        rot_losses.append(
            torch.mean(rotation_geodesic_distance(R_true, R_pred) ** 2).item()
        )

    return float(np.mean(trans_losses) + np.mean(rot_losses))


# --------------------------------------------------------------------------
# Option 2: full weighted ELBO -- TODO: fill in the gaps
# --------------------------------------------------------------------------

def estimate_weighted_elbo(x0_fullatom, seq, atom_mask, sampler, 
                           n_t_samples=64, T=None):
    """
    x0_fullatom: (L, 14 or 27, 3) tensor. Coordinates of the generated structure
    seq:         (L,) integer sequence tensor, as parse_pdb / RFdiffusion expects.
                 One integer per residue, 22 classes (20 standard amino acids +
                 special tokens for masked/unknown residues)
    atom_mask:   (L, 14 or 27) bool tensor of present atoms.
                 Not every residue has all 14/27 atom slots resolved
                 (e.g. missing side-chain atoms in the PDB), 
    sampler:     rfdiffusion.inference.model_runners.Sampler (or subclass) instance, 
                 already through sample_init(), the trained diffusion model.
    n_t_samples: number of samples to draw for estimating the ELBO.
    T:           (optional) number of diffusion steps.
    
    Returns an estimate of the negative ELBO (upper bound on -log p_theta(x0); 
    lower is more likely). Omits the T-prior term (negligible for large T) and 
    the t=1 reconstruction term.
    """

    # the 'encoder' that adds noise to a real structure
    diffuser = sampler.diffuser

    # the 'decoder' that denoises
    denoiser = sampler.denoiser

    # number of diffusion steps
    T = T or diffuser.T

    # number of amino acids
    L = x0_fullatom.shape[0]

    # per-residue boolean mask for wich residues are held fixed (not diffused);
    # used for motif-scaffolding, where part of the structure is contidioned on
    diffusion_mask = torch.zeros(L, dtype=torch.bool)
    seq_onehot = torch.nn.functional.one_hot(seq, num_classes=22).float()

    # make a monte carlo estimate of the denoising term using n_t_samples samples

    # STEP 1: draw integers between 2 and T n_t_samples times (t=1 excluded, separate recon term)
    ts = ...

    terms = []
    for t in ts:
        # call diffuser model to add noise to generated structure
        fa_stack, _ = diffuser.diffuse_pose(
            x0_fullatom, seq, atom_mask,
            diffusion_mask=diffusion_mask, t_list=[int(t)],
        )
        # and keep the t-th step
        x_t = fa_stack[0]

        # STEP 2: use _run_sample_step to denoise from x_t to px0
        px0 = ...  # (L, 14, 3)

        # ---- translation KL, via RFdiffusion's own posterior-mean function ----

        # the noise schedule (beta_schedule/alphabar_schedule) is calibrated for
        # coordinates in a normalized, roughly unit-variance range;
        # crd_scale converts coordinates in angstroms into that internal scale;
        # only the 3 backbone atoms (N, Ca, C) are used for the translation part
        crd_scale = denoiser.crd_scale
        xt_ca = (x_t[:, :3, :] * crd_scale)
        x0_ca = (x0_fullatom[:, :3, :] * crd_scale)
        px0_ca = (px0[:, :3, :] * crd_scale)

        # use RFdiffusion's function get_mu_xt_x0 to get the mean and variance of 
        # q(x(t-1)| x(t), x(0)) and p(x(t-1)| x(t))  
        # both are assumed Gaussian, so the function returns the mean and standard deviation
        
        # here's the function definition:
        # def get_mu_xt_x0(xt, px0, t, beta_schedule, alphabar_schedule, eps=1e-6):
            # """
            # Given xt, predicted x0 and the timestep t, give mu of x(t-1)
            # Assumes t is 0 indexed
            # """

        # STEP 3: calculate the true q(x(t-1)) of the generated structure 
        mu_true, sigma_t = get_mu_xt_x0(
            ..., ..., ...,
            beta_schedule=denoiser.schedule, alphabar_schedule=denoiser.alphabar_schedule,
        )

        # STEP 4: calculate the p(x(t-1)) of the reconstruction of the generated structure
        mu_pred, _ = get_mu_xt_x0(
            ..., ..., ...,
            beta_schedule=denoiser.schedule, alphabar_schedule=denoiser.alphabar_schedule,
        )

        # STEP 5: calculate the KL divergence of the two distributions, 
        # assuming the Gaussian and the same standard deviation sigma_t.
        # The general KL-between-Gaussians formula collapses to just a scaled
        # squared distance between means because the variance terms cancel
        trans_kl = ...

        # RFdiffusion diffuses rotations (backbone orientation) separately from 
        # translations, using an IGSO3 process on SO(3); there's no simple 
        # closed-form Gaussian KL there, so this is an approximantion

        # ---- rotation KL: compare the deterministic (noise_scale=0) reverse
        # step implied by true x0 vs. by predicted x0, in the SO(3) tangent
        # space, weighted by the IGSO3 diffuser's own per-step variance ----

        # compute the deterministic (noise-free) one-step reverse rotation update 
        # implied by x0 vs. by px0 (prediction)
        next_true = get_next_frames(
            x_t, x0_fullatom[:, :3, :], int(t),
            diffuser=diffuser, so3_type="igso3",
            diffusion_mask=diffusion_mask.numpy(), noise_scale=0.0,
        )
        next_pred = get_next_frames(
            x_t, px0[:, :3, :], int(t),
            diffuser=diffuser, so3_type="igso3",
            diffusion_mask=diffusion_mask.numpy(), noise_scale=0.0,
        )

        # convert N/Ca/C backbone atom coordinates into a 3×3 rotation matrix per 
        # residue (the local backbone frame), because SO(3) diffusion operates on 
        # rotation matrices, not raw coordinates
        R_true_next = frames_from_bb_crds(torch.as_tensor(next_true))
        R_pred_next = frames_from_bb_crds(torch.as_tensor(next_pred))

        # make a similar calculation for the rotation features:
        # square geodesic distance between the two rotation matrices, divided by
        # the diffuser's own per-step rotational variance
        continuous_t = t / diffuser.T
        rot_g = diffuser.so3_diffuser.g(continuous_t)
        rot_var = float(rot_g ** 2) * diffuser.so3_diffuser.step_size

        rot_kl = torch.mean(rotation_geodesic_distance(R_true_next, R_pred_next) ** 2) / (2 * rot_var)

        # STEP 6: add the translation and rotation KL 
        # RFdiffusion's forward process treats translations and rotations as independent 
        # random variables (independent noise processes). For independent joint distributions,
        # KL(P1×P2 || Q1×Q2) = KL(P1||Q1) + KL(P2||Q2): the joint KL of a product distribution 
        # is exactly the sum of the marginal KLs
        total_kl = ...

        terms.append(total_kl.item())

    # STEP 7: average over all monte carlo steps
    # MC estimate of the sum over T-1 steps (since ts was drawn from 2..T)
    elbo_estimate = ...

    return elbo_estimate


# --------------------------------------------------------------------------
# CLI entry point -- run standalone
# --------------------------------------------------------------------------

@hydra.main(version_base=None, config_path="../RFdiffusion/config/inference", config_name="base")
def main(conf: DictConfig) -> None:
    input_pdb = conf.inference.input_pdb
    assert input_pdb, (
        "Pass the pdb to score, e.g.:\n"
        "  python scripts/elbo_rfdiffusion.py inference.input_pdb=out_unconditional/out_rfdiffusion/design_0.pdb"
    )

    parsed = parse_pdb(input_pdb)
    x0_fullatom = torch.from_numpy(parsed["xyz"]).float()
    seq = torch.from_numpy(parsed["seq"]).long()
    atom_mask = torch.from_numpy(parsed["mask"])
    L = x0_fullatom.shape[0]

    if conf.contigmap.contigs is None:
        # Fully-designed single chain of the same length as the input pdb
        conf.contigmap.contigs = [f"{L}-{L}"]

    sampler = sampler_selector(conf)
    sampler.sample_init()

    n_t_samples_loss = int(conf.get("n_t_samples_loss", 64))
    n_t_samples_weighted = int(conf.get("n_t_samples_weighted", 64))

    denoising_loss = estimate_denoising_loss(
        x0_fullatom, seq, atom_mask, sampler, n_t_samples=n_t_samples_loss,
    )
    weighted_elbo = estimate_weighted_elbo(
        x0_fullatom, seq, atom_mask, sampler, n_t_samples=n_t_samples_weighted,
    )

    result = {
        "input_pdb": str(input_pdb),
        "length": L,
        "n_t_samples_loss": n_t_samples_loss,
        "denoising_loss": denoising_loss,
        "n_t_samples_weighted": n_t_samples_weighted,
        "weighted_elbo": weighted_elbo,
    }

    output_json = conf.get("output_json", None)
    output_path = Path(output_json) if output_json else Path(input_pdb).with_suffix("").with_suffix(".elbo.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(json.dumps(result, indent=2))
    print(f"\nWrote {output_path}")


if __name__ == "__main__":
    main()
