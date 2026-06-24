# -*- coding: utf-8 -*-
"""
theory_demo.py — why real-valued chi + ShiftScale is robust on non-reversible systems,
and where it silently breaks.

Claim
-----
A non-reversible Koopman operator K has complex eigenvalues, but they come in conjugate
pairs, so the dominant non-stationary INVARIANT SUBSPACE is real (span{Re v, Im v}). Real
chi (ISOKANN/ISA) lives in that real subspace -> robust, good classification. BUT a complex
eigenvalue lambda = r e^{i theta} means K ROTATES the iterate by theta within that 2-D
subspace each lag instead of scaling it, so:
  (1) there is no single real eigenfunction g  =>  the 2-state rate equation
      L chi = -eps1 chi + eps2 (1-chi) has no exact real solution; the generator
      eigenvalue eps = log(lambda)/tau is COMPLEX, Im(eps)=theta/tau = circulation frequency.
  (2) the von-Mises iteration  chi <- shiftscale(K chi)  PRECESSES (rotates), it does not
      converge to a fixed real function -- the late-training drift seen in the benchmark.

This script builds the directed-ring angular transfer operator and a reversible control,
and (A) shows the spectrum, (B) tracks the angle of the ISOKANN iterate in the dominant
2-D subspace over iterations (rotation vs convergence), (C) prints the complex rate.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import systems as ts   # consolidated loader (provides _ring_propagate / _ring_basin)


def angular_transfer_operator(reversible: bool, nbins: int = 72, burst: int = 250,
                              n_per_bin: int = 400, seed: int = 0):
    """Fine angular (1-D, periodic) transfer operator of the ring, estimated from short
    bursts. reversible=True zeroes the tangential drift (kappa=0) -> detailed balance."""
    kappa = 0.0 if reversible else 2.0
    rng = np.random.default_rng(seed)
    edges = np.linspace(-np.pi, np.pi, nbins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    # seed anchors uniformly per angular bin at radius R, run one burst each
    th0 = np.repeat(centers, n_per_bin) + 0.02 * rng.standard_normal(nbins * n_per_bin)
    R = 1.0
    x = np.stack([R * np.cos(th0), R * np.sin(th0)], 1)
    x = ts._ring_propagate(x, 40, 2e-3, 0.25, 6.0, 1.2, R, kappa, rng)   # settle
    a_bin = np.clip(np.digitize(np.arctan2(x[:, 1], x[:, 0]), edges) - 1, 0, nbins - 1)
    xt = ts._ring_propagate(x.copy(), burst, 2e-3, 0.25, 6.0, 1.2, R, kappa, rng)
    b_bin = np.clip(np.digitize(np.arctan2(xt[:, 1], xt[:, 0]), edges) - 1, 0, nbins - 1)
    C = np.zeros((nbins, nbins)); np.add.at(C, (a_bin, b_bin), 1.0)
    C += 1e-6
    T = C / C.sum(1, keepdims=True)
    return T, centers


def spectrum(T):
    w, V = np.linalg.eig(T)
    o = np.argsort(-np.abs(w))
    return w[o], V[:, o]


def iterate_angle(T, n_iter=60, seed=1):
    """ISOKANN-style iteration chi <- shiftscale(centered(T@chi)); track the iterate's
    angle in the dominant non-stationary 2-D subspace (Re v2, Im v2). Returns angles (deg)
    and the per-step subspace energy fraction (how much of chi stays in that plane)."""
    w, V = spectrum(T)
    re2, im2 = np.real(V[:, 1]), np.imag(V[:, 1])      # dominant non-stationary pair basis
    # orthonormalize the real 2-D basis
    B = np.stack([re2, im2], 1)
    Q, _ = np.linalg.qr(B)                              # (nbins, 2)
    rng = np.random.default_rng(seed)
    chi = rng.standard_normal(T.shape[0])
    chi = (chi - chi.min()) / (chi.max() - chi.min())
    angles, frac = [], []
    for _ in range(n_iter):
        chi = T @ chi
        c = chi - chi.mean()
        coords = Q.T @ c                                # projection onto the 2-D plane
        angles.append(np.degrees(np.arctan2(coords[1], coords[0])))
        frac.append(np.linalg.norm(coords) / (np.linalg.norm(c) + 1e-12))
        rng_ = chi.max() - chi.min()
        chi = (chi - chi.min()) / (rng_ + 1e-12)        # shiftscale to [0,1]
    return np.array(angles), np.array(frac), w


def make_figure(path="figures/theory_rotation.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for rev, c, lab in [(False, "crimson", "directed (κ=2)"), (True, "steelblue", "reversible (κ=0)")]:
        T, _ = angular_transfer_operator(reversible=rev)
        ang, _, w = iterate_angle(T)
        ax[0].plot(ang, "o-", ms=3, color=c, label=f"{lab}  argλ₂={np.degrees(np.angle(w[1])):+.1f}°")
        ev = w[:6]
        ax[1].scatter(ev.real, ev.imag, color=c, s=40, label=lab, zorder=3)
    ax[0].set_xlabel("ISOKANN iteration"); ax[0].set_ylabel("iterate angle in dominant 2-D subspace (°)")
    ax[0].set_title("von-Mises iterate: precesses (directed) vs converges (reversible)")
    ax[0].legend(fontsize=8)
    th = np.linspace(0, 2*np.pi, 200)
    ax[1].plot(np.cos(th), np.sin(th), "k:", lw=0.7)
    ax[1].axhline(0, color="gray", lw=0.5); ax[1].set_aspect("equal")
    ax[1].set_xlabel("Re λ"); ax[1].set_ylabel("Im λ")
    ax[1].set_title("transfer-operator spectrum (complex pair ⇒ current)"); ax[1].legend(fontsize=8)
    plt.tight_layout(); fig.savefig(path, dpi=120, bbox_inches="tight"); plt.close(fig)
    print(f"  wrote {path}")


if __name__ == "__main__":
    for rev in [False, True]:
        tag = "REVERSIBLE ring (kappa=0)" if rev else "NON-REVERSIBLE directed ring (kappa=2)"
        T, centers = angular_transfer_operator(reversible=rev)
        w, V = spectrum(T)
        ang, frac, _ = iterate_angle(T)
        print(f"\n=== {tag} ===")
        print(f"  top eigenvalues: {np.round(w[:4], 4)}")
        lam2 = w[1]
        tau = 250 * 2e-3                                # burst steps * dt
        eps = np.log(lam2) / tau                        # generator eigenvalue
        print(f"  dominant non-stationary lambda_2 = {lam2:.4f}  |lambda|={abs(lam2):.4f}  "
              f"arg={np.degrees(np.angle(lam2)):+.2f} deg")
        print(f"  generator rate eps = log(lambda)/tau = {eps.real:+.3f} {eps.imag:+.3f}i   "
              f"(Im = circulation freq; 0 => reversible)")
        print(f"  ISOKANN iterate angle in 2-D subspace, every 10 iters: "
              f"{np.round(ang[::10], 1)}")
        d = np.diff(ang); d = (d + 180) % 360 - 180     # per-step rotation (wrapped)
        print(f"  mean per-iter rotation = {np.mean(d[-30:]):+.2f} deg/iter "
              f"(arg lambda = {np.degrees(np.angle(lam2)):+.2f}); "
              f"subspace energy frac (last) = {frac[-1]:.2f}")
        print(f"  => {'PRECESSES (no fixed real chi)' if abs(np.mean(d[-30:]))>2 else 'CONVERGES (fixed real chi)'}")
    make_figure()
