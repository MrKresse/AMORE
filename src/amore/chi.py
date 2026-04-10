import torch as pt


def chi_coords(nu, featurizer, xs):
    """
    Evaluate the committor/chi function for each frame.

    Parameters
    ----------
    nu : callable
        Neural network (or any callable) mapping features -> chi.
    featurizer : callable
        Maps (nframes, 3*Natoms) coordinates -> features.
    xs : pt.Tensor
        Shape (nframes, 3*Natoms).

    Returns
    -------
    chi : pt.Tensor
        Typically (nframes, 1) or (nframes,).
    """
    return nu(featurizer(xs))


def dchi_dx(nu, featurizer, xs):
    """
    Backprop gradient of the first chi component w.r.t. coordinates.

    Parameters
    ----------
    xs : pt.Tensor
        Shape (nframes, 3*Natoms). Usually called with nframes=1.

    Returns
    -------
    grad : pt.Tensor
        Same shape as xs.
    """
    xs = xs.clone().detach().requires_grad_(True)
    chi = chi_coords(nu, featurizer, xs)
    chi_scalar = chi.reshape(-1)[0]

    (grad_xs,) = pt.autograd.grad(chi_scalar, xs, create_graph=True)
    return grad_xs


def chi_sensitivity(nu, featurizer, xs, nbins=1):
    """
    Average squared gradient norm per atom, optionally binned by chi value.

    Parameters
    ----------
    xs : pt.Tensor
        Shape (nframes, 3*Natoms).
    nbins : int
        Number of chi bins.

    Returns
    -------
    bin_centers : pt.Tensor
        Shape (nbins,).
    avg_gradnorm2_per_atom : pt.Tensor
        Shape (nbins, Natoms).
    """
    N, D = xs.shape
    assert D % 3 == 0
    Natoms = D // 3

    with pt.no_grad():
        chi = chi_coords(nu, featurizer, xs)
        chi_vals = chi.reshape(N, -1)[:, 0]

    if nbins == 1:
        edges = pt.tensor(
            [chi_vals.min(), chi_vals.max()], device=xs.device, dtype=chi_vals.dtype
        )
    else:
        edges = pt.linspace(
            chi_vals.min(), chi_vals.max(), nbins + 1, device=xs.device, dtype=chi_vals.dtype
        )

    bin_centers = 0.5 * (edges[:-1] + edges[1:])
    bin_idx = pt.bucketize(chi_vals, edges[1:-1])

    sum_g2 = pt.zeros((nbins, Natoms), device=xs.device, dtype=xs.dtype)
    cnt = pt.zeros((nbins,), device=xs.device, dtype=xs.dtype)

    for i in range(N):
        grad_x = dchi_dx(nu, featurizer, xs[i : i + 1])
        grad_xyz = grad_x[0].reshape(Natoms, 3)
        g2_atom = (grad_xyz**2).sum(dim=1)

        b = int(bin_idx[i].item())
        sum_g2[b] += g2_atom
        cnt[b] += 1.0

    avg_gradnorm2_per_atom = sum_g2 / cnt.clamp_min(1.0).unsqueeze(1)
    return bin_centers, avg_gradnorm2_per_atom


def pick_representative_xs(nu, featurizer, xs, nbins=1, chi_range=None):
    """
    Pick one representative frame from each chi bin, closest to the bin center.

    Parameters
    ----------
    xs : pt.Tensor
        Shape (nframes, 3*Natoms).
    nbins : int
    chi_range : tuple(float, float) or None
        If given, restrict to frames with chi in [lo, hi].

    Returns
    -------
    xs_rep : pt.Tensor
        Shape (nbins, 3*Natoms).
    chi_centers : pt.Tensor
        Shape (nbins,).
    """
    N, D = xs.shape

    with pt.no_grad():
        chi = chi_coords(nu, featurizer, xs)
        chi_vals = chi.reshape(N, -1)[:, 0]

    if chi_range is not None:
        lo, hi = chi_range
        lo_t = pt.as_tensor(lo, device=xs.device, dtype=chi_vals.dtype)
        hi_t = pt.as_tensor(hi, device=xs.device, dtype=chi_vals.dtype)
        keep = (chi_vals >= lo_t) & (chi_vals <= hi_t)

        if not keep.any():
            return (
                pt.zeros((nbins, D), device=xs.device, dtype=xs.dtype),
                pt.full((nbins,), pt.nan, device=xs.device, dtype=chi_vals.dtype),
            )

        xs = xs[keep]
        chi_vals = chi_vals[keep]

    if nbins == 1:
        edges = pt.tensor(
            [chi_vals.min(), chi_vals.max()], device=xs.device, dtype=chi_vals.dtype
        )
    else:
        edges = pt.linspace(
            chi_vals.min(), chi_vals.max(), nbins + 1, device=xs.device, dtype=chi_vals.dtype
        )

    chi_centers = 0.5 * (edges[:-1] + edges[1:])
    bin_idx = pt.bucketize(chi_vals, edges[1:-1])

    xs_rep = pt.zeros((nbins, D), device=xs.device, dtype=xs.dtype)

    for b in range(nbins):
        mask = bin_idx == b
        if not mask.any():
            continue
        chi_b = chi_vals[mask]
        xs_b = xs[mask]
        j = pt.argmin(pt.abs(chi_b - chi_centers[b]))
        xs_rep[b] = xs_b[j]

    return xs_rep, chi_centers
