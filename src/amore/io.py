import numpy as np
import torch as pt
import h5py
import matplotlib.pyplot as plt
import MDAnalysis as mda
from MDAnalysis.coordinates.PDB import PDBWriter

from .chi import chi_coords


def save_gradient(path, G):
    """
    Save per-atom gradient norms to HDF5.

    The dataset is stored so that loading with
        np.array(f['mean_gradients']).T
    returns shape (n_atoms, n_frames) or (n_atoms,).

    Parameters
    ----------
    path : str
        Output HDF5 file path.
    G : pt.Tensor or np.ndarray
        Shape (n_atoms,) or (nbins, n_atoms).
    """
    if isinstance(G, pt.Tensor):
        G = G.detach().cpu().numpy()
    else:
        G = np.asarray(G)

    with h5py.File(path, "w") as f:
        if G.ndim == 1:
            f.create_dataset("mean_gradients", data=G)
        elif G.ndim == 2:
            f.create_dataset("mean_gradients", data=G.T)
        else:
            raise ValueError(f"G must be 1D or 2D, got shape {G.shape}")


def save_pdb(x, path_pdb, path_out):
    """
    Save coordinates using MDAnalysis (preserves full topology).

    Parameters
    ----------
    x : pt.Tensor or np.ndarray
        Shape (nframes, 3*Natoms) or (3*Natoms,).
    path_pdb : str
        Reference PDB providing atom order/topology.
    path_out : str
        Output PDB path.
    """
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    else:
        x = np.asarray(x)

    if x.ndim == 1:
        x = x[None, :]

    nframes, D = x.shape
    assert D % 3 == 0, "x must be (nframes, 3*Natoms)"
    Natoms = D // 3
    coords = x.reshape(nframes, Natoms, 3)

    u = mda.Universe(path_pdb)
    assert len(u.atoms) == Natoms, (
        f"Atom mismatch: x has {Natoms}, PDB has {len(u.atoms)}"
    )

    with PDBWriter(path_out, multiframe=(nframes > 1)) as W:
        for i in range(nframes):
            u.atoms.positions = coords[i]
            W.write(u.atoms)


def save_pdb_as_frames(x, path_pdb, path_out):
    """
    Save coordinates as a multi-frame PDB (MODEL/ENDMDL blocks).

    Uses raw ATOM/HETATM lines from path_pdb as a template so PyMOL
    loads the output as a single object with N states.

    Parameters
    ----------
    x : pt.Tensor or np.ndarray
        Shape (nframes, 3*Natoms) or (3*Natoms,).
    path_pdb : str
        Reference PDB (ATOM/HETATM lines used as template).
    path_out : str
        Output PDB path.
    """
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    else:
        x = np.asarray(x)

    if x.ndim == 1:
        x = x[None, :]

    nframes, D = x.shape
    assert D % 3 == 0, "x must be (nframes, 3*Natoms)"
    Natoms = D // 3
    coords = x.reshape(nframes, Natoms, 3)

    atom_lines = []
    other_header = []
    with open(path_pdb, "r") as f:
        for line in f:
            rec = line[:6]
            if rec.startswith("ATOM  ") or rec.startswith("HETATM"):
                atom_lines.append(line.rstrip("\n"))
            elif not (
                rec.startswith("MODEL ")
                or rec.startswith("ENDMDL")
                or rec.startswith("END")
            ):
                other_header.append(line.rstrip("\n"))

    assert len(atom_lines) == Natoms, (
        f"Atom mismatch: x has {Natoms}, PDB has {len(atom_lines)} ATOM/HETATM records"
    )

    def _with_xyz(pdb_line, xyz):
        x_, y_, z_ = xyz
        xyz_str = f"{x_:8.3f}{y_:8.3f}{z_:8.3f}"
        s = pdb_line.ljust(54)
        return s[:30] + xyz_str + s[54:]

    with open(path_out, "w") as out:
        for line in other_header:
            out.write(line + "\n")
        for fi in range(nframes):
            out.write(f"MODEL     {fi + 1:4d}\n")
            for ai in range(Natoms):
                out.write(_with_xyz(atom_lines[ai], coords[fi, ai]) + "\n")
            out.write("ENDMDL\n")
        out.write("END\n")


def save_chi_histogram(nu, featurizer, xs, out, bins=100):
    """
    Plot and save a histogram of chi values.

    Parameters
    ----------
    nu : callable
    featurizer : callable
    xs : pt.Tensor
        Shape (nframes, 3*Natoms).
    out : str
        Output image path (e.g. 'chi.png').
    bins : int
    """
    with pt.no_grad():
        chi = chi_coords(nu, featurizer, xs)
        chi_vals = chi.reshape(xs.shape[0], -1)[:, 0]

    chi_np = chi_vals.detach().cpu().numpy()

    plt.figure(figsize=(6, 4))
    plt.hist(chi_np, bins=bins)
    plt.xlabel(r"$\chi_i$")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(out, dpi=300)
    plt.close()

    print(f"saved histogram to: {out}")
