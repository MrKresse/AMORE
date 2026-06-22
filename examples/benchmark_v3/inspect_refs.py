import numpy as np

tw = np.load(r'C:\Users\kr3ss\Desktop\ZIBwork\AMORE\examples\benchmark_v2\panel0\tw_committor.npz')
pA, pB, pC = tw['p_A'], tw['p_B'], tw['p_C']
row_sum = pA + pB + pC
print(f"TW committor: N={len(pA)}")
print(f"  p_A+p_B+p_C: mean={row_sum.mean():.4f} std={row_sum.std():.4f} min={row_sum.min():.4f} max={row_sum.max():.4f}")
print(f"  p_A: mean={pA.mean():.3f}  p_B: mean={pB.mean():.3f}  p_C: mean={pC.mean():.3f}")

ev = np.load(r'C:\Users\kr3ss\Desktop\ZIBwork\AMORE\examples\benchmark_v2\panel0\adp_eigvecs.npz')
evals = ev['eigenvalues']
print(f"\nADP transfer-operator eigenvalues (tau=5ps):")
print(f"  {evals}")
tau = 5.0
its = [-tau / np.log(abs(l)) if abs(l) < 1 - 1e-9 else np.inf for l in evals]
print(f"  ITS (ps): {[f'{t:.1f}' if t < 1e4 else 'inf' for t in its]}")
print(f"  occupied cells: {ev['occupied'].sum()} / 1600")
