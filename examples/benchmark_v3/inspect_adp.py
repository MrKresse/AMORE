import numpy as np
p = r'C:\Users\kr3ss\Desktop\ZIBwork\AMORE\examples\benchmark\data\alanine_koopman.npz'
d = np.load(p)
print("lag_ps  =", d["lag_ps"])
print("n_bursts=", d["n_bursts"])
print("temp_K  =", d["temp_K"])
print("anchors_cart dtype/shape:", d["anchors_cart"].shape)
print("anchors_feat sample[0,:5]:", d["anchors_feat"][0,:5])
