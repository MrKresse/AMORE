# Export benchmark data to raw little-endian binaries Julia can read without NPZ.
# Arrays are written in C-order of their native numpy shape; Julia reshapes to
# the transposed (feature-first) shape using column-major equivalence:
#   numpy (N,F)   C-order  ==  Julia (F,N)   col-major
#   numpy (N,K,F) C-order  ==  Julia (F,K,N) col-major
#   numpy (N,5)   C-order  ==  Julia (5,N)   col-major   (splits stored .T)
import numpy as np, os

D = r"C:\Users\kr3ss\Desktop\ZIBwork\AMORE\examples\benchmark\data"
OUT = r"C:\Users\kr3ss\Desktop\ZIBwork\AMORE\examples\benchmark_v3\_jldata"
os.makedirs(OUT, exist_ok=True)

def w(name, arr, dtype):
    a = np.ascontiguousarray(arr.astype(dtype))
    a.tofile(os.path.join(OUT, name + ".bin"))
    print(name, arr.shape, "->", a.dtype, a.size)

tw   = np.load(os.path.join(D, "triple_well_koopman.npz"))
al5  = np.load(os.path.join(D, "alanine_koopman.npz"))
al01 = np.load(os.path.join(D, "alanine_0p1ps_koopman.npz"))

# triple well
w("tw_anchors", tw["anchors"], np.float32)            # (1600,2)   -> Julia (2,1600)
w("tw_bursts",  tw["bursts"],  np.float32)            # (1600,20,2)-> Julia (2,20,1600)
w("tw_splits",  tw["patch_splits"].T, np.int32)       # (1600,5)   -> Julia (5,1600)

# alanine
w("al_anchors", al5["anchors_feat"], np.float32)      # (1578,231) -> Julia (231,1578)
w("al5_bursts", al5["bursts_feat"],  np.float32)      # (1578,20,231)
w("al01_bursts", al01["bursts_feat"], np.float32)
w("al_splits",  al5["patch_splits"].T, np.int32)      # (1578,5)   -> Julia (5,1578)
joint = np.concatenate([al5["bursts_feat"], al01["bursts_feat"]], axis=1)  # (1578,40,231)
w("al_joint_bursts", joint, np.float32)

print("EXPORT_OK")
