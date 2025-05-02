import numpy as np

# ANSI color codes
COLORS = ["\033[91m", "\033[92m", "\033[93m", "\033[94m", "\033[95m"]
RESET = "\033[0m"

nT, nK, nKappa, nTheta, nSigma = 2, 2, 2, 2, 2  # Small for demo
names = ["T", "K", "kappa", "theta", "sigma"]

def serpentine_idx(idx, nT, nK, nKappa, nTheta, nSigma):
    same = idx
    isigma = same % nSigma
    same //= nSigma
    itheta = same % nTheta
    same //= nTheta
    ikappa = same % nKappa
    same //= nKappa
    ik = same % nK
    same //= nK
    it = same % nT
    # Serpentine ordering
    if ik % 2 == 1:
        ikappa = nKappa - 1 - ikappa
    if itheta % 2 == 1:
        isigma = nSigma - 1 - isigma
    return [it, ik, ikappa, itheta, isigma]

def ou_rowmajor_idx(idx, nT, nK, nKappa, nTheta, nSigma):
    same = idx
    isigma = same % nSigma
    same //= nSigma
    itheta = same % nTheta
    same //= nTheta
    ikappa = same % nKappa
    same //= nKappa
    ik = same % nK
    same //= nK
    it = same % nT
    return [it, ik, ikappa, itheta, isigma]

def print_excerpt(method, idx_fn, n=64):
    print(f"\n{method} (first {n} threads):")
    prev = None
    for idx in range(n):
        params = idx_fn(idx, nT, nK, nKappa, nTheta, nSigma)
        out = f"idx={idx:2d} | "
        if prev is not None:
            for i, (name, val, pval) in enumerate(zip(names, params, prev)):
                if val != pval:
                    out += f"{COLORS[i]}{name}={val}{RESET} "
                else:
                    out += f"{name}={val} "
        else:
            out += " ".join(f"{name}={val}" for name, val in zip(names, params))
        print(out)
        prev = params

if __name__ == "__main__":
    print_excerpt("Serpentine Indexing", serpentine_idx)
    print_excerpt("OU Row-Major Indexing", ou_rowmajor_idx)