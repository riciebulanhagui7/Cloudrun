import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks


# ============================================================
# COMPARE beta = 0 vs beta = 1/2
# ============================================================
# This script compares:
#
#   1. recovered prime list for beta = 0
#   2. recovered prime list for beta = 1/2
#   3. sup_x |pi_rec,0(x) - pi_rec,1/2(x)|
#
# It also prints:
#   - primes recovered by both
#   - primes only in beta=0
#   - primes only in beta=0.5
#   - metrics against the true prime set
#
# and plots:
#   - true pi(x), pi_rec,0(x), pi_rec,0.5(x)
#   - difference pi_rec,0(x) - pi_rec,0.5(x)
#   - raw Z(u) comparison
# ============================================================


# ============================================================
# CONFIG
# ============================================================
ZEROS_FILE = "zerosdata.txt"
MAX_ZEROS = 200_000
ZERO_COUNT = 200_000

BETAS = [0.0, 0.5]

X_MIN = 2
X_MAX = 5000
NUM_X = 50000

BATCH_SIZE = 2000
SHOW_LOG_X = False

# prime staircase extraction from reconstructed psi
PEAK_PROM_FACTOR = 0.02
PEAK_HEIGHT_FACTOR = 0.01
MIN_PEAK_DISTANCE_POINTS = 6
CLUSTER_X_GAP = 1.5

# plotting
TRUE_LINEWIDTH = 3.0
REC_LINEWIDTH = 1.5
TRUE_ALPHA = 0.95
REC_ALPHA = 0.85

SHOW_TRUE_PRIME_MARKERS = True
SHOW_RECOVERED_PRIME_MARKERS = True


# ============================================================
# LOAD ZEROS
# ============================================================
def load_zero_ordinates(filename: str, max_zeros: int | None = None) -> np.ndarray:
    gammas = []

    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue

            parts = s.replace(",", " ").split()
            try:
                gamma = float(parts[-1])
            except ValueError:
                continue

            if gamma > 0:
                gammas.append(gamma)
                if max_zeros is not None and len(gammas) >= max_zeros:
                    break

    if not gammas:
        raise ValueError(f"No valid positive zero ordinates found in {filename!r}")

    arr = np.array(gammas, dtype=np.float64)
    arr.sort()
    return arr


# ============================================================
# PRIME UTILITIES
# ============================================================
def primes_up_to(n: int) -> list[int]:
    if n < 2:
        return []

    sieve = np.ones(n + 1, dtype=bool)
    sieve[:2] = False
    limit = int(n**0.5) + 1

    for p in range(2, limit):
        if sieve[p]:
            sieve[p * p:n + 1:p] = False

    return np.flatnonzero(sieve).tolist()


def is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n in (2, 3):
        return True
    if n % 2 == 0:
        return False

    r = int(math.isqrt(n))
    for d in range(3, r + 1, 2):
        if n % d == 0:
            return False
    return True


def prime_power_decomposition(n: int):
    """
    Return (is_prime_power, base_prime, exponent)
    """
    if n < 2:
        return False, None, None

    if is_prime(n):
        return True, n, 1

    max_k = int(math.log2(n)) + 1
    for k in range(2, max_k + 1):
        a = round(n ** (1.0 / k))
        for cand in (a - 1, a, a + 1):
            if cand >= 2 and cand ** k == n and is_prime(cand):
                return True, cand, k

    return False, None, None


def true_psi_on_grid(x_vals: np.ndarray, primes: list[int]) -> np.ndarray:
    psi = np.zeros_like(x_vals, dtype=np.float64)

    for p in primes:
        logp = math.log(p)
        pk = p
        while pk <= x_vals[-1]:
            psi += (x_vals >= pk) * logp
            if pk > x_vals[-1] / p:
                break
            pk *= p

    return psi


def true_pi_on_grid(x_vals: np.ndarray, primes: list[int]) -> np.ndarray:
    parr = np.array(primes, dtype=np.int64)
    return np.searchsorted(parr, x_vals, side="right")


# ============================================================
# RECONSTRUCT Z_beta(u)
# ============================================================
def reconstruct_Z_beta_u(
    u_vals: np.ndarray,
    gammas: np.ndarray,
    beta: float,
    batch_size: int = 2000,
) -> np.ndarray:
    u_vals = np.asarray(u_vals, dtype=np.float64)
    out = np.zeros_like(u_vals)

    for start in range(0, len(gammas), batch_size):
        chunk = gammas[start:start + batch_size]
        phases = np.outer(chunk, u_vals)

        if beta == 0.0:
            # -2 * Re(e^{i gamma u} / (i gamma)) = -2 * sin(gamma u) / gamma
            out += np.sum((-2.0 / chunk)[:, None] * np.sin(phases), axis=0)
        else:
            denom = beta * beta + chunk * chunk
            a = -2.0 * beta / denom
            b = -2.0 * chunk / denom
            out += np.sum(a[:, None] * np.cos(phases) + b[:, None] * np.sin(phases), axis=0)

    out *= np.exp((beta - 0.5) * u_vals)
    return out


# ============================================================
# PRIME STAIRCASE RECOVERY FROM psi_rec
# ============================================================
def detect_peaks_from_signal(x_vals: np.ndarray, signal_vals: np.ndarray):
    max_sig = float(np.max(signal_vals))
    height = PEAK_HEIGHT_FACTOR * max_sig
    prominence = PEAK_PROM_FACTOR * max_sig

    peaks, props = find_peaks(
        signal_vals,
        height=height,
        prominence=prominence,
        distance=MIN_PEAK_DISTANCE_POINTS,
    )

    prominences = props.get("prominences", np.zeros(len(peaks)))
    out = []
    for i, idx in enumerate(peaks):
        out.append({
            "x": float(x_vals[idx]),
            "y": float(signal_vals[idx]),
            "prominence": float(prominences[i]),
        })

    out.sort(key=lambda d: d["x"])
    return out


def cluster_peaks(peak_info, x_gap=CLUSTER_X_GAP):
    if not peak_info:
        return []

    clusters = []
    current = [peak_info[0]]

    for pk in peak_info[1:]:
        if pk["x"] - current[-1]["x"] <= x_gap:
            current.append(pk)
        else:
            clusters.append(current)
            current = [pk]
    clusters.append(current)

    cluster_info = []
    for cl in clusters:
        strongest = max(cl, key=lambda p: p["prominence"])
        weights = np.array([p["prominence"] for p in cl], dtype=float)
        xs = np.array([p["x"] for p in cl], dtype=float)
        center_x = float(np.sum(weights * xs) / np.sum(weights)) if np.sum(weights) > 0 else float(np.mean(xs))

        cluster_info.append({
            "cluster_center_x": center_x,
            "strongest_peak_x": strongest["x"],
            "strongest_prominence": strongest["prominence"],
        })

    return cluster_info


def nearest_integer_candidate(a: float, b: float) -> int:
    ra = int(round(a))
    rb = int(round(b))
    da = abs(a - ra)
    db = abs(b - rb)
    return ra if da <= db else rb


def recover_prime_bases_from_psi_rec(x_vals: np.ndarray, psi_rec: np.ndarray, x_max: int):
    dpsi = np.gradient(psi_rec, x_vals)
    signal = np.abs(dpsi)

    peak_info = detect_peaks_from_signal(x_vals, signal)
    clusters = cluster_peaks(peak_info)

    recovered_primes = set()
    recovered_prime_powers = set()

    for cl in clusters:
        n = nearest_integer_candidate(cl["cluster_center_x"], cl["strongest_peak_x"])
        if 2 <= n <= x_max:
            ok, p, k = prime_power_decomposition(n)
            if ok:
                recovered_prime_powers.add(n)
                recovered_primes.add(p)

    return sorted(recovered_primes), sorted(recovered_prime_powers), signal, peak_info, clusters


def reconstructed_pi_on_grid(x_vals: np.ndarray, recovered_primes: list[int]) -> np.ndarray:
    parr = np.array(sorted(recovered_primes), dtype=np.int64)
    return np.searchsorted(parr, x_vals, side="right")


# ============================================================
# METRICS
# ============================================================
def set_metrics(recovered_items, true_items):
    rec_set = set(recovered_items)
    true_set = set(true_items)

    tp = len(rec_set & true_set)
    fp = len(rec_set - true_set)
    fn = len(true_set - rec_set)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    print("Loading zeros...")
    gammas_all = load_zero_ordinates(ZEROS_FILE, max_zeros=MAX_ZEROS)
    print(f"Loaded zeros: {len(gammas_all):,}")

    if ZERO_COUNT > len(gammas_all):
        raise ValueError("ZERO_COUNT exceeds loaded zero count.")

    gammas = gammas_all[:ZERO_COUNT]
    print(f"Using ZERO_COUNT = {ZERO_COUNT:,}")

    x_vals = np.linspace(X_MIN, X_MAX, NUM_X, dtype=np.float64)
    u_vals = np.log(x_vals)

    true_primes = primes_up_to(X_MAX)
    psi_true = true_psi_on_grid(x_vals, true_primes)
    pi_true = true_pi_on_grid(x_vals, true_primes)

    results = {}

    for beta in BETAS:
        print(f"Computing beta = {beta} ...")

        Z_vals = reconstruct_Z_beta_u(u_vals, gammas, beta, batch_size=BATCH_SIZE)
        psi_rec = x_vals + Z_vals
        recovered_primes, recovered_prime_powers, dpsi_signal, peak_info, clusters = \
            recover_prime_bases_from_psi_rec(x_vals, psi_rec, X_MAX)
        pi_rec = reconstructed_pi_on_grid(x_vals, recovered_primes)

        results[beta] = {
            "Z": Z_vals,
            "psi_rec": psi_rec,
            "pi_rec": pi_rec,
            "recovered_primes": recovered_primes,
            "recovered_prime_powers": recovered_prime_powers,
            "prime_metrics": set_metrics(recovered_primes, true_primes),
            "peak_info": peak_info,
            "clusters": clusters,
        }

    # --------------------------------------------------------
    # Compare recovered prime lists
    # --------------------------------------------------------
    primes_0 = set(results[0.0]["recovered_primes"])
    primes_05 = set(results[0.5]["recovered_primes"])

    common_primes = sorted(primes_0 & primes_05)
    only_0 = sorted(primes_0 - primes_05)
    only_05 = sorted(primes_05 - primes_0)

    # --------------------------------------------------------
    # sup_x |pi_rec,0 - pi_rec,0.5|
    # --------------------------------------------------------
    pi_rec_0 = results[0.0]["pi_rec"]
    pi_rec_05 = results[0.5]["pi_rec"]

    pi_rec_diff = pi_rec_0 - pi_rec_05
    sup_diff = int(np.max(np.abs(pi_rec_diff)))
    rms_diff = float(np.sqrt(np.mean(pi_rec_diff**2)))

    # --------------------------------------------------------
    # Print summary
    # --------------------------------------------------------
    print("\n" + "=" * 100)
    print("COMPARISON: beta = 0 vs beta = 0.5")
    print("=" * 100)

    for beta in BETAS:
        m = results[beta]["prime_metrics"]
        print(f"\nbeta = {beta}")
        print(f"Recovered primes       : {len(results[beta]['recovered_primes'])}")
        print(f"Recovered prime powers : {len(results[beta]['recovered_prime_powers'])}")
        print(f"TP                     : {m['tp']}")
        print(f"FP                     : {m['fp']}")
        print(f"FN                     : {m['fn']}")
        print(f"Precision              : {m['precision']:.6f}")
        print(f"Recall                 : {m['recall']:.6f}")
        print(f"F1                     : {m['f1']:.6f}")
        print(f"Recovered prime list   :")
        print(results[beta]["recovered_primes"])

    print("\n" + "-" * 100)
    print("Prime-list comparison")
    print("-" * 100)
    print(f"Recovered by BOTH           : {len(common_primes)}")
    print(common_primes)
    print(f"\nRecovered ONLY by beta=0    : {len(only_0)}")
    print(only_0)
    print(f"\nRecovered ONLY by beta=0.5  : {len(only_05)}")
    print(only_05)

    print("\n" + "-" * 100)
    print("Staircase comparison")
    print("-" * 100)
    print(f"sup_x |pi_rec,0(x) - pi_rec,0.5(x)| = {sup_diff}")
    print(f"RMS   |pi_rec,0(x) - pi_rec,0.5(x)| = {rms_diff:.6f}")

    # --------------------------------------------------------
    # Plot 1: true pi, pi_rec,0, pi_rec,0.5
    # --------------------------------------------------------
    plt.figure(figsize=(14, 7))

    plt.step(
        x_vals,
        pi_rec_0,
        where="post",
        color="tab:blue",
        linewidth=REC_LINEWIDTH,
        alpha=REC_ALPHA,
        label="π_rec(x), beta=0",
        zorder=2,
    )

    plt.step(
        x_vals,
        pi_rec_05,
        where="post",
        color="tab:orange",
        linewidth=REC_LINEWIDTH,
        alpha=REC_ALPHA,
        label="π_rec(x), beta=0.5",
        zorder=2,
    )

    if SHOW_RECOVERED_PRIME_MARKERS:
        for p in results[0.0]["recovered_primes"]:
            plt.axvline(p, ymin=0.0, ymax=0.04, color="tab:blue", linewidth=0.7, alpha=0.20, zorder=1)
        for p in results[0.5]["recovered_primes"]:
            plt.axvline(p, ymin=0.0, ymax=0.04, color="tab:orange", linewidth=0.7, alpha=0.20, zorder=1)

    if SHOW_TRUE_PRIME_MARKERS:
        for p in true_primes:
            plt.axvline(p, ymin=0.0, ymax=0.06, color="black", linewidth=0.5, alpha=0.10, zorder=1)

    plt.step(
        x_vals,
        pi_true,
        where="post",
        color="black",
        linewidth=TRUE_LINEWIDTH,
        alpha=TRUE_ALPHA,
        label="True π(x)",
        zorder=10,
    )

    if SHOW_LOG_X:
        plt.xscale("log")
    plt.xlabel("x")
    plt.ylabel("π(x)")
    plt.title(f"True prime staircase vs reconstructed staircases (N={ZERO_COUNT})")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # --------------------------------------------------------
    # Plot 2: pi_rec,0 - pi_rec,0.5
    # --------------------------------------------------------
    plt.figure(figsize=(14, 5))
    plt.step(
        x_vals,
        pi_rec_diff,
        where="post",
        linewidth=1.6,
        color="purple",
    )
    plt.axhline(0, color="black", linewidth=1.0, alpha=0.8)

    if SHOW_LOG_X:
        plt.xscale("log")
    plt.xlabel("x")
    plt.ylabel("π_rec,0(x) - π_rec,0.5(x)")
    plt.title("Difference between reconstructed prime staircases")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.show()

    # --------------------------------------------------------
    # Plot 3: raw Z(u) compare
    # --------------------------------------------------------
    plt.figure(figsize=(14, 6))
    plt.plot(x_vals, results[0.0]["Z"], label="Z(u), beta=0")
    plt.plot(x_vals, results[0.5]["Z"], label="Z(u), beta=0.5")
    plt.xlabel("x = exp(u)")
    plt.ylabel("Z_beta(u)")
    plt.title("Raw Z(u): beta = 0 vs beta = 0.5")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.show()