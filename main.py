from flask import Flask, request, jsonify
import math
import numpy as np
from scipy.signal import find_peaks

app = Flask(__name__)

# ============================================================
# CONFIG
# ============================================================
ZEROS_FILE = "zerosdata.txt"
MAX_ZEROS = 10_000
ZERO_COUNT = 10_000

X_MIN = 2
X_MAX = 500
NUM_X = 50000

BATCH_SIZE = 2000

PEAK_PROM_FACTOR = 0.02
PEAK_HEIGHT_FACTOR = 0.01
MIN_PEAK_DISTANCE_POINTS = 6
CLUSTER_X_GAP = 1.5

_gammas_cache = None


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


def get_gammas() -> np.ndarray:
    global _gammas_cache
    if _gammas_cache is None:
        _gammas_cache = load_zero_ordinates(ZEROS_FILE, max_zeros=MAX_ZEROS)
    if ZERO_COUNT > len(_gammas_cache):
        raise ValueError("ZERO_COUNT exceeds loaded zero count.")
    return _gammas_cache[:ZERO_COUNT]


# ============================================================
# PRIME UTILITIES
# ============================================================
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
            out += np.sum((-2.0 / chunk)[:, None] * np.sin(phases), axis=0)
        else:
            denom = beta * beta + chunk * chunk
            a = -2.0 * beta / denom
            b = -2.0 * chunk / denom
            out += np.sum(a[:, None] * np.cos(phases) + b[:, None] * np.sin(phases), axis=0)

    out *= np.exp((beta - 0.5) * u_vals)
    return out


# ============================================================
# PRIME STAIRCASE RECOVERY
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

    return sorted(recovered_primes), sorted(recovered_prime_powers)


def compute_recovered_primes(beta: float):
    gammas = get_gammas()

    x_vals = np.linspace(X_MIN, X_MAX, NUM_X, dtype=np.float64)
    u_vals = np.log(x_vals)

    Z_vals = reconstruct_Z_beta_u(u_vals, gammas, beta, batch_size=BATCH_SIZE)
    psi_rec = x_vals + Z_vals
    recovered_primes, recovered_prime_powers = recover_prime_bases_from_psi_rec(x_vals, psi_rec, X_MAX)

    return {
        "beta": beta,
        "zero_count": int(len(gammas)),
        "x_min": X_MIN,
        "x_max": X_MAX,
        "recovered_prime_count": len(recovered_primes),
        "recovered_prime_power_count": len(recovered_prime_powers),
        "recovered_primes": recovered_primes,
        "recovered_prime_powers": recovered_prime_powers,
    }


# ============================================================
# ROUTES
# ============================================================
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Prime recovery API is running",
        "usage": "/api/recover?beta=0.5"
    })


@app.route("/api/recover", methods=["GET"])
def recover():
    beta_str = request.args.get("beta")

    if beta_str is None:
        return jsonify({
            "error": "beta is required",
            "message": "Provide beta to proceed, e.g. /api/recover?beta=0.5"
        }), 400

    try:
        beta = float(beta_str)
    except ValueError:
        return jsonify({
            "error": "beta must be numeric",
            "received": beta_str
        }), 400

    try:
        result = compute_recovered_primes(beta)
        return jsonify(result)
    except Exception as e:
        return jsonify({
            "error": "computation failed",
            "details": str(e)
        }), 500


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)