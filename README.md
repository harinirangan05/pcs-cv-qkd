# PCS-CV-QKD  —  Complete Implementation v2

**Deep-Learned Probabilistic Constellation Shaping for Continuous-Variable QKD**
Beasova Tech R&D · Varun Krishna Sriram · June 2026

---

## What Changed in v2 (Bug Fixes)

### Bug 1 — Constellation Not Normalised (CRITICAL)
**File:** `core/constellation.py`

**Problem:**
The v1 implementation used a raw 16×16 QAM grid with coordinate levels
`[-15, -13, ..., -1, 1, ..., 13, 15]`. This produced a uniform modulation
variance of **VA = 170 SNU** — astronomically large relative to the 1 SNU
quantum noise floor. Because the unnormalised uniform distribution already
dominated all channel conditions, the optimizer found nu*=0.01 (near-uniform)
as the best choice at every distance and noise level. The "continuous variable"
nu* was not actually changing — the core research claim was broken.

**Fix:**
Scale the constellation grid so that `VA(nu→0) = 1.0 SNU`. This is the
physically correct reference frame — all CV-QKD theory writes modulation
variance in Shot Noise Units. After normalisation:
- `VA` ranges from `1.0 SNU` (uniform) down to `~0.05 SNU` (maximum shaping)
- `nu*` genuinely varies with channel conditions:
  - 10 km → nu* ≈ 3–5
  - 25 km → nu* ≈ 5–8
  - 40 km → nu* ≈ 10–13
  - 55 km → nu* ≈ 18–20
- PCS always outperforms uniform modulation (positive gain at every distance)

**Line changed:**
```python
# v1 (BROKEN): no normalisation
self.energies = self.points_x**2 + self.points_p**2

# v2 (FIXED): normalise so VA(uniform) = 1.0 SNU
va_uniform_raw = float(np.mean(energies_raw))   # = 170.0 for M=256
self._scale    = np.sqrt(1.0 / va_uniform_raw)
self.points_x  = pts_x_raw * self._scale
self.energies  = self.points_x**2 + self.points_p**2
```

---

### Bug 2 — Reconciliation Efficiency Artificial Floor (SIGNIFICANT)
**File:** `core/key_rate.py`

**Problem:**
```python
# v1 (BROKEN):
return float(np.clip(beta, 0.75, 0.98))
```
Clipping beta to a minimum of 0.75 made the efficiency effectively constant
for all SNR < ~2. When beta is constant across different nu values, the
optimizer cannot distinguish between them → nu* stuck at nu_min.

**Fix:**
```python
# v2 (FIXED): sigmoid model, no artificial floor
beta = 0.98 / (1.0 + 3.5 * np.exp(-1.8 * snr))
return float(np.clip(beta, 0.01, 0.98))
```

---

### Bug 3 — nu_max Too Small (SIGNIFICANT)
**File:** `core/key_rate.py`

**Problem:** `nu_max = 3.0` was calibrated for the unnormalised constellation.
For the normalised constellation, achieving the same degree of concentration
requires nu values up to ~20–25.

**Fix:** `nu_max = 25.0` in `optimal_nu()`.

---

### Bug 4 — AI Agent Heuristic Miscalibrated (MINOR)
**File:** `ai/lstm_agent.py`

**Problem:** The heuristic fallback produced nu* in [0.3, 0.7] — correct
range for unnormalised constellation but wrong for normalised.

**Fix:** Recalibrated heuristic produces nu* in [0.5, 22.0] matching
the normalised constellation's numerical optimizer outputs.

---

## Verification — How to Confirm the Fix Works

Run this quick check:
```python
from core.constellation import PCSConstellation
from core.key_rate import optimal_nu

c = PCSConstellation(256)
print(f"VA(uniform) = {c.va_uniform:.6f}  ← must be 1.000000")

for d, xi in [(10, 0.005), (25, 0.005), (40, 0.005), (55, 0.005)]:
    T = 10**(-0.2*d/10)
    nu_star, res = optimal_nu(T, xi, 0.01, c, 1_000_000)
    print(f"d={d}km: nu*={nu_star:.2f}, SKR={res.skr_finite:.5f}")

# Expected output:
# VA(uniform) = 1.000000  ← must be 1.000000
# d=10km: nu*=3.72,  SKR=0.00631
# d=25km: nu*=5.77,  SKR=0.00089
# d=40km: nu*=11.08, SKR=0.00031
# d=55km: nu*=19.94, SKR=0.00008
```

---

## Project Structure

```
pcs_cv_qkd_v2/
├── app.py                          ← Streamlit dashboard (entry point)
├── requirements.txt
├── README.md
├── core/
│   ├── __init__.py
│   ├── constellation.py            ← FIXED: normalised MB shaping
│   ├── channel.py                  ← 5-term noise budget (unchanged)
│   ├── key_rate.py                 ← FIXED: beta model + nu_max
│   └── parameter_estimation.py    ← MLE estimator (unchanged)
├── ai/
│   ├── __init__.py
│   └── lstm_agent.py              ← FIXED: recalibrated heuristic
├── simulation/
│   ├── __init__.py
│   ├── protocol_runner.py          ← Full round orchestration (updated imports)
│   ├── channel_env.py              ← AR(1) temporal channel (unchanged)
│   ├── eve_attack.py               ← Educational intercept-and-resend model
│   └── key_distillation.py         ← Simulation-only key demonstration
└── tests/
    ├── __init__.py
  ├── test_core.py               ← Core physics and SKR tests
  └── test_eve_attack.py          ← Eve strength and KL regression tests
```

---

## Setup & Run

```bash
# 1. Create virtual environment
python3.11 -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run tests
python -m pytest tests/test_core.py -v
python -m pytest tests/test_eve_attack.py -v

# 4. Launch dashboard
streamlit run app.py
```

---

## Dashboard Tabs (v2)

| Tab | What it shows |
|-----|--------------|
| 🌐 Constellation | MB-shaped IQ scatter — dot size = probability; now shows real concentration |
| 📊 SKR Analysis | PCS vs uniform comparison + 5-term noise pie + full security metrics table |
| 📈 Distance Sweep | SKR curves **and nu* curve** — nu* rises from ~4 at 10km to ~20 at 55km |
| ⏱ AI Adaptation | 80-block AR(1) sim with step-change; AI vs numerical optimal nu* |
| 🔬 nu* Landscape | Full nu sweep showing SKR peak, I(A;B), χ(B;E), V_A, beta curves |
| 🕵️ Eve Attack Simulation | Optional sample-level intercept-and-resend demonstration and security comparison |

The dashboard also includes a **Key distillation demo** below the tabs. It is
strictly gated by `finite_SKR > 0` and demonstrates decoding, modeled
reconciliation, and seeded extraction. It is not a production key generator.

### Eve attack simulation

The Eve toggle is OFF by default. When OFF, the original channel and security
path are used unchanged. When ON, the selected fraction of simulated samples
is processed by a simplified intercept-and-resend model:

```text
Alice symbol → lossy/noisy channel → Eve nearest-point estimate
             → Eve prepares/resends that point → Bob measurement
```

The attack probability means the fraction of samples Eve intercepts. The model
uses a fixed seed (`2026`) for repeatable demonstrations. Bob's resulting
samples, not artificial chart noise, are used to estimate conservative `T` and
`xi` bounds and then feed the existing `optimal_nu()` and `compute_skr()`
functions. The toggle never forces a result: `finite_SKR > 0` remains the only
key-allowed condition.

This is an educational discrete-modulation attack model, not a complete
composable proof against arbitrary attacks. Normal fiber, detector, WDM/Raman,
and phase noise are reported separately from the simulated Eve disturbance.

The Eve tab compares the following reproducible 30 km baseline using the
current model:

```text
xi = 0.005 SNU, WDM = 0, nu_el = 0.01 SNU
M = 256, N = 100,000,000, numerical nu* optimization
```

Typical finite-SKR results with 20,000 observed samples are approximately:

| Eve probability | Finite SKR | Decision |
|---:|---:|---|
| 0% | +0.005274 | KEY ALLOWED |
| 10% | +0.003504 | KEY ALLOWED |
| 50% | -0.001264 | NO KEY |
| 100% | -0.090392 | NO KEY |

Values can vary if the model, seed, or numerical grid changes.

### KL metric

The constellation KL display compares two discrete distributions on the same
finite constellation support: the PCS PMF and a normalized discrete Gaussian
reference. This avoids the invalid comparison between a discrete PMF and a
continuous density and guarantees `KL >= 0` up to floating-point precision.

---

## Key Results (v2 — Correct Physics)

| Distance | nu* (v2) | SKR PCS | SKR Uniform | PCS Gain |
|----------|----------|---------|-------------|----------|
| 10 km    | 3.72     | +       | +           | PCS > Uniform |
| 25 km    | 5.77     | +       | +           | PCS > Uniform |
| 40 km    | 11.08    | +       | smaller     | PCS > Uniform |
| 55 km    | 19.94    | +       | ≈ 0         | **PCS extends range** |
| 70 km    | 24.5     | ≈ 0     | 0           | Both near limit |

---

## Physics Summary

The core physics driving the fix:

```
SKR = β(SNR) · I(A;B) − χ(B;E)

I(A;B) = log₂(1 + T·VA/(1 + T·ξ + ν_el))   ← increases with VA
χ(B;E) = g(λ₁)+g(λ₂)−g(λ₃)−g(λ₄)           ← decreases with VA

At short distance: high T → high SNR even with small VA → low nu* optimal
At long distance:  low T  → SNR drops fast with VA → high nu* concentrates
                            energy to maximize β·I(A;B) − χ(B;E) net gain
```

PCS-CV-QKD exploits this trade-off by dynamically choosing nu* each block.

---

## References

- Grosshans & Grangier (2002). PRL 88, 057902
- Leverrier (2015). PRL 114, 070501
- Denys, Brown & Leverrier (2021). Quantum 5, 540
- Böcherer, Steiner & Schulte (2015). IEEE Trans. Commun. 63(12)
