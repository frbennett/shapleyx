"""SAC-SMA QMC test — compares QMC vs IID pick-freeze on RS-HDMR surrogate.

Uses the St Helens catchment data with 14 SAC-SMA parameters.
"""
import numpy as np
import pandas as pd
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "shapleyx"))

from shapleyx import rshdmr
from shapleyx.utilities.mc_shapley import (
    GaussianCopulaUniform, MCShapley
)

# ============================================================
# Load data and model
# ============================================================
sac_dir = Path(__file__).parent / "shapleyx" / "docs" / "tutorials" / "sac_sma_data"

forcing = pd.read_csv(sac_dir / "st_helens_forcing_original.csv",
                       parse_dates=['date'], dayfirst=True)
prcp = forcing['rainfall'].values
pet = forcing['pet'].values
q_obs = forcing['Q_CUMEC'].values

CATCHMENT_AREA_M2 = 120_500_000.0
WARMUP_DAYS = 365

def flow_mm_to_cumec(flow_mm):
    return flow_mm * CATCHMENT_AREA_M2 / 1000.0 / 86400.0

params_df = pd.read_csv(sac_dir / "es_parameters.csv")
param_names = params_df['parameter'].tolist()
lows = params_df['lower'].values
highs = params_df['upper'].values
d = len(param_names)

print(f"SAC-SMA: {d} parameters, {len(prcp)} daily timesteps, {WARMUP_DAYS}d warmup")

# Load numba-accelerated model
import importlib.util
spec = importlib.util.spec_from_file_location("sacramento", sac_dir / "sacramento.py")
sac = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sac)

def nse_from_params(params):
    flow_mm = sac.sacsma(prcp, pet, params)
    sim = flow_mm_to_cumec(flow_mm[WARMUP_DAYS:])
    obs = q_obs[WARMUP_DAYS:]
    residual = obs - sim
    num = np.sum(residual ** 2)
    den = np.sum((obs - np.mean(obs)) ** 2)
    return 1.0 - num / den if den > 0 else -np.inf

# Warm-up JIT compilation
_ = nse_from_params(np.array([44.1, 52.8, 80.9, 112.7, 52.8, 0.084, 0.49,
                                0.0148, 0.133, 35.3, 2.53, 0.0058, 0.424, 0.0093]))
print("Numba JIT warmup done.")

# ============================================================
# Generate training data
# ============================================================
N_TRAIN = 1000
print(f"\nGenerating {N_TRAIN} training samples...")
np.random.seed(42)

from scipy.stats.qmc import Sobol
sampler = Sobol(d, scramble=True, seed=42)
U_train = sampler.random(N_TRAIN)
X_train = lows + (highs - lows) * U_train

t0 = time.time()
Y_train = np.zeros(N_TRAIN)
for i in range(N_TRAIN):
    Y_train[i] = nse_from_params(X_train[i])
    if (i + 1) % 250 == 0:
        print(f"  {i+1}/{N_TRAIN} ({time.time()-t0:.0f}s)")
dt_gen = time.time() - t0

valid = Y_train > -np.inf
print(f"Generated in {dt_gen:.0f}s. Valid: {valid.sum()}/{N_TRAIN}")
print(f"NSE: mean={Y_train[valid].mean():.4f}, std={Y_train[valid].std():.4f}")

# ============================================================
# Train RS-HDMR
# ============================================================
print("\nTraining RS-HDMR...")
df = pd.DataFrame(X_train, columns=param_names)
df['Y'] = Y_train

t0 = time.time()
analyzer = rshdmr(data_file=df, polys=[4], method='omp_cv',
                  resampling=False, verbose=False)
ind_sobol, ind_shapley, _ = analyzer.run_all()
dt_rs = time.time() - t0

# Extract arrays from DataFrames
anal_sobol_arr = ind_sobol['index'].values.astype(float)
anal_sh_arr = ind_shapley['scaled effect'].values.astype(float)
print(f"R² = {analyzer.evs:.4f} (explained variance), {len(analyzer.non_zero_coefficients)} terms, {dt_rs:.0f}s")

# ============================================================
# QMC vs IID pick-freeze on surrogate
# ============================================================
print("\n=== QMC vs IID pick-freeze on RS-HDMR surrogate ===")

# Identity correlation (can swap for actual correlation matrix)
corr = np.eye(d)
joint = GaussianCopulaUniform(lows, highs, corr)

def surrogate_batch(X):
    return analyzer.predict(X)

def surrogate_1d(x):
    return float(analyzer.predict(x.reshape(1, -1))[0])

mc = MCShapley(surrogate_1d, joint, predict_batch=surrogate_batch)

# Reference: IID at high N (for correlated case we'd need a non-identity corr)
t0 = time.time()
df_ref = mc.compute(N=5000, method='exhaustive', random_state=1, k_max=2)
ref_sh = df_ref['effect'].values
print(f"Reference (IID N=5000): Shapley sum={ref_sh.sum():.4f}, TV={df_ref['total_variance'].iloc[0]:.6f}  ({time.time()-t0:.0f}s)")

# Also get analytical from coefficients (independent case)
anal_sh_arr_use = anal_sh_arr
print(f"Analytical (independent): Shapley sum={anal_sh_arr_use.sum():.4f}")

print(f"\n{'N':>6s}  {'QMC err':>10s}  {'IID err':>10s}  {'Ratio':>8s}  {'QMC t':>7s}  {'IID t':>7s}")
print("-" * 65)

for N in [256, 512, 1024, 2048]:
    t0 = time.time()
    df_qmc = mc.compute(N=N, method='qmc_exhaustive', random_state=42, k_max=2)
    dt_qmc = time.time() - t0
    rmse_qmc = np.sqrt(np.mean((df_qmc['effect'].values - anal_sh_arr_use)**2))
    
    t0 = time.time()
    df_iid = mc.compute(N=N, method='exhaustive', random_state=42, k_max=2)
    dt_iid = time.time() - t0
    rmse_iid = np.sqrt(np.mean((df_iid['effect'].values - anal_sh_arr_use)**2))
    
    ratio = rmse_iid / rmse_qmc if rmse_qmc > 0 else float('inf')
    print(f"  {N:4d}  {rmse_qmc:10.5f}  {rmse_iid:10.5f}  {ratio:6.1f}x  {dt_qmc:5.1f}s  {dt_iid:5.1f}s")

# Top parameters
order_qmc = np.argsort(-df_qmc['effect'].values)
order_anal = np.argsort(-anal_sh_arr_use)
top_n = 6
print(f"\nTop {top_n} parameters (QMC vs Analytical):")
for rank in range(top_n):
    i_q = order_qmc[rank]
    i_a = order_anal[rank]
    print(f"  {rank+1}. QMC: {param_names[i_q]:>8s} = {df_qmc['effect'].values[i_q]:.4f}  |  "
          f"Anal: {param_names[i_a]:>8s} = {anal_sh_arr_use[i_a]:.4f}")

print("\nDone.")
