"""
Lab 03: Real-World Data Linear Regression
Numerical Methods - Section 3H
Author: Cabudsan, Daphne Anne
Date: September 2, 2026

This script performs linear regression on CO₂ vs Temperature data
using the least squares formulas.
"""

import sys
import numpy as np
import matplotlib.pyplot as plt

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

# ============================================================
# 1. DATA ENTRY
# ============================================================

# Independent variable: CO₂ concentration (ppm)
x = np.array([316.9, 320.0, 325.7, 331.1, 338.7, 346.0, 354.4, 
              360.9, 369.5, 379.7, 389.9, 400.8, 408.5, 414.7, 424.6])

# Dependent variable: Temperature anomaly (°C)
y = np.array([0.01, -0.08, 0.02, -0.06, 0.21, 0.11, 0.38, 
              0.45, 0.42, 0.62, 0.72, 0.86, 0.97, 0.98, 1.21])

# Verify data size
n = len(x)
print(f"Number of observations: {n}\n")

# ============================================================
# 2. LEAST SQUARES COMPUTATION (Using Formulas)
# ============================================================

# Compute sums needed for least squares
sum_x = np.sum(x)
sum_y = np.sum(y)
sum_xy = np.sum(x * y)
sum_x2 = np.sum(x ** 2)
sum_y2 = np.sum(y ** 2)

# Calculate the slope (a1) using the formula:
# a1 = (n * Σxy - Σx * Σy) / (n * Σx² - (Σx)²)
a1 = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x**2)

# Calculate the intercept (a0) using the formula:
# a0 = (Σy - a1 * Σx) / n
a0 = (sum_y - a1 * sum_x) / n

print("=" * 50)
print("REGRESSION RESULTS")
print("=" * 50)
print(f"Slope (a1)      = {a1:.6f} °C per ppm")
print(f"Intercept (a0)  = {a0:.6f} °C")
print()

# ============================================================
# 3. GOODNESS OF FIT
# ============================================================

# Calculate predicted y values
y_pred = a0 + a1 * x

# Calculate residuals
residuals = y - y_pred

# Sum of squared errors (SSE or Sr)
SSE = np.sum(residuals ** 2)

# Total sum of squares (SSt)
y_mean = np.mean(y)
SSt = np.sum((y - y_mean) ** 2)

# Coefficient of determination (r²)
r_squared = 1 - (SSE / SSt)

# Standard error of the estimate (sy/x)
syx = np.sqrt(SSE / (n - 2))

print("=" * 50)
print("GOODNESS OF FIT")
print("=" * 50)
print(f"S_r (SSE)        = {SSE:.6f} °C²")
print(f"r²               = {r_squared:.6f}")
print(f"Standard error   = {syx:.6f} °C")
print()

# ============================================================
# 4. PREDICTION
# ============================================================

# Predict temperature for CO₂ = 450 ppm
x_pred = 450.0
y_pred_new = a0 + a1 * x_pred

print("=" * 50)
print("PREDICTION")
print("=" * 50)
print(f"Predicted temperature at {x_pred:.1f} ppm: {y_pred_new:.3f} °C")
print(f"Calculation: {a0:.6f} + {a1:.6f} × {x_pred:.1f} = {y_pred_new:.3f} °C")
print()

# ============================================================
# 5. VISUALIZATION
# ============================================================

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# --- Plot 1: Data with fitted line ---
ax1.scatter(x, y, color='blue', s=80, label='Data points', zorder=5)
ax1.plot(x, y_pred, color='red', linewidth=2, label=f'Fit: y = {a0:.3f} + {a1:.3f}x')
ax1.set_xlabel('CO₂ Concentration (ppm)', fontsize=12)
ax1.set_ylabel('Temperature Anomaly (°C)', fontsize=12)
ax1.set_title('Global Temperature vs. CO₂ Concentration', fontsize=14)
ax1.grid(True, alpha=0.3)
ax1.legend(loc='upper left', fontsize=11)

# Add text box with regression info
textstr = f'$r^2$ = {r_squared:.4f}\n$s_{{y/x}}$ = {syx:.4f} °C'
props = dict(boxstyle='round', facecolor='white', alpha=0.8)
ax1.text(0.05, 0.95, textstr, transform=ax1.transAxes, fontsize=11,
         verticalalignment='top', bbox=props)

# --- Plot 2: Residual plot ---
ax2.scatter(x, residuals, color='purple', s=60, zorder=5)
ax2.axhline(y=0, color='black', linestyle='--', linewidth=1)
ax2.set_xlabel('CO₂ Concentration (ppm)', fontsize=12)
ax2.set_ylabel('Residuals (°C)', fontsize=12)
ax2.set_title('Residual Plot', fontsize=14)
ax2.grid(True, alpha=0.3)

# Add horizontal line at zero
ax2.set_ylim([-0.3, 0.3])

plt.tight_layout()
plt.savefig('regression_results.png', dpi=300, bbox_inches='tight')
print("Plot saved to 'regression_results.png'")
try:
    if sys.stdout.isatty():
        plt.show()
except Exception:
    pass

# ============================================================
# 6. SUMMARY
# ============================================================

print("=" * 50)
print("SUMMARY")
print("=" * 50)
print(f"Regression equation: y = {a0:.6f} + {a1:.6f}x")
print(f"Where:")
print(f"  y = Temperature Anomaly (°C)")
print(f"  x = CO₂ Concentration (ppm)")
print(f"\nInterpretation:")
print(f"  Slope: For every 1 ppm increase in CO₂, temperature")
print(f"         increases by {a1:.4f}°C.")
print(f"  Intercept: The estimated temperature when CO₂ = 0 ppm")
print(f"             is {a0:.3f}°C (not physically meaningful).")