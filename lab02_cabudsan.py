# lab02_cabudsan.py
# Numerical Methods - Laboratory Activity 02
# Student: Cabudsan
# Section: BES6-M
 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.integrate import quad
from scipy.signal import savgol_filter
from scipy import stats
import os
import base64
from io import BytesIO
from datetime import datetime
 
# ============================================================
# 1. LOAD DATA
# ============================================================
 
# Load the Excel file
file_path = os.path.join('..', 'reservoir-stage-15min.xlsx')
df = pd.read_excel(file_path, header=3, usecols=[0,4])
df.columns = ['Reading', 'stage']
df = df.dropna(subset=['stage']).copy()
df['Reading'] = pd.to_numeric(df['Reading'], errors='coerce')
df = df.dropna(subset=['Reading']).copy()
df['Reading'] = df['Reading'].astype(int)

# Build datetime from the first date (2026-07-21 00:00) + reading-based offset
start = pd.Timestamp('2026-07-21 00:00:00')
df['datetime'] = start + pd.to_timedelta((df['Reading'] - 1) * 15, unit='min')

# Convert datetime to hours elapsed
df = df.sort_values('datetime').reset_index(drop=True)
df['hours'] = (df['datetime'] - df['datetime'].iloc[0]).dt.total_seconds() / 3600
t = df['hours'].values
h = df['stage'].values
 
n = len(t)
dt = 0.25  # 15 minutes in hours
 
print(f"Loaded {n} readings from {df['datetime'].iloc[0]} to {df['datetime'].iloc[-1]}")
 
# ============================================================
# 2. DEFINE MODEL - Sum of Two Logistics
# ============================================================
 
def model(t, c, a1, k1, t1, a2, k2, t2):
    """
    h(t) = c + a1/(1 + exp(-k1*(t - t1))) + a2/(1 + exp(-k2*(t - t2)))
    """
    S1 = 1 / (1 + np.exp(-k1 * (t - t1)))
    S2 = 1 / (1 + np.exp(-k2 * (t - t2)))
    return c + a1 * S1 + a2 * S2
 
# Initial guesses (read from the data)
c0 = 14.2      # baseline level
a1_0 = 7.0     # rise height
k1_0 = 0.5     # rise steepness
t1_0 = 30.0    # inflection point
a2_0 = -1.7    # recession amplitude (negative!)
k2_0 = 0.2     # recession steepness
t2_0 = 42.0    # recession inflection
 
p0 = [c0, a1_0, k1_0, t1_0, a2_0, k2_0, t2_0]
 
# ============================================================
# 3. FIT THE CURVE
# ============================================================
 
popt, pcov = curve_fit(model, t, h, p0=p0, method='lm', maxfev=20000)
 
# Extract parameters
c, a1, k1, t1, a2, k2, t2 = popt
 
# Fitted values
h_fit = model(t, *popt)
 
# Residuals
residuals = h - h_fit
 
# ============================================================
# 4. STATISTICS
# ============================================================
 
p = len(popt)
dof = n - p
sse = np.sum(residuals**2)
sst = np.sum((h - np.mean(h))**2)
r2 = 1 - sse/sst
s = np.sqrt(sse / dof)
 
# Parameter standard errors
param_se = np.sqrt(np.diag(pcov))
param_t = popt / param_se
param_p = 2 * (1 - stats.t.cdf(np.abs(param_t), dof))
 
# Sign runs in residuals
def count_runs(arr):
    runs = 1
    for i in range(1, len(arr)):
        if np.sign(arr[i]) != np.sign(arr[i-1]) and arr[i] != 0 and arr[i-1] != 0:
            runs += 1
    return runs
 
runs = count_runs(residuals)
runs_expected = (2*n - 1) / 3  # approximate for random data
 
# ============================================================
# 5. DERIVATIVES
# ============================================================
 
# Analytic first derivative
def d1_model(t, a1, k1, t1, a2, k2, t2):
    S1 = 1 / (1 + np.exp(-k1 * (t - t1)))
    S2 = 1 / (1 + np.exp(-k2 * (t - t2)))
    return a1 * k1 * S1 * (1 - S1) + a2 * k2 * S2 * (1 - S2)
 
# Analytic second derivative
def d2_model(t, a1, k1, t1, a2, k2, t2):
    S1 = 1 / (1 + np.exp(-k1 * (t - t1)))
    S2 = 1 / (1 + np.exp(-k2 * (t - t2)))
    return (a1 * k1**2 * S1 * (1 - S1) * (1 - 2*S1) + 
            a2 * k2**2 * S2 * (1 - S2) * (1 - 2*S2))
 
# Finite differences (central difference)
d1_fd = np.zeros_like(t)
d1_fd[0] = (h[1] - h[0]) / dt
d1_fd[-1] = (h[-1] - h[-2]) / dt
for i in range(1, n-1):
    d1_fd[i] = (h[i+1] - h[i-1]) / (2*dt)
 
# Second derivative (finite differences)
d2_fd = np.zeros_like(t)
for i in range(1, n-1):
    d2_fd[i] = (h[i+1] - 2*h[i] + h[i-1]) / (dt**2)
 
# Savitzky-Golay smoothing
d1_sg = savgol_filter(h, window_length=9, polyorder=3, deriv=1, delta=dt)
d2_sg = savgol_filter(h, window_length=9, polyorder=3, deriv=2, delta=dt)
 
# Smoothing with moving average (2.25 h window = 9 points)
def moving_average(x, window):
    return np.convolve(x, np.ones(window)/window, mode='same')
 
d2_ma = moving_average(d2_fd, 9)
 
# Peak rate from fitted curve
t_fine = np.linspace(t[0], t[-1], 4000)
d1_fine = d1_model(t_fine, *popt[1:])
peak_idx = np.argmax(d1_fine)
peak_rate = d1_fine[peak_idx]
peak_rate_t = t_fine[peak_idx]
 
# Peak from finite differences
peak_fd_idx = np.argmax(d1_fd)
peak_rate_fd = d1_fd[peak_fd_idx]
peak_rate_t_fd = t[peak_fd_idx]
 
# ============================================================
# 6. AREA UNDER CURVE
# ============================================================
 
def integrand(t):
    return model(t, *popt)
 
area, abserr = quad(integrand, t[0], t[-1])
trapz_area = np.trapezoid(h, t)
mean_level = area / (t[-1] - t[0])
 
# ============================================================
# 7. CREATE PLOTS
# ============================================================
 
def fig_to_base64(fig):
    """Convert matplotlib figure to base64 string for embedding in HTML"""
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return img_str
 
# Plot 1: Stage log with fitted curve
fig1, ax1 = plt.subplots(figsize=(12, 5))
ax1.plot(t, h, 'o', markersize=3, color='#6f747c', alpha=0.75, label='Raw data')
ax1.plot(t, h_fit, 'b-', linewidth=2.2, label='Fitted curve')
ax1.set_xlabel('hours elapsed')
ax1.set_ylabel('stage h (m)')
ax1.set_title('Reservoir stage with fitted curve')
ax1.grid(True, alpha=0.3)
ax1.legend()
img1 = fig_to_base64(fig1)
 
# Plot 2: Residuals
fig2, ax2 = plt.subplots(figsize=(12, 4))
ax2.stem(t, residuals, linefmt='gray', markerfmt='o', basefmt='k-')
ax2.axhline(y=0, color='r', linestyle='-', alpha=0.5)
ax2.axhline(y=0.01, color='orange', linestyle='--', alpha=0.3)
ax2.axhline(y=-0.01, color='orange', linestyle='--', alpha=0.3)
ax2.set_xlabel('hours elapsed')
ax2.set_ylabel('residual e (m)')
ax2.set_title(f'Residuals (Runs: {runs}, Expected: {runs_expected:.0f})')
ax2.grid(True, alpha=0.3)
img2 = fig_to_base64(fig2)
 
# Plot 3: First derivative comparison
fig3, ax3 = plt.subplots(figsize=(12, 5))
ax3.plot(t, d1_fd, 'gray', alpha=0.5, label='Finite differences')
ax3.plot(t, d1_sg, '--', color='#0f766e', alpha=0.7, label='Savitzky-Golay')
ax3.plot(t_fine, d1_fine, 'b-', linewidth=2, label='Analytic from fit')
ax3.axhline(y=0, color='k', linestyle='-', alpha=0.3)
ax3.axvline(x=peak_rate_t, color='r', linestyle='--', alpha=0.5)
ax3.set_xlabel('hours elapsed')
ax3.set_ylabel('dh/dt (m/h)')
ax3.set_title(f'First derivative - Peak rate: {peak_rate:.4f} m/h at t = {peak_rate_t:.2f} h')
ax3.grid(True, alpha=0.3)
ax3.legend()
img3 = fig_to_base64(fig3)
 
# Plot 4: Second derivative
fig4, ax4 = plt.subplots(figsize=(12, 5))
ax4.plot(t, d2_fd, 'gray', alpha=0.4, label='Raw second difference')
ax4.plot(t, d2_ma, 'orange', alpha=0.6, label='Moving average (2.25 h)')
ax4.plot(t, d2_sg, '--', color='#0f766e', alpha=0.7, label='Savitzky-Golay')
ax4.plot(t_fine, d2_model(t_fine, *popt[1:]), 'purple', linewidth=2, label='Analytic from fit')
ax4.axhline(y=0, color='k', linestyle='-', alpha=0.3)
ax4.set_xlabel('hours elapsed')
ax4.set_ylabel('d²h/dt² (m/h²)')
ax4.set_title('Second derivative')
ax4.grid(True, alpha=0.3)
ax4.legend()
img4 = fig_to_base64(fig4)
 
# ============================================================
# 8. CREATE PLOTS
# ============================================================
 
# Plot 1: Stage log with fitted curve
fig1, ax1 = plt.subplots(figsize=(12, 5))
ax1.plot(t, h, 'o', markersize=3, color='#6f747c', alpha=0.75, label='Raw data')
ax1.plot(t, h_fit, 'b-', linewidth=2.2, label='Fitted curve')
ax1.set_xlabel('hours elapsed')
ax1.set_ylabel('stage h (m)')
ax1.set_title('Reservoir stage with fitted curve')
ax1.grid(True, alpha=0.3)
ax1.legend()
img1 = fig_to_base64(fig1)
 
# Plot 2: Residuals
fig2, ax2 = plt.subplots(figsize=(12, 4))
ax2.stem(t, residuals, linefmt='gray', markerfmt='o', basefmt='k-')
ax2.axhline(y=0, color='r', linestyle='-', alpha=0.5)
ax2.axhline(y=0.01, color='orange', linestyle='--', alpha=0.3)
ax2.axhline(y=-0.01, color='orange', linestyle='--', alpha=0.3)
ax2.set_xlabel('hours elapsed')
ax2.set_ylabel('residual e (m)')
ax2.set_title(f'Residuals (Runs: {runs}, Expected: {runs_expected:.0f})')
ax2.grid(True, alpha=0.3)
img2 = fig_to_base64(fig2)
 
# Plot 3: First derivative
fig3, ax3 = plt.subplots(figsize=(12, 5))
ax3.plot(t, d1_fd, 'gray', alpha=0.5, label='Finite differences')
ax3.plot(t, d1_sg, '--', color='#0f766e', alpha=0.7, label='Savitzky-Golay')
ax3.plot(t_fine, d1_fine, 'b-', linewidth=2, label='Analytic from fit')
ax3.axhline(y=0, color='k', linestyle='-', alpha=0.3)
ax3.axvline(x=peak_rate_t, color='r', linestyle='--', alpha=0.5)
ax3.set_xlabel('hours elapsed')
ax3.set_ylabel('dh/dt (m/h)')
ax3.set_title(f'First derivative - Peak rate: {peak_rate:.4f} m/h at t = {peak_rate_t:.2f} h')
ax3.grid(True, alpha=0.3)
ax3.legend()
img3 = fig_to_base64(fig3)
 
# Plot 4: Second derivative
fig4, ax4 = plt.subplots(figsize=(12, 5))
ax4.plot(t, d2_fd, 'gray', alpha=0.4, label='Raw second difference')
ax4.plot(t, d2_ma, 'orange', alpha=0.6, label='Moving average (2.25 h)')
ax4.plot(t, d2_sg, '--', color='#0f766e', alpha=0.7, label='Savitzky-Golay')
ax4.plot(t_fine, d2_model(t_fine, *popt[1:]), 'purple', linewidth=2, label='Analytic from fit')
ax4.axhline(y=0, color='k', linestyle='-', alpha=0.3)
ax4.set_xlabel('hours elapsed')
ax4.set_ylabel('d²h/dt² (m/h²)')
ax4.set_title('Second derivative')
ax4.grid(True, alpha=0.3)
ax4.legend()
img4 = fig_to_base64(fig4)

# ============================================================
# 9. GENERATE HTML DASHBOARD - PART 1 (Head + CSS)
# ============================================================
 
html_content = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Lab 02 &middot; Fitting a curve to the dam &middot; Cabudsan</title>
    <style>
        :root{{
            --bg:#f6f5f1; --panel:#ffffff; --ink:#16181d; --dim:#6c7280; --line:#e2e0da;
            --accent:#1f5f8b; --accent2:#c2410c; --good:#166534; --bad:#9a2c2c;
        }}
        *{{box-sizing:border-box}}
        body{{margin:0;background:var(--bg);color:var(--ink);
            font:15px/1.62 "Iowan Old Style","Palatino Linotype",Palatino,Georgia,serif;}}
        .wrap{{max-width:1120px;margin:0 auto;padding:34px 26px 90px}}
        header.top{{border-bottom:2px solid var(--ink);padding-bottom:16px;margin-bottom:22px}}
        .eyebrow{{font:600 11px/1 ui-sans-serif,system-ui,sans-serif;letter-spacing:.16em;
            text-transform:uppercase;color:var(--accent2)}}
        h1{{font-size:31px;margin:10px 0 4px;letter-spacing:-.01em}}
        .sub{{color:var(--dim);font-size:15px;margin:0}}
        .facts{{display:flex;flex-wrap:wrap;gap:0 30px;margin-top:14px;
            font:12.5px/1.5 ui-sans-serif,system-ui,sans-serif;color:var(--dim)}}
        .facts b{{color:var(--ink);font-weight:600}}
        .panel{{background:var(--panel);border:1px solid var(--line);border-radius:5px;
            padding:20px 22px;margin-bottom:20px}}
        .panel > h2{{font-size:19px;margin:0 0 2px}}
        .panel > .lede{{color:var(--dim);margin:0 0 16px;font-size:14px}}
        .tabhint{{font:600 10.5px ui-sans-serif,system-ui,sans-serif;letter-spacing:.11em;
            text-transform:uppercase;color:var(--dim);margin:30px 0 7px}}
        nav.tabs{{display:flex;flex-wrap:wrap;gap:7px;margin:0;padding:0;
            border-bottom:2px solid var(--accent2)}}
        nav.tabs button{{appearance:none;background:#e9e6df;color:#43494f;
            border:1px solid #d3cfc6;border-bottom:none;border-radius:7px 7px 0 0;
            padding:12px 22px;cursor:pointer;position:relative;top:2px;
            font:600 14px ui-sans-serif,system-ui,sans-serif;
            transition:background .12s,color .12s,top .12s}}
        nav.tabs button:hover{{background:#f7f5f0;color:var(--ink)}}
        nav.tabs button[aria-selected="true"]{{background:var(--accent2);color:#fff;
            border-color:var(--accent2);top:0;padding-bottom:14px;cursor:default}}
        section.tab{{display:none;padding-top:20px}}
        section.tab.on{{display:block}}
        table{{width:100%;border-collapse:collapse;margin:10px 0 4px}}
        th,td{{text-align:left;padding:7px 10px;border-bottom:1px solid var(--line);vertical-align:top}}
        th{{font:600 11px ui-sans-serif,system-ui,sans-serif;letter-spacing:.09em;
            text-transform:uppercase;color:var(--dim);border-bottom:1px solid #cfccc4}}
        td.b{{font-weight:700}}
        td.u{{color:var(--dim);font-size:12px}}
        .dim{{color:var(--dim)}}
        .yes{{color:var(--good);font-weight:600}}
        .no{{color:var(--bad);font-weight:600}}
        .headline{{display:grid;grid-template-columns:repeat(auto-fit,minmax(158px,1fr));
            gap:0;border:1px solid var(--line);border-radius:5px;overflow:hidden;
            background:var(--panel);margin:18px 0}}
        .hl{{padding:13px 16px;border-right:1px solid var(--line)}}
        .hl:last-child{{border-right:0}}
        .hlv{{font:700 25px/1.1 "SFMono-Regular",Consolas,monospace;font-variant-numeric:tabular-nums;
            color:var(--accent)}}
        .hlu{{font-size:13px;font-weight:600;color:var(--dim)}}
        .hll{{font:11.5px/1.35 ui-sans-serif,system-ui,sans-serif;color:var(--dim);margin-top:4px}}
        .stats{{display:grid;grid-template-columns:repeat(auto-fit,minmax(178px,1fr));gap:12px;margin:16px 0}}
        .stat{{border:1px solid var(--line);border-radius:4px;padding:12px 14px;background:#fbfaf7}}
        .stat .k{{font:600 10.5px ui-sans-serif,system-ui,sans-serif;letter-spacing:.09em;
            text-transform:uppercase;color:var(--dim)}}
        .stat .v{{font:700 24px/1.2 "SFMono-Regular",Consolas,monospace;margin-top:3px;
            font-variant-numeric:tabular-nums}}
        .stat .n{{font-size:12px;color:var(--dim);margin-top:2px}}
        .eq{{background:#1c2026;color:#e8e6e1;border-radius:4px;padding:12px 15px;
            font-family:"SFMono-Regular",Consolas,monospace;font-size:13.5px;overflow-x:auto;margin:10px 0}}
        .mono{{font-family:"SFMono-Regular",Consolas,"Liberation Mono",Menlo,monospace;
            font-size:13.5px;font-variant-numeric:tabular-nums}}
        .plain{{border-left:3px solid #0f766e;background:#eef7f5;padding:11px 16px;
            margin:14px 0;font-size:14.5px}}
        .plain .plabel{{display:block;font-family:ui-sans-serif,system-ui,sans-serif;
            font-size:11.5px;letter-spacing:.07em;text-transform:uppercase;color:#0f766e;
            font-weight:700;margin-bottom:3px}}
        .plain p{{margin:0 0 7px}}
        .plain p:last-child{{margin-bottom:0}}
        .plain b{{color:#0b3b36}}
        img{{max-width:100%;height:auto;border:1px solid var(--line);border-radius:4px;margin:8px 0}}
        .two{{display:grid;grid-template-columns:1fr 1fr;gap:22px;margin-top:22px}}
        @media (max-width:820px){{.two{{grid-template-columns:1fr}}}}
        footer{{color:var(--dim);font-size:12.5px;border-top:1px solid var(--line);
            padding-top:14px;margin-top:26px}}
        .chosenrow{{background:#f2f7fb}}
        .mark{{fill:#b0342f;stroke:#fff;stroke-width:1.5}}
        .verdict{{border:1px solid #cfccc4;background:#fbfaf7;border-radius:5px;padding:16px 20px;
            margin-top:18px;font-size:15px}}
        .note{{border-left:3px solid var(--accent);background:#f2f7fb;padding:12px 16px;
            margin:16px 0;font-size:14px}}
        @media print{{body{{background:#fff}}nav.tabs,.tabhint{{display:none}}
            section.tab{{display:none}}
            section.tab.on{{display:block!important}}
            .panel{{border-color:#ccc}}}}
    </style>
</head>
<body>
<div class="wrap">
'''

# ============================================================
# 10. HTML TEMPLATE - PART 2 (Body + Footer + Save)
# ============================================================
 
html_body = f'''
<header class="top">
    <div class="eyebrow">Numerical Methods &middot; Laboratory Activity 02</div>
    <h1>Fitting a curve to the dam</h1>
    <p class="sub">Levenberg-Marquardt, residuals, and the area under the level</p>
    <div class="facts">
        <span>Student <b>Cabudsan</b></span>
        <span>Section <b>BES6-M</b></span>
        <span>Readings <b>{n}</b> at <b>0.25 h</b></span>
        <span>Window <b>{df['datetime'].iloc[0].strftime('%d %b %Y %H:%M')} &rarr; {df['datetime'].iloc[-1].strftime('%d %b %Y %H:%M')}</b></span>
        <span>Generated <b>{datetime.now().strftime('%d %B %Y, %H:%M')}</b></span>
    </div>
</header>
 
<div class="panel">
    <h2>Stage log</h2>
    <p class="lede">{n} readings at 0.25 h intervals.</p>
    <img src="data:image/png;base64,{img1}" alt="Stage log with fitted curve">
    <div class="headline">
        <div class="hl"><div class="hlv">{np.max(h):.2f}<span class="hlu"> m</span></div><div class="hll">Crest</div></div>
        <div class="hl"><div class="hlv">{peak_rate:.4f}<span class="hlu"> m/h</span></div><div class="hll">Peak rate (fitted)</div></div>
        <div class="hl"><div class="hlv">{r2:.6f}</div><div class="hll">R²</div></div>
        <div class="hl"><div class="hlv">{s:.4f}<span class="hlu"> m</span></div><div class="hll">s (typical miss)</div></div>
        <div class="hl"><div class="hlv">{area:.1f}<span class="hlu"> m·h</span></div><div class="hll">Area under curve</div></div>
    </div>
</div>
 
<p class="tabhint">Three panels &mdash; click a tab to switch</p>
<nav class="tabs" role="tablist">
    <button role="tab" aria-selected="true" data-tab="tab1">1 &middot; Derivatives</button>
    <button role="tab" aria-selected="false" data-tab="tab2">2 &middot; Fitted curve</button>
    <button role="tab" aria-selected="false" data-tab="tab3">3 &middot; Area under the curve</button>
</nav>
 
<!-- TAB 1 -->
<section class="tab on" id="tab1">
    <div class="panel">
        <h2>The derivative you already have</h2>
        <p class="lede">Recomputed from the raw log with central differences.</p>
        <img src="data:image/png;base64,{img3}" alt="First derivative">
        <img src="data:image/png;base64,{img4}" alt="Second derivative">
        <div class="stats">
            <div class="stat"><div class="k">Max dh/dt (fitted)</div><div class="v">{peak_rate:.4f}</div><div class="n">m/h at t = {peak_rate_t:.2f} h</div></div>
            <div class="stat"><div class="k">Max dh/dt (finite diff)</div><div class="v">{peak_rate_fd:.4f}</div><div class="n">m/h at t = {peak_rate_t_fd:.2f} h</div></div>
        </div>
        <div class="plain"><span class="plabel">In plain words</span>
        <p><b>A derivative is just a rate.</b> The first derivative, dh/dt, is how fast the water level is changing, in metres per hour. The second derivative, d²h/dt², is whether that rate is itself speeding up or slowing down.</p>
        <p>The peak rate from the fitted curve is <b>{peak_rate:.4f} m/h</b> at t = {peak_rate_t:.2f} h. The finite differences give <b>{peak_rate_fd:.4f} m/h</b> at t = {peak_rate_t_fd:.2f} h.</p></div>
    </div>
</section>
 
<!-- TAB 2 -->
<section class="tab" id="tab2">
    <div class="panel">
        <h2>The fit, and the arithmetic behind it</h2>
        <p class="lede">Chosen model: <b>Sum of two logistics</b>, fitted with <span class="mono">scipy.optimize.curve_fit(..., method="lm")</span>.</p>
        <div class="eq">h(t) = c + a1 / (1 + exp(-k1(t - t1))) + a2 / (1 + exp(-k2(t - t2)))</div>
        
        <h3>Why this model</h3>
        <p>A single sigmoid cannot come back down. Two logistics have exactly two limbs, with <span class="mono">a2</span> negative making the second one a recession.</p>
        
        <h3>Fitted parameters</h3>
        <table>
            <thead><tr><th>Parameter</th><th>Unit</th><th>Value</th><th>SE</th><th>t</th><th>p-value</th></tr></thead>
            <tbody>
                <tr><td class="mono b">c</td><td>m</td><td class="mono">{c:.4f}</td><td class="mono">{param_se[0]:.4f}</td><td class="mono">{param_t[0]:.2f}</td><td class="mono">{param_p[0]:.2e}</td></tr>
                <tr><td class="mono b">a1</td><td>m</td><td class="mono">{a1:.4f}</td><td class="mono">{param_se[1]:.4f}</td><td class="mono">{param_t[1]:.2f}</td><td class="mono">{param_p[1]:.2e}</td></tr>
                <tr><td class="mono b">k1</td><td>1/h</td><td class="mono">{k1:.4f}</td><td class="mono">{param_se[2]:.4f}</td><td class="mono">{param_t[2]:.2f}</td><td class="mono">{param_p[2]:.2e}</td></tr>
                <tr><td class="mono b">t1</td><td>h</td><td class="mono">{t1:.4f}</td><td class="mono">{param_se[3]:.4f}</td><td class="mono">{param_t[3]:.2f}</td><td class="mono">{param_p[3]:.2e}</td></tr>
                <tr><td class="mono b">a2</td><td>m</td><td class="mono">{a2:.4f}</td><td class="mono">{param_se[4]:.4f}</td><td class="mono">{param_t[4]:.2f}</td><td class="mono">{param_p[4]:.2e}</td></tr>
                <tr><td class="mono b">k2</td><td>1/h</td><td class="mono">{k2:.4f}</td><td class="mono">{param_se[5]:.4f}</td><td class="mono">{param_t[5]:.2f}</td><td class="mono">{param_p[5]:.2e}</td></tr>
                <tr><td class="mono b">t2</td><td>h</td><td class="mono">{t2:.4f}</td><td class="mono">{param_se[6]:.4f}</td><td class="mono">{param_t[6]:.2f}</td><td class="mono">{param_p[6]:.2e}</td></tr>
            </tbody>
        </table>
        
        <div class="stats">
            <div class="stat"><div class="k">SSE</div><div class="v">{sse:.4f}</div><div class="n">n = {n}, p = {p}</div></div>
            <div class="stat"><div class="k">SST</div><div class="v">{sst:.4f}</div><div class="n">m²</div></div>
            <div class="stat"><div class="k">R²</div><div class="v">{r2:.6f}</div><div class="n">coefficient of determination</div></div>
            <div class="stat"><div class="k">s</div><div class="v">{s:.4f}</div><div class="n">m, standard error</div></div>
        </div>
        
        <h3>Residuals</h3>
        <img src="data:image/png;base64,{img2}" alt="Residuals">
        
        <div class="plain"><span class="plabel">In plain words</span>
        <p><b>Residuals</b> are the misses: actual reading minus predicted value. Positive means the curve sat too low there, negative means too high.</p>
        <p><b>Sign runs:</b> The residuals switch sign only <b>{runs}</b> times. Under random noise, we'd expect about <b>{runs_expected:.0f}</b> switches. This pattern means the model is missing some real structure in the data.</p></div>
        
        <div class="verdict"><b>Verdict.</b> R² = {r2:.6f} and s = {s:.4f} m. Trust this fit for volumes and timing to about a decimetre. Do not trust it for the level at an instant to the centimetre.</div>
    </div>
</section>
 
<!-- TAB 3 -->
<section class="tab" id="tab3">
    <div class="panel">
        <h2>The area under the fitted level</h2>
        <p class="lede">The fitted function integrated with <span class="mono">scipy.integrate.quad</span> between the first and last logged times.</p>
        <div class="eq">A = ∫ ĥ(t) dt, from t = {t[0]:.2f} h to t = {t[-1]:.2f} h</div>
        
        <div class="stats">
            <div class="stat"><div class="k">Area (quad)</div><div class="v">{area:.4f}</div><div class="n">metre-hours</div></div>
            <div class="stat"><div class="k">Area (trapezoid)</div><div class="v">{trapz_area:.4f}</div><div class="n">metre-hours</div></div>
            <div class="stat"><div class="k">Gap</div><div class="v">{(area - trapz_area):.4f}</div><div class="n">m·h ({(area/trapz_area - 1)*100:.4f}%)</div></div>
            <div class="stat"><div class="k">Mean level</div><div class="v">{mean_level:.4f}</div><div class="n">m</div></div>
        </div>
        
        <div class="plain"><span class="plabel">In plain words</span>
        <p><b>Integrating</b> means measuring the area between the curve and the bottom axis. Here the curve is water level and the axis is time, so the area adds up "how high, for how long" across the whole three days.</p>
        <p>The area is <b>{area:.1f} metre-hours</b>. Divide by the 71.75 h record and you get an average level of <b>{mean_level:.4f} m</b>.</p>
        <p><b>It is not a volume of water.</b> The sensor measures depth at one point; it knows nothing about how wide the reservoir is at any height.</p></div>
        
        <div class="note"><b>Units</b><br>
        A = {area:.4f} m·h. Not a volume - that needs surface area as a function of stage, which a depth sensor does not measure.</div>
    </div>
</section>
 
<footer>
    Numerical Methods BES6-M &middot; Laboratory Activity 02 &middot; Cabudsan &middot;
    generated by <span class="mono">lab02_cabudsan.py</span> on {datetime.now().strftime('%d %B %Y, %H:%M')}
</footer>
 
<script>
document.querySelectorAll('nav.tabs button').forEach(function(b){{
    b.addEventListener('click', function(){{
        document.querySelectorAll('nav.tabs button').forEach(function(x){{ x.setAttribute('aria-selected','false'); }});
        document.querySelectorAll('section.tab').forEach(function(s){{ s.classList.remove('on'); }});
        b.setAttribute('aria-selected','true');
        document.getElementById(b.dataset.tab).classList.add('on');
    }});
}});
</script>
</div>
</body>
</html>'''
 
# ============================================================
# 11. SAVE OUTPUT
# ============================================================
 
# Combine and save
html_file = 'lab02_cabudsan.html'
with open(html_file, 'w', encoding='utf-8') as f:
    f.write(html_content + html_body)
 
print(f"\nHTML dashboard saved as: {html_file}")
print(f"\nSUMMARY:")
print(f"   R² = {r2:.6f}")
print(f"   s = {s:.4f} m")
print(f"   Peak rate = {peak_rate:.4f} m/h at t = {peak_rate_t:.2f} h")
print(f"   Area = {area:.4f} m·h")
print(f"   Mean level = {mean_level:.4f} m")
print(f"\nOpen {html_file} in your browser to view the dashboard.")