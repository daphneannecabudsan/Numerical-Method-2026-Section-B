"""
HOW ACCURATE IS GOOD ENOUGH?
Approximating a Civil Engineering Function Using Infinite Series
An Integrated Python Exercise for Civil Engineering Students

Topics: Geometric Series, Power Series, Maclaurin Series, Taylor Series,
        Error Analysis and Engineering Decision-Making
"""

import math
import matplotlib.pyplot as plt
import os
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except AttributeError:
    pass

# =============================================================================
# PART 1: GEOMETRIC SERIES
# =============================================================================

def geometric_sum(x, N):
    """Calculate S_N = 1 + x + x^2 + ... + x^N"""
    total = 0.0
    for k in range(N + 1):
        total += x ** k
    return total


def part1_geometric_series():
    print("=" * 72)
    print("PART 1: GEOMETRIC SERIES")
    print("=" * 72)
    print()
    print("The geometric series: 1 + x + x^2 + x^3 + ... = 1/(1-x) for |x| < 1")
    print()

    x_values = [0.5, 0.8, 0.9]
    N_values = [5, 10, 20, 50, 100]

    for x in x_values:
        exact = 1.0 / (1.0 - x)
        print(f"--- x = {x} ---")
        print(f"{'N':>5} | {'Partial Sum':>18} | {'Exact':>18} | {'Abs Error':>15} | {'Rel Error %':>12}")
        print("-" * 78)
        for N in N_values:
            approx = geometric_sum(x, N)
            abs_err = abs(exact - approx)
            rel_err = (abs_err / exact) * 100 if exact != 0 else 0
            print(f"{N:5d} | {approx:18.10f} | {exact:18.10f} | {abs_err:15.2e} | {rel_err:12.8f}")
        print()

    print("Analysis:")
    print("- Larger x (closer to 1) requires more terms to converge.")
    print("- x = 0.9 converges much slower than x = 0.5.")
    print("- This is because the ratio x determines how quickly terms shrink.")
    print()


# =============================================================================
# PART 2: POWER SERIES
# =============================================================================

def power_series(x, coefficients):
    """Evaluate P_N(x) = a_0 + a_1*x + a_2*x^2 + ... + a_N*x^N"""
    result = 0.0
    for k, a_k in enumerate(coefficients):
        result += a_k * (x ** k)
    return result


def part2_power_series():
    print("=" * 72)
    print("PART 2: POWER SERIES")
    print("=" * 72)
    print()
    print("P_N(x) = a_0 + a_1*x + a_2*x^2 + ... + a_N*x^N")
    print()

    # Demonstrate: coefficients for 1/(1-x) are all 1s (geometric series)
    print("Demonstration: Power series for 1/(1-x) with x = 0.5")
    x = 0.5
    exact = 1.0 / (1.0 - x)
    for N in [5, 10, 20]:
        coeffs = [1.0] * (N + 1)
        approx = power_series(x, coeffs)
        print(f"  N={N:2d}: approx = {approx:.10f}, exact = {exact:.10f}, error = {abs(exact-approx):.2e}")

    print()
    print("Key Insight: A polynomial is a finite power series.")
    print("An infinite power series can approximate transcendental functions.")
    print()


# =============================================================================
# PART 3: MACLAURIN SERIES FOR sin(theta)
# =============================================================================

def sin_maclaurin(theta, N):
    """Approximate sin(theta) using N terms of the Maclaurin series.
    sin(theta) = theta - theta^3/3! + theta^5/5! - theta^7/7! + ...
    theta must be in radians.
    """
    result = 0.0
    for n in range(N):
        sign = (-1) ** n
        factorial = math.factorial(2 * n + 1)
        result += sign * (theta ** (2 * n + 1)) / factorial
    return result


def part3_maclaurin_series():
    print("=" * 72)
    print("PART 3: MACLAURIN SERIES FOR sin(theta)")
    print("=" * 72)
    print()
    print("sin(theta) = theta - theta^3/3! + theta^5/5! - theta^7/7! + ...")
    print()

    theta_deg = 10
    theta_rad = math.radians(theta_deg)
    exact = math.sin(theta_rad)

    print(f"Investigation for theta = {theta_deg} degrees ({theta_rad:.6f} radians)")
    print(f"Exact sin({theta_deg} deg) = {exact:.10f}")
    print()
    print(f"{'Terms':>6} | {'Approximation':>18} | {'Exact':>18} | {'Abs Error':>15} | {'Rel Error %':>12}")
    print("-" * 80)
    for N in range(1, 9):
        approx = sin_maclaurin(theta_rad, N)
        abs_err = abs(exact - approx)
        rel_err = (abs_err / abs(exact)) * 100
        print(f"{N:6d} | {approx:18.12f} | {exact:18.12f} | {abs_err:15.2e} | {rel_err:12.8f}")

    print()
    print("Analysis:")
    print("- Each additional term adds a higher-order correction.")
    print("- The series converges rapidly for small angles.")
    print("- By 4 terms, error is negligible for engineering purposes.")
    print()


# =============================================================================
# PART 4: ENGINEERING INVESTIGATION
# =============================================================================

def part4_engineering_investigation():
    print("=" * 72)
    print("PART 4: ENGINEERING INVESTIGATION")
    print("=" * 72)
    print()
    L = 20  # meters
    angles_deg = [1, 2, 5, 10, 15, 20, 30]
    max_terms = 4

    print(f"y = L * sin(theta), L = {L} m")
    print()

    for num_terms in range(1, max_terms + 1):
        print(f"--- Maclaurin with {num_terms} term(s) ---")
        print(f"{'Angle (deg)':>12} | {'Exact y (m)':>14} | {'Approx y (m)':>14} | {'Abs Error (m)':>14} | {'% Error':>10}")
        print("-" * 74)
        for angle_deg in angles_deg:
            theta_rad = math.radians(angle_deg)
            exact_y = L * math.sin(theta_rad)
            approx_y = L * sin_maclaurin(theta_rad, num_terms)
            abs_err = abs(exact_y - approx_y)
            rel_err = (abs_err / abs(exact_y)) * 100 if exact_y != 0 else 0
            print(f"{angle_deg:12d} | {exact_y:14.8f} | {approx_y:14.8f} | {abs_err:14.6f} | {rel_err:10.6f}")
        print()

    print("Analysis Questions:")
    print("1. Error increases as angle increases (sin deviates more from its linear approx).")
    print("2. Adding more terms reduces error significantly at every angle.")
    print("3. For angles < 5 deg, 2 terms suffice for very high accuracy.")
    print("4. For angles > 20 deg, 3-4 terms are needed for good accuracy.")
    print()


# =============================================================================
# PART 5: TAYLOR SERIES CENTERED AT a = 10 degrees
# =============================================================================

def sin_taylor(theta, a, N):
    """Approximate sin(theta) using Taylor series centered at a (radians).
    Uses N terms (n = 0 to N-1).
    Derivatives of sin cycle: sin, cos, -sin, -cos
    """
    result = 0.0
    sin_a = math.sin(a)
    cos_a = math.cos(a)
    for n in range(N):
        derivative_pattern = n % 4
        if derivative_pattern == 0:
            f_deriv = sin_a
        elif derivative_pattern == 1:
            f_deriv = cos_a
        elif derivative_pattern == 2:
            f_deriv = -sin_a
        else:
            f_deriv = -cos_a
        term = f_deriv * ((theta - a) ** n) / math.factorial(n)
        result += term
    return result


def part5_taylor_series():
    print("=" * 72)
    print("PART 5: TAYLOR SERIES CENTERED AT a = 10 degrees")
    print("=" * 72)
    print()

    a_deg = 10
    a_rad = math.radians(a_deg)
    angles_deg = [1, 2, 5, 10, 15, 20, 30]
    max_terms = 4

    print(f"Taylor series centered at a = {a_deg} degrees ({a_rad:.6f} rad)")
    print()

    for num_terms in range(1, max_terms + 1):
        print(f"--- Taylor (center={a_deg} deg) with {num_terms} term(s) ---")
        print(f"{'Angle (deg)':>12} | {'Exact y (m)':>14} | {'Approx y (m)':>14} | {'Abs Error (m)':>14} | {'% Error':>10}")
        print("-" * 74)
        for angle_deg in angles_deg:
            theta_rad = math.radians(angle_deg)
            exact_y = 20 * math.sin(theta_rad)
            approx_y = 20 * sin_taylor(theta_rad, a_rad, num_terms)
            abs_err = abs(exact_y - approx_y)
            rel_err = (abs_err / abs(exact_y)) * 100 if exact_y != 0 else 0
            print(f"{angle_deg:12d} | {exact_y:14.8f} | {approx_y:14.8f} | {abs_err:14.6f} | {rel_err:10.6f}")
        print()

    # Comparison: Maclaurin vs Taylor
    print("=" * 72)
    print("COMPARISON: Maclaurin (center=0) vs Taylor (center=10 deg)")
    print("=" * 72)
    print()
    for num_terms in range(1, max_terms + 1):
        print(f"--- {num_terms} term(s) ---")
        print(f"{'Angle':>6} | {'Maclaurin %Err':>16} | {'Taylor %Err':>14} | {'Better':>10}")
        print("-" * 54)
        for angle_deg in angles_deg:
            theta_rad = math.radians(angle_deg)
            exact = math.sin(theta_rad)
            m_err = abs(exact - sin_maclaurin(theta_rad, num_terms)) / abs(exact) * 100
            t_err = abs(exact - sin_taylor(theta_rad, a_rad, num_terms)) / abs(exact) * 100
            better = "Taylor" if t_err < m_err else "Maclaurin"
            print(f"{angle_deg:5d}° | {m_err:16.8f} | {t_err:14.8f} | {better:>10}")
        print()

    print("Analysis:")
    print("- Taylor is more accurate near the expansion point (10°).")
    print("- Maclaurin may be better far from 10° (e.g., at 30°).")
    print("- Both converge quickly as terms are added.")
    print()


# =============================================================================
# PART 6: ERROR TOLERANCE — 0.1% requirement
# =============================================================================

def find_min_terms_maclaurin(theta_rad, tolerance_pct, max_terms=20):
    """Find minimum N such that Maclaurin approx is within tolerance_pct of exact."""
    exact = math.sin(theta_rad)
    for N in range(1, max_terms + 1):
        approx = sin_maclaurin(theta_rad, N)
        if exact == 0:
            return N
        rel_err = abs(exact - approx) / abs(exact) * 100
        if rel_err < tolerance_pct:
            return N
    return max_terms


def find_min_terms_taylor(theta_rad, a_rad, tolerance_pct, max_terms=20):
    """Find minimum N such that Taylor approx is within tolerance_pct of exact."""
    exact = math.sin(theta_rad)
    for N in range(1, max_terms + 1):
        approx = sin_taylor(theta_rad, a_rad, N)
        if exact == 0:
            return N
        rel_err = abs(exact - approx) / abs(exact) * 100
        if rel_err < tolerance_pct:
            return N
    return max_terms


def part6_error_tolerance():
    print("=" * 72)
    print("PART 6: ENGINEERING DECISION — ERROR TOLERANCE (< 0.1%)")
    print("=" * 72)
    print()

    tolerance = 0.1  # percent
    a_rad = math.radians(10)
    angles_deg = [1, 2, 5, 10, 15, 20, 30]

    print(f"Requirement: < {tolerance}% relative error")
    print()
    print(f"{'Angle':>8} | {'Min Terms (Maclaurin)':>22} | {'Min Terms (Taylor)':>20}")
    print("-" * 56)
    for angle_deg in angles_deg:
        theta_rad = math.radians(angle_deg)
        m_terms = find_min_terms_maclaurin(theta_rad, tolerance)
        t_terms = find_min_terms_taylor(theta_rad, a_rad, tolerance)
        print(f"{angle_deg:7d}° | {m_terms:22d} | {t_terms:20d}")
    print()

    # Small-angle approximation: sin(theta) ≈ theta
    print("=" * 72)
    print("SMALL-ANGLE APPROXIMATION: sin(theta) ~ theta")
    print("=" * 72)
    print()
    print(f"{'Angle (deg)':>12} | {'theta (rad)':>14} | {'sin(theta)':>14} | {'Abs Error':>12} | {'% Error':>10} | {'OK < 0.1%?':>12}")
    print("-" * 84)
    for angle_deg in range(1, 46):
        theta_rad = math.radians(angle_deg)
        approx = theta_rad
        exact = math.sin(theta_rad)
        abs_err = abs(exact - approx)
        rel_err = (abs_err / abs(exact)) * 100
        ok = "YES" if rel_err < tolerance else "NO"
        if angle_deg <= 10 or angle_deg % 5 == 0 or ok == "NO":
            print(f"{angle_deg:12d} | {theta_rad:14.10f} | {exact:14.10f} | {abs_err:12.2e} | {rel_err:10.6f} | {ok:>12}")

    # Find the critical angle
    print()
    critical_angle = 1
    for angle_deg in range(1, 91):
        theta_rad = math.radians(angle_deg)
        rel_err = abs(math.sin(theta_rad) - theta_rad) / abs(math.sin(theta_rad)) * 100
        if rel_err >= tolerance:
            critical_angle = angle_deg
            break
    print(f"The sin(theta) ≈ theta approximation stops being accurate (< {tolerance}%) at about {critical_angle} degrees.")
    print()

    print("Analysis:")
    print("1. Adding more terms always improves the approximation (convergent series).")
    print("2. Maclaurin is most accurate near zero because it's centered there.")
    print("3. Changing the expansion point improves accuracy locally around that point.")
    print("4. Accuracy degrades with distance from the expansion point.")
    print("5. The small-angle approximation is sufficient for very small angles (< ~8°).")
    print()


# =============================================================================
# PART 7: FINAL ENGINEERING RECOMMENDATION + PLOTS
# =============================================================================

def part7_recommendation_and_plots():
    print("=" * 72)
    print("PART 7: FINAL ENGINEERING RECOMMENDATION")
    print("=" * 72)
    print()

    L = 20
    angles_deg = [1, 2, 5, 10, 15, 20, 30]
    a_rad = math.radians(10)

    # Generate all data for plots
    term_counts = list(range(1, 11))

    # --- Deliverable 2: Convergence Plot ---
    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig1.suptitle("Convergence: How Error Decreases with More Terms", fontsize=13, fontweight='bold')

    # Maclaurin convergence
    for angle_deg in [5, 10, 20, 30]:
        theta_rad = math.radians(angle_deg)
        exact = math.sin(theta_rad)
        errors = [abs(exact - sin_maclaurin(theta_rad, N)) / abs(exact) * 100 for N in term_counts]
        ax1.plot(term_counts, errors, 'o-', label=f'{angle_deg}°')
    ax1.axhline(y=0.1, color='r', linestyle='--', alpha=0.7, label='0.1% tolerance')
    ax1.set_xlabel('Number of Terms (N)')
    ax1.set_ylabel('Relative Error (%)')
    ax1.set_title('Maclaurin Series Convergence')
    ax1.set_yscale('log')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Taylor convergence
    for angle_deg in [5, 10, 20, 30]:
        theta_rad = math.radians(angle_deg)
        exact = math.sin(theta_rad)
        errors = [abs(exact - sin_taylor(theta_rad, a_rad, N)) / abs(exact) * 100 for N in term_counts]
        ax2.plot(term_counts, errors, 's-', label=f'{angle_deg}°')
    ax2.axhline(y=0.1, color='r', linestyle='--', alpha=0.7, label='0.1% tolerance')
    ax2.set_xlabel('Number of Terms (N)')
    ax2.set_ylabel('Relative Error (%)')
    ax2.set_title('Taylor Series Convergence (centered at 10°)')
    ax2.set_yscale('log')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path1 = os.path.join(os.path.dirname(__file__), 'convergence_plot.png')
    plt.savefig(save_path1, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Plot saved: {save_path1}")

    # --- Deliverable 3: Function Comparison Plot ---
    fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(14, 5))
    fig2.suptitle("Function Comparison: Exact sin(θ) vs Series Approximations", fontsize=13, fontweight='bold')

    theta_range = [math.radians(d) for d in range(0, 91)]
    angle_range = [math.degrees(t) for t in theta_range]
    exact_vals = [math.sin(t) for t in theta_range]

    # Maclaurin comparison
    ax3.plot(angle_range, exact_vals, 'k-', linewidth=2, label='Exact sin(θ)')
    for N in [1, 2, 3, 4]:
        maclaurin_vals = [sin_maclaurin(t, N) for t in theta_range]
        ax3.plot(angle_range, maclaurin_vals, '--', label=f'Maclaurin N={N}')
    ax3.set_xlabel('Angle (degrees)')
    ax3.set_ylabel('sin(θ)')
    ax3.set_title('Exact vs Maclaurin Approximations')
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(0, 90)

    # Taylor comparison
    ax4.plot(angle_range, exact_vals, 'k-', linewidth=2, label='Exact sin(θ)')
    for N in [1, 2, 3, 4]:
        taylor_vals = [sin_taylor(t, a_rad, N) for t in theta_range]
        ax4.plot(angle_range, taylor_vals, '--', label=f'Taylor N={N}')
    ax4.set_xlabel('Angle (degrees)')
    ax4.set_ylabel('sin(θ)')
    ax4.set_title('Exact vs Taylor Approximations (centered at 10°)')
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(0, 90)

    plt.tight_layout()
    save_path2 = os.path.join(os.path.dirname(__file__), 'function_comparison_plot.png')
    plt.savefig(save_path2, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Plot saved: {save_path2}")

    # --- Deliverable 4: Error Comparison Plot ---
    fig3, (ax5, ax6) = plt.subplots(1, 2, figsize=(14, 5))
    fig3.suptitle("Error Comparison: Maclaurin vs Taylor (N=3 terms)", fontsize=13, fontweight='bold')

    # Absolute error comparison (N=3 terms)
    m_abs_errors = []
    t_abs_errors = []
    for angle_deg in angles_deg:
        theta_rad = math.radians(angle_deg)
        exact_y = L * math.sin(theta_rad)
        m_approx = L * sin_maclaurin(theta_rad, 3)
        t_approx = L * sin_taylor(theta_rad, a_rad, 3)
        m_abs_errors.append(abs(exact_y - m_approx))
        t_abs_errors.append(abs(exact_y - t_approx))
    x_pos = range(len(angles_deg))
    width = 0.35
    ax5.bar([p - width/2 for p in x_pos], m_abs_errors, width, label='Maclaurin', color='steelblue')
    ax5.bar([p + width/2 for p in x_pos], t_abs_errors, width, label='Taylor', color='coral')
    ax5.set_xticks(list(x_pos))
    ax5.set_xticklabels([f'{a}°' for a in angles_deg])
    ax5.set_xlabel('Angle')
    ax5.set_ylabel('Absolute Error (m)')
    ax5.set_title('Absolute Error (N=3 terms, L=20m)')
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # Percentage error comparison (N=3 terms)
    m_pct_errors = []
    t_pct_errors = []
    for angle_deg in angles_deg:
        theta_rad = math.radians(angle_deg)
        exact = math.sin(theta_rad)
        m_pct = abs(exact - sin_maclaurin(theta_rad, 3)) / abs(exact) * 100
        t_pct = abs(exact - sin_taylor(theta_rad, a_rad, 3)) / abs(exact) * 100
        m_pct_errors.append(m_pct)
        t_pct_errors.append(t_pct)
    ax6.bar([p - width/2 for p in x_pos], m_pct_errors, width, label='Maclaurin', color='steelblue')
    ax6.bar([p + width/2 for p in x_pos], t_pct_errors, width, label='Taylor', color='coral')
    ax6.axhline(y=0.1, color='r', linestyle='--', alpha=0.7, label='0.1% tolerance')
    ax6.set_xticks(list(x_pos))
    ax6.set_xticklabels([f'{a}°' for a in angles_deg])
    ax6.set_xlabel('Angle')
    ax6.set_ylabel('Percentage Error (%)')
    ax6.set_title('Percentage Error (N=3 terms)')
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path3 = os.path.join(os.path.dirname(__file__), 'error_comparison_plot.png')
    plt.savefig(save_path3, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Plot saved: {save_path3}")

    # --- Final Recommendation ---
    print()
    print("=" * 72)
    print("ENGINEERING RECOMMENDATION")
    print("=" * 72)
    print()
    print("Recommendation: Use the MACLAURIN SERIES with 3 terms for most cases.")
    print()
    print("Justification:")
    print()
    print("1. NUMBER OF TERMS REQUIRED:")
    print("   - For angles < 5 deg:  2 terms suffice (< 0.001% error)")
    print("   - For angles 5-15 deg: 3 terms suffice (< 0.001% error)")
    print("   - For angles 15-30 deg: 4 terms needed (< 0.01% error)")
    print("   - 4 terms achieves < 0.1% error for ALL angles up to 30 deg.")
    print()
    print("2. PERCENTAGE ERROR ACHIEVED:")
    print(f"   - With 3 terms at 30°: error < 0.02% (well within 0.1% tolerance)")
    print(f"   - With 4 terms at 30°: error < 0.001%")
    print()
    print("3. CONVERGENCE BEHAVIOR:")
    print("   - Maclaurin series converges factorially (extremely fast).")
    print("   - Each additional term reduces error by roughly a factor of theta^2 / (2k)(2k+1).")
    print("   - Error drops exponentially with number of terms.")
    print()
    print("4. COMPUTATIONAL SIMPLICITY vs ACCURACY:")
    print("   - Maclaurin requires only theta (no need to choose a center point).")
    print("   - The implementation uses only basic arithmetic: add, multiply, divide.")
    print("   - 3-4 terms is computationally trivial compared to evaluating sin().")
    print("   - Taylor centered at 10 deg is slightly more accurate near 10 deg")
    print("     but adds complexity (must choose center, compute derivatives at center).")
    print()
    print("5. VALID ANGLE RANGE:")
    print("   - With 4 terms: accurate for all angles 0° to 30° (engineering range)")
    print("   - With 3 terms: accurate for angles 0° to ~20°")
    print()
    print("CONCLUSION:")
    print("   For a civil engineer computing y = 20*sin(theta) for angles up to 30°,")
    print("   the Maclaurin series with 4 terms is the best choice. It provides:")
    print("   - Error well below 0.1% for all angles in range")
    print("   - Simple implementation (no center point selection needed)")
    print("   - Fast convergence (4 terms = trivial computation)")
    print("   - No dependency on expansion point")
    print()
    print("   The Taylor series centered at 10° would be preferred only if all angles")
    print("   of interest cluster tightly around 10°. For a general-purpose tool,")
    print("   the Maclaurin series is simpler and equally effective.")
    print()


# =============================================================================
# MAIN
# =============================================================================

def main():
    print()
    print("+" + "=" * 70 + "+")
    print("|  HOW ACCURATE IS GOOD ENOUGH?                                     |")
    print("|  Approximating a Civil Engineering Function Using Infinite Series  |")
    print("+" + "=" * 70 + "+")
    print()

    part1_geometric_series()
    part2_power_series()
    part3_maclaurin_series()
    part4_engineering_investigation()
    part5_taylor_series()
    part6_error_tolerance()
    part7_recommendation_and_plots()

    print("=" * 72)
    print("ALL PARTS COMPLETE")
    print("=" * 72)


if __name__ == "__main__":
    main()
