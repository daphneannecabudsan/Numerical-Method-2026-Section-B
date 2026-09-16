import numpy as np
import matplotlib.pyplot as plt
import math
 
# ============================================
# Exercise 1: Convergence of (1 + 1/n)^n to e
# ============================================
 
def exercise1():
    """Compute (1 + 1/n)^n for different n values and plot histogram"""
    
    # Values of n (compounding frequency, up to nanosecond)
    # yearly=1, twice a year=2, quarterly=4, monthly=12, weekly=52, daily=365,
    # hourly=~8760 -> use 1000, 10000, 100000, 1000000 (micro), 1000000000 (nano)
    n_values = [1, 2, 4, 12, 52, 365, 1000, 10000, 100000, 1000000, 1000000000]
    freq_labels = ['yearly', 'twice a year', 'quarterly', 'monthly',
                   'weekly', 'daily', '1000', '10000', '100000', '1e6', 'nanosecond']
    
    # Compute (1 + 1/n)^n
    results = [(1 + 1/n)**n for n in n_values]
    
    # Actual value of e
    e_actual = math.e
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Bar chart of values
    ax1.bar(range(len(n_values)), results, color='skyblue', edgecolor='navy')
    ax1.axhline(y=e_actual, color='red', linestyle='--', label=f'e = {e_actual:.6f}')
    ax1.set_xticks(range(len(n_values)))
    ax1.set_xticklabels(freq_labels, rotation=45)
    ax1.set_xlabel('n (frequency per year)')
    ax1.set_ylabel('(1 + 1/n)^n')
    ax1.set_title('Value per compounding period')
    ax1.set_ylim(1.9, 3.1)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Convergence errors
    errors = [abs(result - e_actual) for result in results]
    ax2.bar(range(len(n_values)), errors, color='salmon', edgecolor='darkred')
    ax2.set_yscale('log')
    ax2.set_xticks(range(len(n_values)))
    ax2.set_xticklabels(freq_labels, rotation=45)
    ax2.set_xlabel('n (frequency per year)')
    ax2.set_ylabel('|(1 + 1/n)^n - e|')
    ax2.set_title('Error shrinks like e/(2n)')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('Series/exercise1_convergence.png', dpi=150)
    plt.close()
    
    # Print table
    print("\n" + "="*60)
    print("Exercise 1: (1 + 1/n)^n convergence to e")
    print("="*60)
    print(f"{'How often':<18} {'n':<8} {'(1 + 1/n)^n':<20} {'Error':<15}")
    print("-"*60)
    for n, result, label in zip(n_values, results, freq_labels):
        print(f"{label:<18} {n:<8} {result:<20.10f} {abs(result - e_actual):<15.10f}")
    print(f"{'e (actual)':<18} {'':<8} {e_actual:<20.10f}")
    print("="*60)
 
# ============================================
# Exercise 2: (a^h - 1)/h settling to ln(a)
# ============================================
 
def exercise2():
    """Compute (a^h - 1)/h for different bases and h values"""
    
    # Values of h (shrinking, down to 1e-7 as shown in the PDF)
    h_values = [0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001, 0.0000001]
    
    # Bases to test
    bases = [2, math.e, 3]
    base_labels = ['a = 2 (ln a = 0.6931)',
                   'a = e (ln a = 1.0000)',
                   'a = 3 (ln a = 1.0986)']
    
    # Short legend labels for the plot (PDF uses these)
    plot_labels = ['a = 2', 'a = e', 'a = 3']
    
    # Compute results
    results = []
    for a in bases:
        row = []
        for h in h_values:
            # (a^h - 1) / h
            value = (a**h - 1) / h
            row.append(value)
        results.append(row)
    
    # Actual log values
    log_values = [math.log(2), 1.0, math.log(3)]
    
    # Single-panel grouped bar chart (as in the PDF "Figure 2")
    fig, ax1 = plt.subplots(figsize=(14, 6))
    
    x_pos = np.arange(len(h_values))
    width = 0.25
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    for i, (row, label) in enumerate(zip(results, plot_labels)):
        ax1.bar(x_pos + i*width, row, width, label=label, color=colors[i])
    
    # Add dashed horizontal lines for ln(a) (the limits)
    for i, (log_val, label, color) in enumerate(zip(log_values, plot_labels, colors)):
        ax1.axhline(y=log_val, color=color, linestyle='--',
                    label=f'{label}: ln(a) = {log_val:.4f}', alpha=0.7)
    
    ax1.set_xticks(x_pos + width)
    ax1.set_xticklabels(['h = ' + str(h) for h in h_values], rotation=30)
    ax1.set_xlabel('h (step size)')
    ax1.set_ylabel('(a^h - 1) / h')
    ax1.set_title('Exercise 2: (a^h - 1)/h settling to ln(a)')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0.6, 1.25)
    
    plt.tight_layout()
    plt.savefig('Series/exercise2_convergence.png', dpi=150)
    plt.close()
    
    # Print table with tolerance check
    tolerance = 1e-6
    print("\n" + "="*80)
    print("Exercise 2: (a^h - 1)/h convergence to ln(a)")
    print("="*80)
    print(f"{'h':<12}", end="")
    for label in base_labels:
        print(f"{label:<26}", end="")
    print()
    print("-"*80)
    
    for i, h in enumerate(h_values):
        print(f"{h:<12}", end="")
        for j, row in enumerate(results):
            val = row[i]
            log_val = log_values[j]
            status = "V" if abs(val - log_val) < tolerance else " "
            print(f"{val:.10f} {status:<10}", end="")
        print()
    
    print("-"*80)
    print(f"{'ln(a)':<12}", end="")
    for log_val in log_values:
        print(f"{log_val:.10f} {' ':>10}", end="")
    print()
    print(f"{'Tolerance':<12} {tolerance}")
    print("="*80)
 
# ============================================
# Exercise 3: e^x = sum_{n=0}^{∞} x^n / n!
# ============================================
 
def exercise3():
    """Compute e^x = sum x^n/n! (x = 1) up to N terms in a single merged figure"""
    
    # Value of x (as shown in the PDF: x = 1)
    x = 1.0
    
    # Number of terms (starting from n=0)
    N = 10000
    
    # Actual e (since x = 1, e^x = e)
    actual = math.exp(x)
    
    # Compute Taylor series partial sums S_N = sum_{n=0}^{N} x^n / n!
    terms = []
    cumulative_sum = 0
    for n in range(N):
        term = (x**n) / math.factorial(n)
        cumulative_sum += term
        terms.append(cumulative_sum)
        # Stop once machine precision is reached
        if n > 0 and abs(terms[-1] - terms[-2]) < 1e-16:
            break
    
    # Number of terms actually summed
    num_terms = len(terms)
    
    # Errors |S_N - e|
    errors = [abs(val - actual) for val in terms]
    
    print("\n" + "="*80)
    print("Exercise 3: e^x = sum x^n / n!  (x = 1)")
    print(f"Summation up to N = {num_terms} terms")
    print("="*80)
    print(f"{'e (actual)':<15} {actual:.10f}")
    print(f"{'e (series)':<15} {terms[-1]:.10f}")
    print(f"{'Error':<15} {errors[-1]:.10e}")
    print(f"{'Terms needed':<15} {num_terms}")
    print("="*80)
    
    # Single merged figure with two side-by-side panels (as in the PDF "Figure 3")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Panel 1: Histogram of the partial sums
    ax1.bar(range(1, num_terms + 1), terms, color='skyblue', edgecolor='navy')
    ax1.axhline(y=actual, color='red', linestyle='--', label=f'e = {actual:.6f}')
    ax1.set_xlabel('number of terms N in the summation')
    ax1.set_ylabel('S_N = sum_(n=0..N) x^n / n!')
    ax1.set_title('Histogram of the partial sums')
    ax1.set_ylim(1.0, 3.0)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Panel 2: How many correct digits each N buys (log-log)
    ax2.loglog(range(1, num_terms + 1), errors, 'o-', color='#45B7D1',
               linewidth=2, label='|S_N - e|')
    ax2.axhline(y=2e-16, color='green', linestyle='--',
                label='machine precision (~2e-16)')
    ax2.axhspan(1e-2, 10, color='red', alpha=0.10, label='rough (error > 1e-2)')
    ax2.axhspan(1e-8, 1e-2, color='orange', alpha=0.10, label='engineering (1e-8 to 1e-2)')
    ax2.axhspan(1e-4, 1e-8, color='blue', alpha=0.05, label='high precision (~1e-4)')
    ax2.set_xlabel('number of terms N in the summation (log scale)')
    ax2.set_ylabel('|S_N - e| (log scale)')
    ax2.set_title('How many correct digits each N buys (log-log)')
    ax2.legend(loc='lower left')
    ax2.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    plt.savefig('Series/exercise3_taylor_series.png', dpi=150)
    plt.close()
    
    print("="*80)
 
# ============================================
# Main execution
# ============================================
 
if __name__ == "__main__":
    print("\n" + "="*80)
    print("SERIES EXERCISES - PYTHON IMPLEMENTATION")
    print("="*80)
    
    # Exercise 1
    exercise1()
    
    # Exercise 2
    exercise2()
    
    # Exercise 3
    exercise3()
    
    print("\n" + "="*80)
    print("All exercises completed successfully!")
    print("="*80)