"""
Philippine Life Table (2023)
==============================
Data source: PSA 2023 Deaths Statistical Tables (Table 12)
            Deaths by Age Group and Sex, Philippines

This script builds an abridged life table for the Philippines
using registered death counts from the PSA and population
estimates from the 2020 Census projections.

[ron cedryx senson ortilla]
[github.com/rcsopro/ron-actuarial-portfolio]
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import os

# =============================================================
# STEP 1: INPUT DATA
# Death counts from PSA Table 12 (2023, Philippines TOTAL row)
# =============================================================

age_groups = [
    'Under 1', '1-4', '5-9', '10-14', '15-19',
    '20-24', '25-29', '30-34', '35-39', '40-44',
    '45-49', '50-54', '55-59', '60-64', '65-69',
    '70-74', '75-79', '80-84', '85+'
]

# Width of each age interval (years)
n = [1, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, np.inf]

deaths_male = [
    12484, 3448, 2496, 2953, 5532,
    7645, 9485, 11158, 13383, 18092,
    22683, 30053, 36344, 42463, 46270,
    45035, 34420, 25936, 23383
]

deaths_female = [
    9109, 2768, 1887, 2036, 2872,
    3651, 4292, 5507, 6945, 9581,
    12434, 16461, 19628, 24585, 29079,
    32549, 32452, 33585, 52058
]

# =============================================================
# STEP 2: POPULATION ESTIMATES (2023)
# Source: PSA 2020 Census-based population projections
# Mid-year 2023 estimates by age group and sex
# Note: Using 2020 Census projections interpolated to 2023.
# A full analysis would use official PSA mid-year estimates.
# =============================================================

pop_male = [
    999_000,  3_850_000,  5_100_000,  5_200_000,  5_050_000,
    4_800_000,  4_550_000,  4_000_000,  3_650_000,  3_200_000,
    2_750_000,  2_350_000,  1_950_000,  1_550_000,  1_150_000,
    850_000,  580_000,  330_000,  210_000
]

pop_female = [
    956_000,  3_700_000,  4_950_000,  5_050_000,  4_900_000,
    4_700_000,  4_450_000,  3_950_000,  3_600_000,  3_150_000,
    2_750_000,  2_400_000,  2_050_000,  1_700_000,  1_350_000,
    1_050_000,  750_000,  460_000,  380_000
]

# =============================================================
# STEP 3: BUILD LIFE TABLE FUNCTION
# =============================================================

def build_life_table(deaths, population, n_widths, age_labels, radix=100_000):
    """
    Builds an abridged life table from death counts and population.

    Parameters:
        deaths     : list of deaths per age group
        population : list of mid-year population per age group
        n_widths   : list of age interval widths
        age_labels : list of age group labels
        radix      : starting cohort size (default 100,000)

    Returns:
        DataFrame with life table columns
    """
    deaths = np.array(deaths, dtype=float)
    population = np.array(population, dtype=float)
    n_arr = np.array(n_widths, dtype=float)

    # Central death rate: mx = deaths / mid-year population
    mx = deaths / population

    # Convert mx to qx using UDD assumption
    # For open-ended last group (85+), qx = 1
    qx = np.where(
        np.isinf(n_arr),
        1.0,
        (n_arr * mx) / (1 + (n_arr / 2) * mx)
    )

    # Survivors: lx
    lx = np.zeros(len(age_labels))
    lx[0] = radix
    for i in range(1, len(age_labels)):
        lx[i] = lx[i - 1] * (1 - qx[i - 1])

    # Deaths in table: dx
    dx = lx * qx

    # Person-years lived: nLx
    # For open-ended group: nLx = lx / mx (if mx > 0)
    nLx = np.where(
        np.isinf(n_arr),
        np.where(mx > 0, lx / mx, 0),
        n_arr * lx - (n_arr / 2) * dx
    )

    # Total future person-years: Tx (cumulative from bottom)
    Tx = np.cumsum(nLx[::-1])[::-1]

    # Life expectancy: ex
    ex = np.where(lx > 0, Tx / lx, 0)

    df = pd.DataFrame({
        'Age Group': age_labels,
        'Deaths (dx_raw)': deaths.astype(int),
        'Population': population.astype(int),
        'mx': np.round(mx, 6),
        'qx': np.round(qx, 6),
        'lx': np.round(lx, 2),
        'dx': np.round(dx, 2),
        'nLx': np.round(nLx, 2),
        'Tx': np.round(Tx, 2),
        'ex': np.round(ex, 2)
    })

    return df


# =============================================================
# STEP 4: BUILD TABLES FOR MALE AND FEMALE
# =============================================================

lt_male = build_life_table(deaths_male, pop_male, n, age_groups)
lt_female = build_life_table(deaths_female, pop_female, n, age_groups)

print("=" * 60)
print("PHILIPPINE LIFE TABLE 2023 — MALE")
print("=" * 60)
print(lt_male[['Age Group', 'mx', 'qx', 'lx', 'ex']].to_string(index=False))

print("\n")
print("=" * 60)
print("PHILIPPINE LIFE TABLE 2023 — FEMALE")
print("=" * 60)
print(lt_female[['Age Group', 'mx', 'qx', 'lx', 'ex']].to_string(index=False))

# Key summary statistics
e0_male = lt_male.loc[0, 'ex']
e0_female = lt_female.loc[0, 'ex']
print(f"\n📊 Life Expectancy at Birth:")
print(f"   Male:   {e0_male:.1f} years")
print(f"   Female: {e0_female:.1f} years")
print(f"   Gap:    {e0_female - e0_male:.1f} years (female advantage)")

# =============================================================
# STEP 5: VISUALIZATIONS
# =============================================================

os.makedirs('outputs', exist_ok=True)
ages_numeric = [0, 1, 5, 10, 15, 20, 25, 30, 35, 40,
                45, 50, 55, 60, 65, 70, 75, 80, 85]

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('Philippine Life Table Analysis (2023)\nSource: PSA Deaths Statistical Tables',
             fontsize=13, fontweight='bold')

# --- Plot 1: Life Expectancy by Age ---
axes[0].plot(ages_numeric, lt_male['ex'], color='steelblue',
             linewidth=2, marker='o', markersize=4, label='Male')
axes[0].plot(ages_numeric, lt_female['ex'], color='coral',
             linewidth=2, marker='o', markersize=4, label='Female')
axes[0].set_title('Life Expectancy (ex) by Age')
axes[0].set_xlabel('Age')
axes[0].set_ylabel('Remaining Life Expectancy (years)')
axes[0].legend()
axes[0].grid(alpha=0.3)
axes[0].annotate(f"e₀ Male: {e0_male:.1f}y", xy=(0, e0_male),
                 xytext=(5, e0_male - 3), fontsize=9, color='steelblue')
axes[0].annotate(f"e₀ Female: {e0_female:.1f}y", xy=(0, e0_female),
                 xytext=(5, e0_female + 1), fontsize=9, color='coral')

# --- Plot 2: Probability of Death (qx) ---
axes[1].plot(ages_numeric[:-1], lt_male['qx'].iloc[:-1],
             color='steelblue', linewidth=2, label='Male')
axes[1].plot(ages_numeric[:-1], lt_female['qx'].iloc[:-1],
             color='coral', linewidth=2, label='Female')
axes[1].set_title('Probability of Death (qx) by Age')
axes[1].set_xlabel('Age')
axes[1].set_ylabel('qx')
axes[1].legend()
axes[1].grid(alpha=0.3)
axes[1].yaxis.set_major_formatter(mticker.FormatStrFormatter('%.3f'))

# --- Plot 3: Survivors (lx) ---
axes[2].plot(ages_numeric, lt_male['lx'], color='steelblue',
             linewidth=2, label='Male')
axes[2].plot(ages_numeric, lt_female['lx'], color='coral',
             linewidth=2, label='Female')
axes[2].set_title('Survivors (lx) out of 100,000')
axes[2].set_xlabel('Age')
axes[2].set_ylabel('lx')
axes[2].legend()
axes[2].grid(alpha=0.3)
axes[2].yaxis.set_major_formatter(mticker.FuncFormatter(
    lambda x, _: f'{int(x):,}'))

plt.tight_layout()
plt.savefig('outputs/ph_life_table_2023.png', dpi=150, bbox_inches='tight')
print("\n✅ Chart saved to outputs/ph_life_table_2023.png")

# =============================================================
# STEP 6: EXPORT TO CSV
# =============================================================

lt_male['Sex'] = 'Male'
lt_female['Sex'] = 'Female'
combined = pd.concat([lt_male, lt_female], ignore_index=True)
combined.to_csv('outputs/ph_life_table_2023.csv', index=False)
print("✅ Full life table saved to outputs/ph_life_table_2023.csv")

plt.show()
