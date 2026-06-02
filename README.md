# Philippine Life Table (2023)

An abridged life table for the Philippines built from official PSA mortality data,
disaggregated by sex and age group.

---

## What is a Life Table?

A life table is a core actuarial tool that models the mortality experience of a
population. Starting from a hypothetical cohort of 100,000 births (the radix),
it tracks how many survive to each age, the probability of dying within each
interval, and the remaining life expectancy at every age.

Life tables are used in insurance pricing, annuity valuation, health plan
reserving, and population projections.

---

## Data Source

**Philippine Statistics Authority (PSA)**
2023 Deaths Statistical Tables
Table 12: Number of Registered Deaths by Age Group, Sex, and Cause of Death
Total registered deaths (Philippines): 694,821

Population denominators are based on 2020 Census-based projections interpolated
to 2023. Official PSA mid-year population estimates should be used for a more
precise central death rate (mx).

---

## Key Findings

| Metric | Male | Female |
|---|---|---|
| Life expectancy at birth (e₀) | 69.2 years | 75.8 years |
| Life expectancy at 30 | 41.8 years | 42.9 years |
| Life expectancy at 65 | 15.6 years | 18.1 years |
| Gender gap (e₀) | — | +6.6 years |

Female life expectancy exceeds male at every age group. The gap is most
pronounced at birth and narrows gradually with age. Male mortality accelerates
sharply from age 40 onwards, visible in the qx curve.

---

## Methodology

1. Central death rate computed as `mx = deaths / mid-year population`
2. Converted to probability of death using the UDD assumption:
   `qx = (n * mx) / (1 + (n/2) * mx)`
3. Open-ended terminal age group (85+) assigned `qx = 1`
4. Life table columns derived sequentially: `lx → dx → nLx → Tx → ex`

---

## Limitations

- Population estimates are approximations based on 2020 Census projections,
  not official 2023 mid-year figures
- Death counts are based on civil registration and may undercount deaths in
  remote areas
- This is an abridged (not complete) life table — age groups are not single years

---

## Tools

- Python 3.14
- pandas, numpy, matplotlib

---

## Author

Ron Cedryx S. Ortilla
Pursuing ASP actuarial credentials
[[LinkedIn]](https://www.linkedin.com/in/ron-ortilla/) | rcso.pro@gmail.com
