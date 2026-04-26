# ============================================================
#  UAC CARE PIPELINE — EDA & KPI ANALYSIS
#  Step 1 of the UAC Analytics Project
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# 1. LOAD & CLEAN DATA
# ─────────────────────────────────────────────
df = pd.read_csv("HHS_Unaccompanied_Alien_Children_Program.csv")

# Fix comma-formatted number
df['Children in HHS Care'] = (
    df['Children in HHS Care']
    .astype(str)
    .str.replace(',', '', regex=False)
    .pipe(pd.to_numeric, errors='coerce')
)

# Parse dates, drop blanks, sort
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df = df.dropna(subset=['Date']).sort_values('Date').reset_index(drop=True)

# Short column names
df.columns = ['Date', 'CBP_Apprehended', 'CBP_Custody',
              'CBP_Transferred', 'HHS_Care', 'HHS_Discharged']

# Time features
df['Year']      = df['Date'].dt.year
df['Month']     = df['Date'].dt.month
df['MonthName'] = df['Date'].dt.strftime('%b')
df['DayOfWeek'] = df['Date'].dt.day_name()
df['IsWeekend'] = df['DayOfWeek'].isin(['Saturday', 'Sunday'])

print("=" * 60)
print("  SECTION 1 — DATA OVERVIEW")
print("=" * 60)
print(f"  Records   : {len(df)}")
print(f"  Date range: {df['Date'].min().date()} → {df['Date'].max().date()}")
print(f"  Years     : {sorted(df['Year'].unique())}")
print()
print(df[['CBP_Apprehended','CBP_Custody','CBP_Transferred',
          'HHS_Care','HHS_Discharged']].describe().round(2))


# ─────────────────────────────────────────────
# 2. COMPUTE KPIs
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("  SECTION 2 — KPI CALCULATIONS")
print("=" * 60)

# KPI 1 – Transfer Efficiency Ratio  (how fast CBP pushes kids to HHS)
df['Transfer_Efficiency_Ratio'] = (
    df['CBP_Transferred'] / df['CBP_Custody'].replace(0, np.nan)
).round(4)

# KPI 2 – Discharge Effectiveness Index  (placement rate from HHS)
df['Discharge_Effectiveness_Index'] = (
    df['HHS_Discharged'] / df['HHS_Care'].replace(0, np.nan)
).round(6)

# KPI 3 – Pipeline Throughput Rate  (total exits vs total entries)
total_entries = (df['CBP_Apprehended'] + df['CBP_Transferred']).replace(0, np.nan)
total_exits   = df['CBP_Transferred'] + df['HHS_Discharged']
df['Pipeline_Throughput_Rate'] = (total_exits / total_entries).round(4)

# KPI 4 – Backlog Accumulation Rate  (net daily build-up)
df['Backlog_Accumulation_Rate'] = df['CBP_Apprehended'] - df['HHS_Discharged']

# KPI 5 – Outcome Stability Score  (7-day rolling std of discharges)
df['Outcome_Stability_Score'] = df['HHS_Discharged'].rolling(7, min_periods=3).std().round(4)

kpi_cols = ['Transfer_Efficiency_Ratio', 'Discharge_Effectiveness_Index',
            'Pipeline_Throughput_Rate', 'Backlog_Accumulation_Rate',
            'Outcome_Stability_Score']

print(df[kpi_cols].describe().round(4))


# ─────────────────────────────────────────────
# 3. BOTTLENECK / DELAY IDENTIFICATION
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("  SECTION 3 — BOTTLENECK ANALYSIS")
print("=" * 60)

# Cumulative backlog
df['Cumulative_Backlog'] = df['Backlog_Accumulation_Rate'].cumsum()

# Flag sustained backlog: daily inflow > outflow for 7+ consecutive days
df['Backlog_Flag'] = df['Backlog_Accumulation_Rate'] > 0
df['Backlog_Streak'] = (
    df['Backlog_Flag']
    .groupby((df['Backlog_Flag'] != df['Backlog_Flag'].shift()).cumsum())
    .transform('cumsum')
)
peak_backlog_day = df.loc[df['Cumulative_Backlog'].idxmax(), 'Date']
peak_hhs         = df.loc[df['HHS_Care'].idxmax()]

print(f"  Peak cumulative backlog date : {peak_backlog_day.date()}")
print(f"  Peak HHS Care load           : {int(peak_hhs['HHS_Care']):,} on {peak_hhs['Date'].date()}")
print(f"  Days with positive backlog   : {df['Backlog_Flag'].sum()} / {len(df)}")
print(f"  Longest backlog streak       : {int(df['Backlog_Streak'].max())} consecutive days")


# ─────────────────────────────────────────────
# 4. TEMPORAL PATTERNS
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("  SECTION 4 — TEMPORAL PATTERNS")
print("=" * 60)

# Weekday vs weekend
wk = df.groupby('IsWeekend')[['CBP_Transferred','HHS_Discharged',
                               'Transfer_Efficiency_Ratio']].mean().round(3)
wk.index = ['Weekday', 'Weekend']
print("\nWeekday vs Weekend averages:")
print(wk.to_string())

# Month-over-month
monthly = df.groupby(['Year','Month']).agg(
    Avg_Transfers   = ('CBP_Transferred', 'mean'),
    Avg_Discharges  = ('HHS_Discharged',  'mean'),
    Avg_HHS_Load    = ('HHS_Care',        'mean'),
    Avg_TER         = ('Transfer_Efficiency_Ratio', 'mean'),
    Avg_DEI         = ('Discharge_Effectiveness_Index', 'mean'),
).round(3).reset_index()
print("\nMonthly KPI averages (last 6 months):")
print(monthly.tail(6).to_string(index=False))

# Year-over-year
yearly = df.groupby('Year').agg(
    Avg_TER = ('Transfer_Efficiency_Ratio', 'mean'),
    Avg_DEI = ('Discharge_Effectiveness_Index', 'mean'),
    Avg_HHS = ('HHS_Care', 'mean'),
    Total_Discharges = ('HHS_Discharged', 'sum'),
).round(4)
print("\nYear-over-Year KPI summary:")
print(yearly.to_string())


# ─────────────────────────────────────────────
# 5. OUTCOME STABILITY ANALYSIS
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("  SECTION 5 — OUTCOME STABILITY")
print("=" * 60)

# Sudden drops: discharge < 25th percentile
p25 = df['HHS_Discharged'].quantile(0.25)
sudden_drops = df[df['HHS_Discharged'] < p25][['Date','HHS_Discharged','Discharge_Effectiveness_Index']]
print(f"  Days with discharge < 25th percentile ({p25:.0f}): {len(sudden_drops)}")
print(f"  Worst 5 discharge days:")
print(sudden_drops.nsmallest(5, 'HHS_Discharged')[['Date','HHS_Discharged']].to_string(index=False))

# Zero-discharge days
zero_days = df[df['HHS_Discharged'] == 0]
print(f"\n  Zero-discharge days: {len(zero_days)}")
if len(zero_days) > 0:
    print(zero_days[['Date','Year']].to_string(index=False))


# ─────────────────────────────────────────────
# 6. VISUALIZATIONS  (saved to PNG)
# ─────────────────────────────────────────────
plt.rcParams.update({'figure.dpi': 150, 'font.size': 9,
                     'axes.spines.top': False, 'axes.spines.right': False})

BLUE   = '#2563EB'
GREEN  = '#16A34A'
RED    = '#DC2626'
ORANGE = '#EA580C'
PURPLE = '#7C3AED'
GREY   = '#94A3B8'

fig, axes = plt.subplots(4, 2, figsize=(16, 20))
fig.suptitle('UAC Care Pipeline — EDA & KPI Dashboard', fontsize=15, fontweight='bold', y=0.98)

# ── Plot 1: Pipeline Volumes Over Time
ax = axes[0, 0]
ax.plot(df['Date'], df['CBP_Custody'],    color=ORANGE, lw=1.2, label='CBP Custody')
ax.plot(df['Date'], df['HHS_Care']/100,   color=BLUE,   lw=1.5, label='HHS Care (/100)', alpha=0.8)
ax.plot(df['Date'], df['CBP_Apprehended'],color=RED,    lw=1,   label='Apprehended', alpha=0.7)
ax.set_title('Pipeline Volume Over Time')
ax.set_ylabel('Children')
ax.legend(fontsize=8)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
plt.setp(ax.xaxis.get_majorticklabels(), rotation=30)

# ── Plot 2: HHS Care Load (full scale)
ax = axes[0, 1]
ax.fill_between(df['Date'], df['HHS_Care'], alpha=0.3, color=BLUE)
ax.plot(df['Date'], df['HHS_Care'], color=BLUE, lw=1.5)
ax.axhline(df['HHS_Care'].mean(), color=RED, ls='--', lw=1, label=f"Mean: {df['HHS_Care'].mean():.0f}")
ax.set_title('HHS Care Load (Active Children)')
ax.set_ylabel('Children in HHS Care')
ax.legend()
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:,.0f}'))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
plt.setp(ax.xaxis.get_majorticklabels(), rotation=30)

# ── Plot 3: Transfer Efficiency Ratio
ax = axes[1, 0]
ax.plot(df['Date'], df['Transfer_Efficiency_Ratio'].rolling(7).mean(),
        color=GREEN, lw=1.5, label='7-day avg')
ax.scatter(df['Date'], df['Transfer_Efficiency_Ratio'],
           color=GREEN, s=4, alpha=0.3)
ax.axhline(1.0, color=RED, ls='--', lw=1, label='Ideal = 1.0')
ax.set_title('Transfer Efficiency Ratio (CBP → HHS)')
ax.set_ylabel('Ratio')
ax.legend()
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
plt.setp(ax.xaxis.get_majorticklabels(), rotation=30)

# ── Plot 4: Discharge Effectiveness Index
ax = axes[1, 1]
ax.plot(df['Date'], df['Discharge_Effectiveness_Index'].rolling(14).mean(),
        color=PURPLE, lw=1.5, label='14-day avg')
ax.set_title('Discharge Effectiveness Index (HHS → Sponsor)')
ax.set_ylabel('Ratio (Discharges / HHS Load)')
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
plt.setp(ax.xaxis.get_majorticklabels(), rotation=30)

# ── Plot 5: Backlog Accumulation Rate
ax = axes[2, 0]
colors_bar = [RED if v > 0 else GREEN for v in df['Backlog_Accumulation_Rate']]
ax.bar(df['Date'], df['Backlog_Accumulation_Rate'], color=colors_bar,
       width=1.5, alpha=0.7)
ax.axhline(0, color='black', lw=0.8)
ax.set_title('Daily Backlog Accumulation Rate\n(Red = Inflow > Outflow)')
ax.set_ylabel('Net Daily Children')
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
plt.setp(ax.xaxis.get_majorticklabels(), rotation=30)

# ── Plot 6: Cumulative Backlog
ax = axes[2, 1]
ax.plot(df['Date'], df['Cumulative_Backlog'], color=ORANGE, lw=1.5)
ax.fill_between(df['Date'], df['Cumulative_Backlog'],
                where=df['Cumulative_Backlog'] > 0, alpha=0.3, color=RED,   label='Stress')
ax.fill_between(df['Date'], df['Cumulative_Backlog'],
                where=df['Cumulative_Backlog'] <= 0, alpha=0.3, color=GREEN, label='Relief')
ax.axhline(0, color='black', lw=0.8)
ax.set_title('Cumulative Backlog Over Time')
ax.set_ylabel('Cumulative Net Children')
ax.legend()
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
plt.setp(ax.xaxis.get_majorticklabels(), rotation=30)

# ── Plot 7: Weekday vs Weekend
ax = axes[3, 0]
wk_plot = df.groupby('DayOfWeek')[['CBP_Transferred','HHS_Discharged']].mean()
order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
wk_plot = wk_plot.reindex([d for d in order if d in wk_plot.index])
x = np.arange(len(wk_plot))
w = 0.35
ax.bar(x - w/2, wk_plot['CBP_Transferred'], w, label='Transfers',  color=BLUE)
ax.bar(x + w/2, wk_plot['HHS_Discharged'],  w, label='Discharges', color=GREEN)
ax.set_xticks(x)
ax.set_xticklabels(wk_plot.index, rotation=30)
ax.set_title('Avg Daily Transfers & Discharges by Day of Week')
ax.set_ylabel('Avg Children/Day')
ax.legend()

# ── Plot 8: Year-over-Year KPIs
ax = axes[3, 1]
yr_data = df.groupby('Year').agg(
    TER=('Transfer_Efficiency_Ratio','mean'),
    DEI=('Discharge_Effectiveness_Index','mean'),
).reset_index()
ax.bar(yr_data['Year'].astype(str), yr_data['TER'], color=GREEN, alpha=0.7, label='Avg TER')
ax2 = ax.twinx()
ax2.plot(yr_data['Year'].astype(str), yr_data['DEI']*1000, 'o--',
         color=PURPLE, lw=2, label='Avg DEI (×1000)')
ax.set_title('Year-over-Year: Transfer Efficiency & Discharge Effectiveness')
ax.set_ylabel('Transfer Efficiency Ratio')
ax2.set_ylabel('Discharge Effectiveness (×1000)')
ax.legend(loc='upper left')
ax2.legend(loc='upper right')

plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig('UAC_EDA_KPI_Dashboard.png', bbox_inches='tight')
print("\n✅ Dashboard saved: UAC_EDA_KPI_Dashboard.png")

# ─────────────────────────────────────────────
# 7. SAVE PROCESSED DATASET WITH KPIs
# ─────────────────────────────────────────────
df.to_csv('UAC_Processed_with_KPIs.csv', index=False)
print("✅ Processed dataset saved: UAC_Processed_with_KPIs.csv")
print("\n🎉 EDA & KPI Analysis Complete!")
