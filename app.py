# ============================================================
#  UAC CARE PIPELINE ANALYTICS DASHBOARD
#  U.S. Department of Health and Human Services
#  Office of Refugee Resettlement
# ============================================================
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="UAC Care Pipeline Analytics", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Merriweather:wght@400;700&family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"], .stApp {
    font-family: 'Inter', sans-serif !important;
}

/* ── METRIC CARDS — forced visible in both modes ── */
[data-testid="stMetricLabel"] > div,
[data-testid="stMetricLabel"] p {
    font-size: 0.67rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.07em !important;
    text-transform: uppercase !important;
    opacity: 0.75;
}
[data-testid="stMetricValue"] > div {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 1.55rem !important;
    font-weight: 500 !important;
}
[data-testid="stMetricDelta"] svg { display: none; }
[data-testid="stMetricDelta"] > div {
    font-size: 0.72rem !important;
    opacity: 0.65;
}
div[data-testid="stMetric"] {
    border-radius: 10px !important;
    padding: 18px 20px 14px !important;
    border: 1px solid rgba(128,128,128,0.2) !important;
    background: rgba(128,128,128,0.05) !important;
}

/* ── SIDEBAR — always dark ── */
[data-testid="stSidebar"] > div:first-child {
    background: #0A1628 !important;
}
[data-testid="stSidebar"] * { color: #94A3B8 !important; }
[data-testid="stSidebar"] label p,
[data-testid="stSidebar"] .stCheckbox label p { color: #94A3B8 !important; font-size: 0.78rem !important; }
[data-testid="stSidebar"] hr { border-color: #1E3A5F !important; opacity: 0.5; }

/* ── TABS ── */
.stTabs [data-baseweb="tab"] {
    font-size: 0.80rem !important;
    font-weight: 500 !important;
    padding: 8px 18px !important;
}
.stTabs [aria-selected="true"] { font-weight: 700 !important; }

/* ── SECTION TITLE ── */
.sec-title {
    font-family: 'Merriweather', serif !important;
    font-size: 1.0rem !important;
    font-weight: 700 !important;
    margin: 2rem 0 0 0 !important;
}
.sec-divider {
    height: 2px;
    background: linear-gradient(90deg, #2563EB, transparent);
    border-radius: 2px;
    margin: 6px 0 14px 0;
}

/* ── ALERT BOXES — color only, no bg that fights dark/light ── */
.al-red    { border-left: 4px solid #DC2626; border-radius: 6px; padding: 11px 16px; background: rgba(220,38,38,0.10); font-size: 0.80rem; font-weight: 500; color: #DC2626; margin: 2px 0; }
.al-yellow { border-left: 4px solid #D97706; border-radius: 6px; padding: 11px 16px; background: rgba(217,119,6,0.10);  font-size: 0.80rem; font-weight: 500; color: #D97706; margin: 2px 0; }
.al-green  { border-left: 4px solid #16A34A; border-radius: 6px; padding: 11px 16px; background: rgba(22,163,74,0.10);  font-size: 0.80rem; font-weight: 500; color: #16A34A; margin: 2px 0; }

/* ── BANNER — always dark so text is always white ── */
.banner {
    background: linear-gradient(135deg, #0A1628 0%, #0F2845 55%, #1a3a6b 100%);
    border-radius: 12px;
    padding: 32px 40px;
    margin-bottom: 22px;
    border: 1px solid rgba(255,255,255,0.07);
    box-shadow: 0 6px 30px rgba(0,0,0,0.30);
    text-align: left;
}
.banner-tag {
    display: inline-block;
    background: rgba(37,99,235,0.25);
    border: 1px solid rgba(37,99,235,0.5);
    color: #93C5FD !important;
    font-size: 0.62rem;
    font-weight: 700;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    padding: 3px 10px;
    border-radius: 20px;
    margin-bottom: 14px;
}
.banner-title {
    font-family: 'Merriweather', serif;
    color: #F1F5F9 !important;
    font-size: 1.65rem;
    font-weight: 700;
    margin: 0 0 10px 0;
    line-height: 1.3;
    letter-spacing: -0.01em;
}
.banner-sub {
    color: #64748B !important;
    font-size: 0.73rem;
    margin: 0;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    line-height: 2;
}
.banner-stats {
    display: flex;
    gap: 40px;
    margin-top: 22px;
    padding-top: 20px;
    border-top: 1px solid rgba(255,255,255,0.08);
}
.banner-stat-val {
    font-family: 'JetBrains Mono', monospace;
    color: #F1F5F9 !important;
    font-size: 1.4rem;
    font-weight: 600;
    display: block;
}
.banner-stat-lbl {
    color: #475569 !important;
    font-size: 0.65rem;
    font-weight: 600;
    letter-spacing: 0.09em;
    text-transform: uppercase;
}

details > summary p { font-size: 0.84rem !important; font-weight: 600 !important; }
hr { opacity: 0.2; }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    df = pd.read_csv("HHS_Unaccompanied_Alien_Children_Program.csv")
    df["Children in HHS Care"] = (
        df["Children in HHS Care"].astype(str)
        .str.replace(",", "", regex=False)
        .pipe(pd.to_numeric, errors="coerce")
    )
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
    df.columns = ["Date","CBP_Apprehended","CBP_Custody","CBP_Transferred","HHS_Care","HHS_Discharged"]
    df["TER"]     = (df["CBP_Transferred"] / df["CBP_Custody"].replace(0, np.nan)).round(4)
    df["DEI"]     = (df["HHS_Discharged"]  / df["HHS_Care"].replace(0, np.nan)).round(6)
    total_in      = (df["CBP_Apprehended"] + df["CBP_Transferred"]).replace(0, np.nan)
    df["PTR"]     = ((df["CBP_Transferred"] + df["HHS_Discharged"]) / total_in).round(4)
    df["BAR"]     = df["CBP_Apprehended"] - df["HHS_Discharged"]
    df["OSS"]     = df["HHS_Discharged"].rolling(7, min_periods=3).std().round(4)
    df["CumBack"] = df["BAR"].cumsum()
    df["Year"]    = df["Date"].dt.year
    df["DOW"]     = df["Date"].dt.day_name()
    df["IsWknd"]  = df["DOW"].isin(["Saturday","Sunday"])
    return df

df = load_data()

# Plotly base — fully transparent so it adapts to dark/light
def base(title="", h=380):
    return dict(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", size=11),
        title=dict(text=title, font=dict(family="Merriweather, serif", size=13)),
        xaxis=dict(showgrid=True, gridcolor="rgba(128,128,128,0.12)",
                   linecolor="rgba(128,128,128,0.2)", tickfont=dict(size=10)),
        yaxis=dict(showgrid=True, gridcolor="rgba(128,128,128,0.12)",
                   linecolor="rgba(128,128,128,0.2)", tickfont=dict(size=10)),
        legend=dict(font=dict(size=10), bgcolor="rgba(0,0,0,0)", borderwidth=0),
        margin=dict(t=60, b=80, l=58, r=22),
        hovermode="x unified",
        height=h,
    )

C = dict(blue="#2563EB", green="#16A34A", red="#DC2626",
         orange="#EA580C", purple="#7C3AED", slate="#64748B")


# ── SIDEBAR ────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:20px 8px 14px; text-align:center;">
        <div style="color:#1D4ED8; font-size:1.6rem; font-weight:900; letter-spacing:0.02em; line-height:1;">HHS</div>
        <div style="color:#475569; font-size:0.58rem; letter-spacing:0.13em; text-transform:uppercase;
                    font-weight:600; line-height:1.7; margin-top:6px;">
            U.S. Dept. of Health &amp;<br>Human Services
        </div>
        <div style="color:#E2E8F0; font-size:0.82rem; font-weight:600; margin-top:10px; line-height:1.5;">
            UAC Care Pipeline<br>Analytics
        </div>
        <div style="height:2px; background:linear-gradient(90deg,transparent,#1D4ED8,transparent);
                    margin:14px 0 0 0; border-radius:2px;"></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<p style='color:#475569;font-size:0.62rem;letter-spacing:0.10em;text-transform:uppercase;font-weight:700;margin-bottom:4px;'>Reporting Period</p>", unsafe_allow_html=True)
    min_d, max_d = df["Date"].min().date(), df["Date"].max().date()
    d_start = st.date_input("From", value=min_d, min_value=min_d, max_value=max_d, label_visibility="collapsed")
    d_end   = st.date_input("To",   value=max_d, min_value=min_d, max_value=max_d, label_visibility="collapsed")
    st.caption(f"{min_d}  to  {max_d}")

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("<p style='color:#475569;font-size:0.62rem;letter-spacing:0.10em;text-transform:uppercase;font-weight:700;margin-bottom:4px;'>Alert Thresholds</p>", unsafe_allow_html=True)
    ter_thr  = st.slider("Transfer Efficiency (min)",            0.0, 1.5,  0.5, 0.05)
    dei_thr  = st.slider("Discharge Effectiveness (min, x1000)", 0.0, 50.0, 10.0, 1.0)
    back_thr = st.slider("Daily Backlog (max)",                   0,   300,  100,  10)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("<p style='color:#475569;font-size:0.62rem;letter-spacing:0.10em;text-transform:uppercase;font-weight:700;margin-bottom:4px;'>Display Options</p>", unsafe_allow_html=True)
    show_roll   = st.checkbox("Show Rolling Averages", value=True)
    show_alerts = st.checkbox("Show Alert Banners",    value=True)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.caption("Version 1.0  |  UAC Analytics Project")

mask = (df["Date"].dt.date >= d_start) & (df["Date"].dt.date <= d_end)
dff  = df[mask].copy()

total_apprehended = int(dff["CBP_Apprehended"].sum())
total_discharged  = int(dff["HHS_Discharged"].sum())
avg_hhs_load      = int(dff["HHS_Care"].mean())
peak_hhs          = int(dff["HHS_Care"].max())


# ── BANNER ─────────────────────────────────────────────────
st.markdown(f"""
<div class="banner">
    <div class="banner-tag">U.S. Department of Health &amp; Human Services &nbsp;&bull;&nbsp; Office of Refugee Resettlement</div>
    <p class="banner-title">Unaccompanied Alien Children<br>Care Pipeline Analytics</p>
    <p class="banner-sub">
        Reporting Period: {d_start} &ndash; {d_end}
        &nbsp;&nbsp;&bull;&nbsp;&nbsp;
        {len(dff):,} Daily Records
        &nbsp;&nbsp;&bull;&nbsp;&nbsp;
        Data Source: HHS / ORR Operations
    </p>
    <div class="banner-stats">
        <div>
            <span class="banner-stat-val">{total_apprehended:,}</span>
            <span class="banner-stat-lbl">Total Apprehended</span>
        </div>
        <div>
            <span class="banner-stat-val">{total_discharged:,}</span>
            <span class="banner-stat-lbl">Total Discharged</span>
        </div>
        <div>
            <span class="banner-stat-val">{avg_hhs_load:,}</span>
            <span class="banner-stat-lbl">Avg HHS Load / Day</span>
        </div>
        <div>
            <span class="banner-stat-val">{peak_hhs:,}</span>
            <span class="banner-stat-lbl">Peak HHS Load</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# ── ALERTS ─────────────────────────────────────────────────
if show_alerts and len(dff) > 0:
    avg_ter  = dff["TER"].mean()
    avg_dei  = dff["DEI"].mean() * 1000
    max_back = dff["BAR"].max()
    c1,c2,c3 = st.columns(3)
    with c1:
        cls = "al-red" if avg_ter < ter_thr else "al-green"
        st.markdown(f'<div class="{cls}"><b>Transfer Efficiency</b><br>Avg {avg_ter:.2f} — {"Below Threshold" if avg_ter < ter_thr else "Normal Range"}</div>', unsafe_allow_html=True)
    with c2:
        cls = "al-yellow" if avg_dei < dei_thr else "al-green"
        st.markdown(f'<div class="{cls}"><b>Discharge Effectiveness</b><br>Avg {avg_dei:.1f} — {"Below Threshold" if avg_dei < dei_thr else "Normal Range"}</div>', unsafe_allow_html=True)
    with c3:
        cls = "al-red" if max_back > back_thr else "al-green"
        st.markdown(f'<div class="{cls}"><b>Daily Backlog</b><br>Peak {max_back:.0f} — {"Spike Detected" if max_back > back_thr else "Within Limits"}</div>', unsafe_allow_html=True)
    st.write("")


# ── KPI ROW ────────────────────────────────────────────────
st.markdown('<p class="sec-title">Key Performance Indicators</p><div class="sec-divider"></div>', unsafe_allow_html=True)
k1,k2,k3,k4,k5 = st.columns(5)
with k1: st.metric("Transfer Efficiency Ratio",       f"{dff['TER'].mean():.2f}",  delta=f"{dff['TER'].mean()-1.0:+.2f} vs ideal")
with k2: st.metric("Discharge Effectiveness (x1000)", f"{dff['DEI'].mean()*1000:.1f}")
with k3: st.metric("Pipeline Throughput Rate",         f"{dff['PTR'].mean():.2f}")
with k4: st.metric("Avg Daily Backlog Rate",           f"{dff['BAR'].mean():.1f}")
with k5: st.metric("Outcome Stability Score",          f"{dff['OSS'].mean():.1f}")
st.write("")


# ── MODULE 1 ───────────────────────────────────────────────
st.markdown('<p class="sec-title"> Care Pipeline Flow</p><div class="sec-divider"></div>', unsafe_allow_html=True)
t1a, t1b = st.tabs(["Pipeline Volume Over Time", "Stage Flow Diagram (Sankey)"])

with t1a:
    fig = go.Figure()
    for col, label, color in [
        ("CBP_Apprehended","Apprehended",      C["red"]),
        ("CBP_Custody",    "CBP Custody",       C["orange"]),
        ("CBP_Transferred","Transferred to HHS",C["blue"]),
        ("HHS_Discharged", "HHS Discharges",   C["green"]),
    ]:
        y = dff[col].rolling(7).mean() if show_roll else dff[col]
        fig.add_trace(go.Scatter(x=dff["Date"], y=y, name=label,
                                 line=dict(color=color, width=2.2),
                                 hovertemplate=f"<b>{label}</b><br>%{{x|%b %d, %Y}}: %{{y:.0f}}<extra></extra>"))
    y2 = dff["HHS_Care"].rolling(7).mean() if show_roll else dff["HHS_Care"]
    fig.add_trace(go.Scatter(x=dff["Date"], y=y2, name="HHS Active Care (right axis)",
                             line=dict(color=C["purple"], width=2, dash="dot"), yaxis="y2"))
    layout = base("Pipeline Volume Over Time", 430)
    layout["yaxis"]["title"] = "Children / Day"
    layout["yaxis2"] = dict(overlaying="y", side="right", showgrid=False, tickfont=dict(size=10),
                            title=dict(text="HHS Active Care", font=dict(color=C["purple"])))
    layout["legend"] = dict(orientation="h", yanchor="top", y=-0.18, x=0, font=dict(size=10))
    fig.update_layout(**layout)
    st.plotly_chart(fig, use_container_width=True)

with t1b:
    ta = int(dff["CBP_Apprehended"].sum())
    tt = int(dff["CBP_Transferred"].sum())
    td = int(dff["HHS_Discharged"].sum())
    fig_s = go.Figure(go.Sankey(
        node=dict(pad=28, thickness=24,
                  label=["CBP Apprehension","CBP Custody","HHS Care",
                         "Sponsor Placement","Remaining in CBP","Remaining in HHS"],
                  color=["#DC2626","#EA580C","#2563EB","#16A34A","#FCA5A5","#93C5FD"],
                  line=dict(color="rgba(100,100,100,0.3)", width=0.5)),
        link=dict(source=[0,1,2,1,2], target=[1,2,3,4,5],
                  value=[ta,tt,td,max(0,ta-tt),max(0,tt-td)],
                  color=["rgba(234,88,12,0.2)","rgba(37,99,235,0.2)","rgba(22,163,74,0.2)",
                         "rgba(252,165,165,0.2)","rgba(147,197,253,0.2)"])
    ))
    fig_s.update_layout(height=420, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                        font=dict(family="Inter", size=11),
                        title=dict(text=f"Cumulative Pipeline Flow  |  {d_start} to {d_end}",
                                   font=dict(family="Merriweather, serif", size=13)),
                        margin=dict(t=48,b=16,l=16,r=16))
    st.plotly_chart(fig_s, use_container_width=True)


# ── MODULE 2 ───────────────────────────────────────────────
st.markdown('<p class="sec-title"> Transfer and Discharge Efficiency</p><div class="sec-divider"></div>', unsafe_allow_html=True)
cl, cr = st.columns(2)

with cl:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dff["Date"], y=dff["TER"], mode="markers",
                             marker=dict(color=C["green"], size=3, opacity=0.2), showlegend=False))
    if show_roll:
        fig.add_trace(go.Scatter(x=dff["Date"], y=dff["TER"].rolling(7).mean(),
                                 fill="tozeroy", fillcolor="rgba(22,163,74,0.08)",
                                 line=dict(color=C["green"], width=2.5), name="7-Day Average"))
    fig.add_hline(y=1.0, line_dash="dash", line_color=C["red"],
                  annotation_text="Ideal = 1.0", annotation_position="top right",
                  annotation_font=dict(size=10))
    fig.add_hline(y=ter_thr, line_dash="dot", line_color=C["orange"],
                  annotation_text=f"Threshold = {ter_thr}", annotation_position="bottom right",
                  annotation_font=dict(size=10))
    layout = base("Transfer Efficiency Ratio — CBP to HHS", 360)
    layout["yaxis"]["title"] = "Ratio"
    fig.update_layout(**layout)
    st.plotly_chart(fig, use_container_width=True)

with cr:
    dei = dff["DEI"] * 1000
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dff["Date"], y=dei, mode="markers",
                             marker=dict(color=C["purple"], size=3, opacity=0.2), showlegend=False))
    if show_roll:
        fig.add_trace(go.Scatter(x=dff["Date"], y=dei.rolling(14).mean(),
                                 fill="tozeroy", fillcolor="rgba(124,58,237,0.08)",
                                 line=dict(color=C["purple"], width=2.5), name="14-Day Average"))
    fig.add_hline(y=dei_thr, line_dash="dot", line_color=C["orange"],
                  annotation_text=f"Threshold = {dei_thr}", annotation_position="top right",
                  annotation_font=dict(size=10))
    layout = base("Discharge Effectiveness Index — HHS to Sponsor (x1000)", 360)
    layout["yaxis"]["title"] = "DEI x1000"
    fig.update_layout(**layout)
    st.plotly_chart(fig, use_container_width=True)


# ── MODULE 3 ───────────────────────────────────────────────
st.markdown('<p class="sec-title">Bottleneck and Delay Detection</p><div class="sec-divider"></div>', unsafe_allow_html=True)
t3a, t3b = st.tabs(["Daily Backlog Rate", "Cumulative Backlog"])

with t3a:
    bar_c = [C["red"] if v > 0 else C["green"] for v in dff["BAR"]]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=dff["Date"], y=dff["BAR"], marker_color=bar_c, marker_line_width=0,
                         hovertemplate="Date: %{x|%b %d, %Y}<br>Net Backlog: %{y:.0f}<extra></extra>"))
    fig.add_hline(y=0, line_color="rgba(128,128,128,0.4)", line_width=1)
    fig.add_hline(y=back_thr, line_dash="dash", line_color=C["orange"],
                  annotation_text=f"Alert: {back_thr}", annotation_font=dict(size=10))
    layout = base("Daily Backlog Accumulation Rate  (Red = Stress  |  Green = Relief)", 380)
    layout["yaxis"]["title"] = "Net Children / Day"
    layout["bargap"] = 0.08
    fig.update_layout(**layout)
    st.plotly_chart(fig, use_container_width=True)
    b1,b2,b3,b4 = st.columns(4)
    b1.metric("Stress Days",  f"{(dff['BAR']>0).sum()}")
    b2.metric("Relief Days",  f"{(dff['BAR']<0).sum()}")
    b3.metric("Peak Backlog", f"{dff['BAR'].max():.0f}")
    b4.metric("Best Relief",  f"{dff['BAR'].min():.0f}")

with t3b:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dff["Date"], y=dff["CumBack"],
                             line=dict(color=C["orange"], width=2.5),
                             fill="tozeroy", fillcolor="rgba(234,88,12,0.08)",
                             hovertemplate="Date: %{x|%b %d, %Y}<br>Cumulative: %{y:.0f}<extra></extra>"))
    fig.add_hline(y=0, line_color="rgba(128,128,128,0.4)", line_width=1)
    pk = dff["CumBack"].idxmax()
    if pk in dff.index:
        pd_, pv_ = dff.loc[pk,"Date"], dff.loc[pk,"CumBack"]
        fig.add_annotation(x=pd_, y=pv_,
                           text=f"Peak: {pv_:.0f}<br>{pd_.strftime('%b %Y')}",
                           showarrow=True, arrowhead=2, bgcolor="rgba(220,38,38,0.1)",
                           font=dict(color=C["red"], size=10), bordercolor=C["red"])
    layout = base("Cumulative System Backlog Over Time", 380)
    layout["yaxis"]["title"] = "Cumulative Net Children"
    fig.update_layout(**layout)
    st.plotly_chart(fig, use_container_width=True)


# ── MODULE 4 ───────────────────────────────────────────────
st.markdown('<p class="sec-title">Outcome Trend Analysis</p><div class="sec-divider"></div>', unsafe_allow_html=True)
t4a, t4b, t4c = st.tabs(["Monthly Trends", "Weekday vs. Weekend", "Year-over-Year"])

with t4a:
    monthly = dff.groupby(dff["Date"].dt.to_period("M")).agg(
        Avg_Transfers  = ("CBP_Transferred","mean"),
        Avg_Discharges = ("HHS_Discharged","mean"),
        Avg_TER        = ("TER","mean"),
    ).reset_index()
    monthly["Date"] = monthly["Date"].dt.to_timestamp()
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=monthly["Date"], y=monthly["Avg_Transfers"],
                         name="Avg Transfers",  marker_color=C["blue"],  opacity=0.8), secondary_y=False)
    fig.add_trace(go.Bar(x=monthly["Date"], y=monthly["Avg_Discharges"],
                         name="Avg Discharges", marker_color=C["green"], opacity=0.8), secondary_y=False)
    fig.add_trace(go.Scatter(x=monthly["Date"], y=monthly["Avg_TER"], name="Avg TER",
                             line=dict(color=C["orange"], width=2.5), mode="lines+markers",
                             marker=dict(size=5)), secondary_y=True)
    layout = base("Month-over-Month: Transfers, Discharges and Transfer Efficiency", 400)
    layout["barmode"] = "group"
    layout["legend"] = dict(orientation="h", yanchor="top", y=-0.18, x=0, font=dict(size=10))
    fig.update_layout(**layout)
    fig.update_yaxes(title_text="Avg Children / Day", secondary_y=False,
                     showgrid=True, gridcolor="rgba(128,128,128,0.12)")
    fig.update_yaxes(title_text="Transfer Efficiency Ratio", secondary_y=True, showgrid=False)
    st.plotly_chart(fig, use_container_width=True)

with t4b:
    day_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    wk = dff.groupby("DOW")[["CBP_Transferred","HHS_Discharged","TER"]].mean().round(3)
    wk = wk.reindex([d for d in day_order if d in wk.index])
    is_we = [d in ["Saturday","Sunday"] for d in wk.index]
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=("Avg Daily Volume by Day of Week",
                                        "Avg Transfer Efficiency by Day"))
    fig.add_trace(go.Bar(x=wk.index, y=wk["CBP_Transferred"], name="Transfers",
                         marker_color=[C["slate"] if w else C["blue"]  for w in is_we], opacity=0.85), row=1, col=1)
    fig.add_trace(go.Bar(x=wk.index, y=wk["HHS_Discharged"],  name="Discharges",
                         marker_color=[C["slate"] if w else C["green"] for w in is_we], opacity=0.85), row=1, col=1)
    fig.add_trace(go.Bar(x=wk.index, y=wk["TER"],
                         marker_color=[C["slate"] if w else C["blue"] for w in is_we], showlegend=False), row=1, col=2)
    fig.add_hline(y=1.0, line_dash="dash", line_color=C["red"], row=1, col=2)
    layout = base("Weekday vs. Weekend Operational Comparison", 380)
    layout["barmode"] = "group"
    layout["legend"] = dict(orientation="h", yanchor="top", y=-0.18, x=0, font=dict(size=10))
    fig.update_layout(**layout)
    st.plotly_chart(fig, use_container_width=True)

with t4c:
    yearly = dff.groupby("Year").agg(
        Avg_TER        = ("TER","mean"),
        Avg_DEI_x1000  = ("DEI", lambda x: x.mean()*1000),
        Avg_HHS        = ("HHS_Care","mean"),
        Tot_Discharges = ("HHS_Discharged","sum"),
        Tot_Transfers  = ("CBP_Transferred","sum"),
    ).round(3).reset_index()
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=("Average TER and DEI by Year",
                                        "Total Transfers and Discharges by Year"))
    fig.add_trace(go.Bar(x=yearly["Year"].astype(str), y=yearly["Avg_TER"],
                         name="Avg TER", marker_color=C["blue"], opacity=0.85), row=1, col=1)
    fig.add_trace(go.Scatter(x=yearly["Year"].astype(str), y=yearly["Avg_DEI_x1000"],
                             name="Avg DEI x1000", mode="lines+markers",
                             line=dict(color=C["orange"], width=2.5), marker=dict(size=8)), row=1, col=1)
    fig.add_trace(go.Bar(x=yearly["Year"].astype(str), y=yearly["Tot_Discharges"],
                         name="Total Discharges", marker_color=C["green"], opacity=0.85), row=1, col=2)
    fig.add_trace(go.Bar(x=yearly["Year"].astype(str), y=yearly["Tot_Transfers"],
                         name="Total Transfers", marker_color=C["purple"], opacity=0.85), row=1, col=2)
    layout = base("Year-over-Year Performance Summary", 380)
    layout["barmode"] = "group"
    layout["legend"] = dict(orientation="h", yanchor="top", y=-0.18, x=0, font=dict(size=10))
    fig.update_layout(**layout)
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(yearly.rename(columns={
        "Avg_TER":"Avg TER","Avg_DEI_x1000":"Avg DEI x1000",
        "Avg_HHS":"Avg HHS Load","Tot_Discharges":"Total Discharges","Tot_Transfers":"Total Transfers"
    }), use_container_width=True, hide_index=True)


# ── MODULE 5 ───────────────────────────────────────────────
st.markdown('<p class="sec-title"> Outcome Stability Analysis</p><div class="sec-divider"></div>', unsafe_allow_html=True)
ca, cb = st.columns([2,1])
p25 = dff["HHS_Discharged"].quantile(0.25)

with ca:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dff["Date"], y=dff["HHS_Discharged"], mode="markers",
                             marker=dict(color=C["green"], size=4, opacity=0.25), name="Daily Discharges"))
    if show_roll:
        fig.add_trace(go.Scatter(x=dff["Date"], y=dff["HHS_Discharged"].rolling(14).mean(),
                                 line=dict(color=C["green"], width=2.5), name="14-Day Average"))
    fig.add_trace(go.Scatter(x=dff["Date"], y=dff["OSS"],
                             line=dict(color=C["orange"], width=1.5, dash="dot"),
                             name="Stability Score (7-day std)", yaxis="y2"))
    fig.add_hline(y=p25, line_dash="dash", line_color=C["red"],
                  annotation_text=f"25th Pct ({p25:.0f})", annotation_font=dict(size=10))
    layout = base("Daily Discharges and Outcome Stability Score", 380)
    layout["yaxis"]["title"] = "Children Discharged"
    layout["yaxis2"] = dict(title="Stability Score", overlaying="y", side="right",
                            showgrid=False, tickfont=dict(size=10))
    layout["legend"] = dict(orientation="h", yanchor="top", y=-0.18, x=0, font=dict(size=10))
    fig.update_layout(**layout)
    st.plotly_chart(fig, use_container_width=True)

with cb:
    st.markdown("**Stability Summary**")
    st.metric("Zero-Discharge Days",              f"{(dff['HHS_Discharged']==0).sum()}")
    st.metric("Low-Discharge Days (< 25th Pct)",  f"{(dff['HHS_Discharged']<p25).sum()}")
    worst = dff.nsmallest(5,"HHS_Discharged")[["Date","HHS_Discharged"]].copy()
    worst["Date"] = worst["Date"].dt.strftime("%b %d, %Y")
    st.markdown("**Five Lowest Discharge Days**")
    st.dataframe(worst.rename(columns={"HHS_Discharged":"Discharges"}),
                 hide_index=True, use_container_width=True)


# ── RAW DATA ───────────────────────────────────────────────
with st.expander("View Processed Data Table with KPIs"):
    cols = ["Date","CBP_Apprehended","CBP_Custody","CBP_Transferred",
            "HHS_Care","HHS_Discharged","TER","DEI","PTR","BAR","OSS"]
    st.dataframe(dff[cols].sort_values("Date", ascending=False).reset_index(drop=True),
                 use_container_width=True, height=300)
    st.download_button("Download Processed Dataset (CSV)",
                       dff[cols].to_csv(index=False).encode("utf-8"),
                       "UAC_Processed_KPIs.csv", "text/csv")

# ── FOOTER ─────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center; opacity:0.4; font-size:0.72rem; padding:20px 0 6px; letter-spacing:0.04em;">
    UAC Care Pipeline Analytics &bull; U.S. Department of Health &amp; Human Services &bull;
    Office of Refugee Resettlement &bull; Version 1.0
</div>
""", unsafe_allow_html=True)