import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# --- PAGE CONFIG ---
st.set_page_config(page_title="Core PCE Decomposition", page_icon="📊", layout="wide")

# --- CONFIG ---
WEIGHTS = {'housing': 0.175, 'non_housing_services': 0.575, 'core_goods': 0.250}

# --- DATA ---
@st.cache_data(ttl=3600)
def fetch_pce_data():
    url = "https://www.frbsf.org/wp-content/uploads/pce-contributions-data.xlsx"
    
    # YoY Core PCE data
    df_yoy_raw = pd.read_excel(url, sheet_name='chart3_corePCEPI_YoY')
    df_yoy_raw['date'] = pd.to_datetime(df_yoy_raw['date'])
    df_yoy_raw = df_yoy_raw.set_index('date')
    
    # MoM Core PCE data
    df_mom_raw = pd.read_excel(url, sheet_name='extra_corePCEPI_MoM')
    df_mom_raw['date'] = pd.to_datetime(df_mom_raw['date'])
    df_mom_raw = df_mom_raw.set_index('date')
    
    df_yoy = pd.DataFrame({
        'core_pce': df_yoy_raw['Total: Core PCEPI, YoY'],
        'housing': df_yoy_raw['Housing'],
        'non_housing_services': df_yoy_raw['Core Services exc. Housing'],
        'core_goods': df_yoy_raw['Core Goods']
    })
    
    df_mom = pd.DataFrame({
        'core_pce': df_mom_raw['Total: Core PCEPI, MoM'],
        'housing': df_mom_raw['Housing'],
        'non_housing_services': df_mom_raw['Core Services exc. Housing'],
        'core_goods': df_mom_raw['Core Goods']
    })
    
    return df_mom, df_yoy

def generate_forecast(df_mom, housing_pace, non_housing_pace, goods_pace, months=12):
    last_date = df_mom.index[-1]
    forecast_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=months, freq='MS')
    forecast = pd.DataFrame(index=forecast_dates)
    
    forecast['housing_contrib'] = housing_pace * WEIGHTS['housing']
    forecast['non_housing_contrib'] = non_housing_pace * WEIGHTS['non_housing_services']
    forecast['goods_contrib'] = goods_pace * WEIGHTS['core_goods']
    forecast['total_mom'] = forecast['housing_contrib'] + forecast['non_housing_contrib'] + forecast['goods_contrib']
    forecast['annualized'] = ((1 + forecast['total_mom']/100)**12 - 1) * 100
    
    return forecast

def calculate_yoy_path(df_mom, forecast):
    recent_actual = df_mom.tail(12).copy()
    
    combined_housing = pd.concat([recent_actual['housing'], forecast['housing_contrib']])
    combined_non_housing = pd.concat([recent_actual['non_housing_services'], forecast['non_housing_contrib']])
    combined_goods = pd.concat([recent_actual['core_goods'], forecast['goods_contrib']])
    
    yoy_path = []
    for i in range(len(forecast)):
        h_12m = combined_housing.iloc[i:i+12].sum()
        nh_12m = combined_non_housing.iloc[i:i+12].sum()
        g_12m = combined_goods.iloc[i:i+12].sum()
        total_yoy = h_12m + nh_12m + g_12m
        
        yoy_path.append({
            'date': forecast.index[i],
            'total_yoy': total_yoy,
            'housing_contrib_yoy': h_12m,
            'non_housing_contrib_yoy': nh_12m,
            'goods_contrib_yoy': g_12m
        })
    
    return pd.DataFrame(yoy_path).set_index('date')

def create_yoy_chart(df_yoy, yoy_path):
    colors = {'housing': '#00B6B2', 'non_housing': '#003865', 'goods': '#D29F2A', 'total': '#1B1B1E'}
    
    cutoff = df_yoy.index[-1] - pd.DateOffset(years=2)
    df_plot = df_yoy[df_yoy.index >= cutoff]
    
    fig = go.Figure()
    
    # Historical stacked area
    fig.add_trace(go.Scatter(
        x=df_plot.index, y=df_plot['core_goods'],
        name='Core Goods', mode='lines',
        fill='tozeroy', fillcolor='rgba(210, 159, 42, 0.6)',
        line=dict(color=colors['goods'], width=0.5),
        stackgroup='hist'
    ))
    fig.add_trace(go.Scatter(
        x=df_plot.index, y=df_plot['non_housing_services'],
        name='Services ex. Housing', mode='lines',
        fill='tonexty', fillcolor='rgba(0, 56, 101, 0.6)',
        line=dict(color=colors['non_housing'], width=0.5),
        stackgroup='hist'
    ))
    fig.add_trace(go.Scatter(
        x=df_plot.index, y=df_plot['housing'],
        name='Housing', mode='lines',
        fill='tonexty', fillcolor='rgba(0, 182, 178, 0.6)',
        line=dict(color=colors['housing'], width=0.5),
        stackgroup='hist'
    ))
    
    # Forecast stacked area (lighter - 0.4 opacity)
    fig.add_trace(go.Scatter(
        x=yoy_path.index, y=yoy_path['goods_contrib_yoy'],
        name= 'Core Goods', mode='lines', fill='tozeroy', fillcolor='rgba(210, 159, 42, 0.4)',
        line=dict(color=colors['goods'], width=1, dash='dot'),
        stackgroup='fcst', showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=yoy_path.index, y=yoy_path['non_housing_contrib_yoy'],
        name='Services ex. Housing', mode='lines', fill='tonexty', fillcolor='rgba(0, 56, 101, 0.4)',
        line=dict(color=colors['non_housing'], width=1, dash='dot'),
        stackgroup='fcst', showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=yoy_path.index, y=yoy_path['housing_contrib_yoy'],
        name='Housing',mode='lines', fill='tonexty', fillcolor='rgba(0, 182, 178, 0.4)',
        line=dict(color=colors['housing'], width=1, dash='dot'),
        stackgroup='fcst', showlegend=False
    ))
    
    # Core PCE lines
    fig.add_trace(go.Scatter(
        x=df_plot.index, y=df_plot['core_pce'],
        name='Core PCE YoY', mode='lines',
        line=dict(color=colors['total'], width=2.5)
    ))
    fig.add_trace(go.Scatter(
        x=yoy_path.index, y=yoy_path['total_yoy'],
        name='Core PCE',mode='lines+markers', showlegend=False,
        line=dict(color=colors['total'], width=2.5, dash='dot'),
        marker=dict(size=5)
    ))
    
    fig.add_hline(y=2.0, line=dict(color='rgba(0,0,0,0.3)', width=1, dash='dash'), annotation_text='2% Target')
    fig.add_vrect(x0=yoy_path.index[0], x1=yoy_path.index[-1], fillcolor='rgba(200,200,200,0.15)', layer='below', line_width=0)
    
    fig.update_layout(
        height=450,
        hovermode='x unified',
        yaxis_title='% Change Year-Over-Year',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0),
        margin=dict(l=60, r=40, t=80, b=60)
    )
    fig.update_xaxes(dtick='M3', tickformat='%b\n%Y')
    fig.update_yaxes(ticksuffix='%')
    
    return fig

def create_mom_chart(df_mom, forecast):
    colors = {'housing': '#00B6B2', 'non_housing': '#003865', 'goods': '#D29F2A', 'total': '#1B1B1E'}
    
    cutoff = df_mom.index[-1] - pd.DateOffset(years=2)
    df_plot = df_mom[df_mom.index >= cutoff]
    
    fig = go.Figure()
    
    # Historical bars
    fig.add_trace(go.Bar(x=df_plot.index, y=df_plot['core_goods'], name='Core Goods', marker_color='rgba(241, 143, 1, 0.8)'))
    fig.add_trace(go.Bar(x=df_plot.index, y=df_plot['non_housing_services'], name='Non-Housing Services', marker_color='rgba(162, 59, 114, 0.8)'))
    fig.add_trace(go.Bar(x=df_plot.index, y=df_plot['housing'], name='Housing', marker_color='rgba(46, 134, 171, 0.8)'))
    
    # Forecast bars
    fig.add_trace(go.Bar(x=forecast.index, y=forecast['goods_contrib'], marker_color='rgba(241, 143, 1, 0.4)', showlegend=False))
    fig.add_trace(go.Bar(x=forecast.index, y=forecast['non_housing_contrib'], marker_color='rgba(162, 59, 114, 0.4)', showlegend=False))
    fig.add_trace(go.Bar(x=forecast.index, y=forecast['housing_contrib'], marker_color='rgba(46, 134, 171, 0.4)', showlegend=False))
    
    # Core PCE MoM lines
    fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['core_pce'], name='Core PCE MoM', mode='lines', line=dict(color=colors['total'], width=2.5)))
    fig.add_trace(go.Scatter(x=forecast.index, y=forecast['total_mom'], mode='lines+markers', showlegend=False, line=dict(color=colors['total'], width=2.5, dash='dot'), marker=dict(size=5)))
    
    fig.add_hline(y=0.17, line=dict(color='rgba(0,0,0,0.3)', width=1, dash='dash'), annotation_text='~2% ann.')
    fig.add_vrect(x0=forecast.index[0], x1=forecast.index[-1], fillcolor='rgba(200,200,200,0.15)', layer='below', line_width=0)
    
    fig.update_layout(
        title=dict(text='<b>Core PCE MoM: History + Forecast</b>', y=0.95),
        height=450,
        barmode='relative',
        hovermode='x unified',
        yaxis_title='Contribution (pp)',
        legend=dict(orientation='h', yanchor='top', y=-0.15, xanchor='center', x=0.5),
        margin=dict(l=60, r=40, t=80, b=60)
    )
    fig.update_xaxes(dtick='M3', tickformat='%b\n%Y')
    fig.update_yaxes(ticksuffix='%')
    
    return fig

def create_vu_ratio_chart():
    data = """date,vu_ratio,non_housing_services,recessions
1/31/10,0.188555098,2.24,0
2/28/10,0.17640442,2.47,0
3/31/10,0.176226812,2.9,0
4/30/10,0.205742251,3.01,0
5/31/10,0.201225672,3.16,0
6/30/10,0.193519414,3.09,0
7/31/10,0.212375965,2.94,0
8/31/10,0.204737848,2.8,0
9/30/10,0.200150902,2.67,0
10/31/10,0.222857537,2.37,0
11/30/10,0.212916915,2.31,0
12/31/10,0.21313075,2.19,0
1/31/11,0.221508599,2.13,0
2/28/11,0.233429812,2.14,0
3/31/11,0.237388076,2.1,0
4/30/11,0.233502902,2.13,0
5/31/11,0.229447853,2.18,0
6/30/11,0.247457384,2.14,0
7/31/11,0.263242026,2.23,0
8/31/11,0.240917644,2.31,0
9/30/11,0.270576427,2.36,0
10/31/11,0.266220391,2.02,0
11/30/11,0.268004811,2.11,0
12/31/11,0.287787367,2.3,0
1/31/12,0.305462218,2.45,0
2/29/12,0.282213377,2.46,0
3/31/12,0.312986707,2.4,0
4/30/12,0.299936739,2.35,0
5/31/12,0.302922591,2.22,0
6/30/12,0.308146864,2.33,0
7/31/12,0.295116941,2.31,0
8/31/12,0.305428594,2.2,0
9/30/12,0.32042922,2.2,0
10/31/12,0.311365886,2.56,0
11/30/12,0.32303207,2.45,0
12/31/12,0.322816718,2.37,0
1/31/13,0.314569802,2.22,0
2/28/13,0.335062762,2.2,0
3/31/13,0.34870391,2.21,0
4/30/13,0.339115646,2.06,0
5/31/13,0.355671872,2.1,0
6/30/13,0.353161433,2.14,0
7/31/13,0.342743714,2.2,0
8/31/13,0.362177498,2.22,0
9/30/13,0.366282165,2.26,0
10/31/13,0.379130747,2.2,0
11/30/13,0.381848521,2.28,0
12/31/13,0.396097655,2.31,0
1/31/14,0.404528524,2.21,0
2/28/14,0.422552904,2.09,0
3/31/14,0.422736031,2.09,0
4/30/14,0.470624613,2.27,0
5/31/14,0.481488995,2.32,0
6/30/14,0.526638478,2.18,0
7/31/14,0.504371357,2.22,0
8/31/14,0.557245546,2.16,0
9/30/14,0.530554956,2.12,0
10/31/14,0.557508343,1.94,0
11/30/14,0.532783278,1.91,0
12/31/14,0.58850522,1.87,0
1/31/15,0.60146314,1.69,0
2/28/15,0.635655309,1.67,0
3/31/15,0.611861421,1.6,0
4/30/15,0.654736842,1.57,0
5/31/15,0.629726058,1.48,0
6/30/15,0.636352613,1.55,0
7/31/15,0.741520754,1.47,0
8/31/15,0.684059059,1.52,0
9/30/15,0.694068547,1.51,0
10/31/15,0.728730119,1.5,0
11/30/15,0.7135,1.46,0
12/31/15,0.739218414,1.4,0
1/31/16,0.788252262,1.71,0
2/29/16,0.749156063,1.85,0
3/31/16,0.769878156,1.9,0
4/30/16,0.71935044,1.94,0
5/31/16,0.754966022,2.07,0
6/30/16,0.741477273,2.08,0
7/31/16,0.780264363,2.16,0
8/31/16,0.729316547,2.2,0
9/30/16,0.737834779,2.16,0
10/31/16,0.715785431,2.27,0
11/30/16,0.790546803,2.28,0
12/31/16,0.792979657,2.37,0
1/31/17,0.752142475,2.38,0
2/28/17,0.80268329,2.41,0
3/31/17,0.821575004,2.09,0
4/30/17,0.859218508,2.18,0
5/31/17,0.832285714,2.06,0
6/30/17,0.917357777,2.04,0
7/31/17,0.905107371,1.86,0
8/31/17,0.886190342,1.85,0
9/30/17,0.922089291,2,0
10/31/17,0.95641791,2.08,0
11/30/17,0.925745497,2.06,0
12/31/17,0.955367913,2.09,0
1/31/18,1.020342117,2.13,0
2/28/18,0.995593375,2.21,0
3/31/18,1.053461063,2.58,0
4/30/18,1.0647159,2.38,0
5/31/18,1.132343447,2.44,0
6/30/18,1.121451838,2.48,0
7/31/18,1.160613398,2.61,0
8/31/18,1.170890188,2.54,0
9/30/18,1.220319447,2.59,0
10/31/18,1.175978103,2.47,0
11/30/18,1.241864268,2.55,0
12/31/18,1.172170919,2.64,0
1/31/19,1.158610039,2.31,0
2/28/19,1.151564537,2.18,0
3/31/19,1.179780716,2.04,0
4/30/19,1.219929542,2.18,0
5/31/19,1.226812816,2.05,0
6/30/19,1.212131424,2.02,0
7/31/19,1.161854479,2.03,0
8/31/19,1.207401177,2.06,0
9/30/19,1.238310447,1.89,0
10/31/19,1.241526145,1.94,0
11/30/19,1.173824131,1.81,0
12/31/19,1.144541261,1.87,0
1/31/20,1.213835406,2.05,0
2/29/20,1.210846515,2.09,100
3/31/20,0.817929503,1.89,100
4/30/20,0.199532143,1.28,100
5/31/20,0.26814468,1.44,0
6/30/20,0.34721278,1.52,0
7/31/20,0.39933953,1.62,0
8/31/20,0.477601961,1.77,0
9/30/20,0.518639477,1.98,0
10/31/20,0.613981214,1.89,0
11/30/20,0.637740789,1.88,0
12/31/20,0.628876509,1.97,0
1/31/21,0.703790543,2.19,0
2/28/21,0.775587794,2.41,0
3/31/21,0.868183682,3.1,0
4/30/21,0.958410162,3.98,0
5/31/21,1.068961796,4.1,0
6/30/21,1.088353414,4.24,0
7/31/21,1.26549078,4.49,0
8/31/21,1.323827936,4.53,0
9/30/21,1.421719812,4.38,0
10/31/21,1.558043687,4.57,0
11/30/21,1.647015257,5.12,0
12/31/21,1.784463895,5.26,0
1/31/22,1.725737101,5.13,0
2/28/22,1.850658626,5.11,0
3/31/22,2.01762554,5.08,0
4/30/22,1.979506831,5.03,0
5/31/22,1.919271269,4.94,0
6/30/22,1.891446816,5.05,0
7/31/22,2.012829404,4.68,0
8/31/22,1.714478797,4.89,0
9/30/22,1.866712707,5.17,0
10/31/22,1.758551308,5.35,0
11/30/22,1.786808009,5.16,0
12/31/22,1.886229622,4.97,0
1/31/23,1.808421785,5.04,0
2/28/23,1.651459242,5.03,0
3/31/23,1.639321802,4.79,0
4/30/23,1.737436967,4.78,0
5/31/23,1.528986697,4.71,0
6/30/23,1.542631756,4.53,0
7/31/23,1.453716216,4.68,0
8/31/23,1.477880331,4.23,0
9/30/23,1.466803667,4.21,0
10/31/23,1.319648094,3.83,0
11/30/23,1.385193476,3.64,0
12/31/23,1.359461599,3.55,0
1/31/24,1.377134493,3.88,0
2/29/24,1.306870938,3.74,0
3/31/24,1.245651839,3.91,0
4/30/24,1.173598275,3.76,0
5/31/24,1.19080633,3.61,0
6/30/24,1.082201781,3.53,0
7/31/24,1.057348175,3.47,0
8/31/24,1.081742328,3.6,0
9/30/24,1.02927112,3.53,0
10/31/24,1.092226047,3.85,0
11/30/24,1.127791041,3.81,0
12/31/24,1.090328202,3.86,0
1/31/25,1.133304132,3.35,0
2/28/25,1.060692002,3.69,0
3/31/25,1.016518424,3.35,0
4/30/25,1.031956461,3.17,0
5/31/25,1.065634932,3.31,0
6/30/25,1.048752673,3.2,0
7/31/25,0.996130459,3.31,0
8/31/25,0.978737811,3.41,0
9/30/25,1.007233987,3.26,0
10/31/25,0.965271479,,0
11/30/25,0.951219512,,0"""

    from io import StringIO
    from plotly.subplots import make_subplots
    
    df = pd.read_csv(StringIO(data))
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig.add_trace(
        go.Scatter(
            x=df.index, y=df['vu_ratio'],
            name='V/U Ratio', mode='lines',
            line=dict(color='#003865', width=2.5)
        ),
        secondary_y=False
    )
    
    fig.add_trace(
        go.Scatter(
            x=df.index, y=df['non_housing_services'],
            name='Non-Housing Services Inflation YoY', mode='lines',
            line=dict(color='#00B6B2', width=2.5)
        ),
        secondary_y=True
    )
    
    recession_periods = df[df['recessions'] == 100].index
    if len(recession_periods) > 0:
        start = recession_periods[0]
        end = recession_periods[-1]
        fig.add_vrect(
            x0=start, x1=end,
            fillcolor='rgba(128,128,128,0.2)',
            layer='below', line_width=0
        )
    
    fig.update_layout(
        height=450,
        hovermode='x unified',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0),
        margin=dict(l=60, r=60, t=80, b=60)
    )
    
    fig.update_xaxes(dtick='M12', tickformat='%Y')
    fig.update_yaxes(title_text='Ratio', tickfont=dict(size=10), title_font=dict(size=12), secondary_y=False)
    fig.update_yaxes(title_text='% Change Year-Over-Year', ticksuffix='%', tickfont=dict(size=10), title_font=dict(size=12), secondary_y=True)
    
    return fig

# --- MAIN APP ---
st.title("Choose Your Own Inflation Adventure")
st.markdown("""
Inflation drives the path for rates. This tool allows you to stress-test and forecast core PCE (the Fed's preferred gauge) by setting your own assumptions for each component.

**How it works:**  
Set monthly run-rates for housing, core services ex. housing, and core goods on the sidebar (top-left in mobile). The model will determine the trajectory of the core PCE YoY rate on a go forward basis.

---
""")

df_mom, df_yoy = fetch_pce_data()

# Sidebar
st.sidebar.header("Assumed Monthly Inflation Run-Rate (% Chg. MoM)")
st.sidebar.caption("""\
Defaults to the trailing 3 month average.
Sliders are bounded by historical mix/max.""")

housing_pace = st.sidebar.slider("🏠 Housing", 0.06, 0.75, 0.26, 0.01, format="%.2f")
non_housing_pace = st.sidebar.slider("💼 Services ex. Housing", 0.11, 0.61, 0.29, 0.01, format="%.2f")
goods_pace = st.sidebar.slider("📦 Core Goods", -0.16, 0.61, 0.02, 0.01, format="%.2f")

st.sidebar.markdown("---")
st.sidebar.caption(f"""Weights:
Housing {WEIGHTS['housing']*100:.1f}%
Non-Housing {WEIGHTS['non_housing_services']*100:.1f}%
Goods {WEIGHTS['core_goods']*100:.1f}%""")

st.sidebar.caption("""Pre-pandemic avg:
Housing 0.27%
Non-Housing 0.17%
Goods -0.06%""")
st.sidebar.caption(f"Data through: {df_mom.index[-1].strftime('%B %Y')}")

# Generate forecast
forecast = generate_forecast(df_mom, housing_pace, non_housing_pace, goods_pace)
yoy_path = calculate_yoy_path(df_mom, forecast)

# Metrics
current_yoy = df_yoy['core_pce'].iloc[-1]
final_yoy = yoy_path['total_yoy'].iloc[-1]
implied_monthly = forecast['total_mom'].iloc[0]
implied_annual = forecast['annualized'].iloc[0]

col1, col2 = st.columns(2)
col1.metric("Current Core PCE YoY", f"{current_yoy:.2f}%")
col2.metric("12-Month Forecast YoY", f"{final_yoy:.2f}%", f"{final_yoy - current_yoy:.2f}pp", delta_color="inverse")
st.markdown("---")
st.markdown("**Core PCE Inflation Rate: Trailing 2 Years and Forecast for Next 12 Months**")
# Charts
st.plotly_chart(create_yoy_chart(df_yoy, yoy_path), use_container_width=True)

# Appendix
with st.expander("Appendix: Background Charts"):
    st.markdown("**V/U Ratio vs Non-Housing Services Inflation**")
    st.plotly_chart(create_vu_ratio_chart(), use_container_width=True)