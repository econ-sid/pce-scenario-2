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
        title=dict(text='<b>Core PCE YoY: History + Forecast</b>', y=0.95),
        height=450,
        hovermode='x unified',
        yaxis_title='Contribution (pp)',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5),
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

# --- MAIN APP ---
st.title("Choose Your Own Inflation Adventure")
st.markdown(""" 
            
Since inflation is a key to the path for interest rates, we have built this tool where you can build your own forecast for Core PCE inflation. 

**How it works:**  
If you choose the monthly pace (top left sidebar) for the three components of core PCE inflation the model will determine the forecast for the next twelve months.

---
""")

df_mom, df_yoy = fetch_pce_data()

# Sidebar
st.sidebar.header("Monthly Pace Assumptions (% MoM)")
st.sidebar.caption("""\
By default the sliders are set for the current 3-month average.
The range is constrained by the historic min/max.""")

housing_pace = st.sidebar.slider("🏠 Housing", 0.06, 0.75, 0.26, 0.01, format="%.2f")
non_housing_pace = st.sidebar.slider("💼 Non-Housing Services", 0.11, 0.61, 0.29, 0.01, format="%.2f")
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

# Charts
st.plotly_chart(create_yoy_chart(df_yoy, yoy_path), use_container_width=True)