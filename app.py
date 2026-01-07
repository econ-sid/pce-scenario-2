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

def create_core_goods_chart():
    data = """date,core_goods,supply_chain_stress
1/31/10,1.410836319,-0.57
2/28/10,0.971517753,-0.36
3/31/10,0.545986224,-0.03
4/30/10,-0.280277184,0.17
5/31/10,-0.413754435,0.37
6/30/10,-0.609561715,0.24
7/31/10,-0.673368098,0.12
8/31/10,-0.513322163,0.08
9/30/10,-0.839776446,0.21
10/31/10,-1.085353511,0.45
11/30/10,-0.979957175,0.50
12/31/10,-1.048853209,0.62
1/31/11,-0.781046351,0.65
2/28/11,-0.572858302,0.62
3/31/11,-0.600028987,0.63
4/30/11,-0.03586001,0.91
5/31/11,0.336844149,1.10
6/30/11,0.599692864,0.92
7/31/11,0.880533775,0.48
8/31/11,0.928747359,0.13
9/30/11,0.815738496,-0.12
10/31/11,1.025151288,-0.34
11/30/11,1.144532544,-0.26
12/31/11,1.254997562,-0.09
1/31/12,1.218254354,0.16
2/29/12,1.025038135,0.12
3/31/12,1.193693256,0.02
4/30/12,0.876461577,-0.22
5/31/12,0.624987907,-0.45
6/30/12,0.371969895,-0.56
7/31/12,0.301248443,-0.69
8/31/12,-0.005787428,-0.51
9/30/12,0.141937104,-0.36
10/31/12,0.132148817,-0.13
11/30/12,-0.054893728,-0.20
12/31/12,-0.234020628,-0.17
1/31/13,-0.302820557,-0.18
2/28/13,-0.390467214,-0.19
3/31/13,-0.724289639,-0.32
4/30/13,-0.764085116,-0.55
5/31/13,-0.84128144,-0.71
6/30/13,-0.719043585,-0.75
7/31/13,-0.734494277,-0.72
8/31/13,-0.634724647,-0.63
9/30/13,-0.727963438,-0.50
10/31/13,-0.859278668,-0.34
11/30/13,-0.867219117,-0.38
12/31/13,-0.880359866,-0.45
1/31/14,-1.050073766,-0.60
2/28/14,-1.045649403,-0.47
3/31/14,-0.946317297,-0.51
4/30/14,-0.830984688,-0.57
5/31/14,-0.749517613,-0.76
6/30/14,-0.684499859,-0.78
7/31/14,-0.642952734,-0.78
8/31/14,-0.766923279,-0.71
9/30/14,-0.640060607,-0.76
10/31/14,-0.583971394,-0.68
11/30/14,-0.83592535,-0.79
12/31/14,-0.834615609,-0.65
1/31/15,-1.031008205,-0.63
2/28/15,-0.775693001,-0.41
3/31/15,-0.59490085,-0.41
4/30/15,-0.510777104,-0.36
5/31/15,-0.504103165,-0.43
6/30/15,-0.734124079,-0.59
7/31/15,-0.893075009,-0.61
8/31/15,-0.826656492,-0.65
9/30/15,-0.780058651,-0.50
10/31/15,-0.974441675,-0.44
11/30/15,-0.764555969,-0.43
12/31/15,-0.703167199,-0.49
1/31/16,-0.544505711,-0.68
2/29/16,-0.548704938,-0.72
3/31/16,-0.720315248,-0.74
4/30/16,-0.607637185,-0.56
5/31/16,-0.775696163,-0.55
6/30/16,-0.844782314,-0.43
7/31/16,-0.845972031,-0.39
8/31/16,-0.587921952,-0.12
9/30/16,-0.583239739,-0.12
10/31/16,-0.507313607,-0.08
11/30/16,-0.867246148,-0.22
12/31/16,-0.89902976,-0.19
1/31/17,-0.533626382,-0.12
2/28/17,-0.581395349,0.03
3/31/17,-0.591915113,0.13
4/30/17,-0.926410603,0.08
5/31/17,-0.961861975,-0.02
6/30/17,-0.786519083,-0.01
7/31/17,-0.659508745,0.01
8/31/17,-0.941673778,0.19
9/30/17,-1.267466059,0.35
10/31/17,-0.990040078,0.58
11/30/17,-0.874833104,0.73
12/31/17,-0.837325349,0.77
1/31/18,-0.756459768,0.70
2/28/18,-0.82149819,0.45
3/31/18,-0.666135617,0.36
4/30/18,-0.374826792,0.34
5/31/18,-0.135888571,0.45
6/30/18,-0.357889054,0.45
7/31/18,-0.452923003,0.42
8/31/18,-0.671147663,0.48
9/30/18,-0.490811093,0.49
10/31/18,-0.552071018,0.51
11/30/18,-0.224156648,0.49
12/31/18,-0.25764636,0.49
1/31/19,-0.241705781,0.48
2/28/19,-0.435209883,0.38
3/31/19,-0.62148535,0.30
4/30/19,-0.884557271,0.13
5/31/19,-0.898485182,-0.13
6/30/19,-0.498630522,-0.36
7/31/19,-0.57550922,-0.51
8/31/19,-0.162365494,-0.40
9/30/19,-0.265276069,-0.20
10/31/19,-0.337514483,-0.02
11/30/19,-0.493648059,0.13
12/31/19,-0.459109026,0.09
1/31/20,-0.795238622,0.08
2/29/20,-0.455241318,0.47
3/31/20,-0.649579891,1.35
4/30/20,-1.25589331,2.45
5/31/20,-1.534609482,2.87
6/30/20,-1.673792046,2.73
7/31/20,-0.953622046,2.60
8/31/20,-0.409099174,2.24
9/30/20,-0.408580184,1.70
10/31/20,-0.457945815,0.75
11/30/20,-0.157941096,0.51
12/31/20,0.096300051,0.87
1/31/21,0.375977948,1.29
2/28/21,0.07284795,1.70
3/31/21,0.570575754,1.90
4/30/21,2.435358709,2.38
5/31/21,3.658436552,2.75
6/30/21,4.30698552,2.92
7/31/21,4.02154069,2.98
8/31/21,3.919142333,3.07
9/30/21,4.153338411,3.28
10/31/21,5.208851697,3.61
11/30/21,5.591441464,3.95
12/31/21,6.23120158,4.29
1/31/22,6.901773905,4.22
2/28/22,7.662676427,3.70
3/31/22,7.480390474,3.17
4/30/22,6.351867932,3.12
5/31/22,5.72228377,3.10
6/30/22,5.719847418,2.95
7/31/22,5.503480733,2.37
8/31/22,5.597470158,1.95
9/30/22,5.730024862,1.48
10/31/22,4.688404958,1.25
11/30/22,3.930701342,1.16
12/31/22,3.173559076,1.26
1/31/23,2.674650322,1.24
2/28/23,2.282907773,0.74
3/31/23,2.511529178,-0.10
4/30/23,2.50387124,-0.89
5/31/23,2.500912229,-1.35
6/30/23,1.698997526,-1.33
7/31/23,1.183052517,-1.21
8/31/23,0.56011535,-1.04
9/30/23,0.231460135,-0.89
10/31/23,0.380810121,-0.70
11/30/23,0.224538449,-0.28
12/31/23,-0.037883338,-0.12
1/31/24,-0.523386837,-0.11
2/29/24,-0.396628656,-0.23
3/31/24,-0.499344896,-0.29
4/30/24,-0.517290314,-0.50
5/31/24,-1.021414097,-0.63
6/30/24,-0.795530399,-0.62
7/31/24,-0.414232705,-0.32
8/31/24,-0.471516021,-0.06
9/30/24,-0.333968149,0.09
10/31/24,-0.432641964,-0.01
11/30/24,-0.185313235,-0.17
12/31/24,-0.203353484,-0.30
1/31/25,0.232093651,-0.24
2/28/25,0.329996497,-0.14
3/31/25,-0.036833091,-0.12
4/30/25,0.195107585,-0.14
5/31/25,0.64923687,-0.05
6/30/25,0.993630808,0.04
7/31/25,1.05418492,0.13
8/31/25,1.07309415,0.01
9/30/25,1.210191083,-0.03"""

    from io import StringIO
    from plotly.subplots import make_subplots
    
    df = pd.read_csv(StringIO(data))
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig.add_trace(
        go.Scatter(
            x=df.index, y=df['core_goods'],
            name='Core Goods Inflation YoY', mode='lines',
            line=dict(color='#D29F2A', width=2.5)
        ),
        secondary_y=False
    )
    
    fig.add_trace(
        go.Scatter(
            x=df.index, y=df['supply_chain_stress'],
            name='Supply Chain Stress Index (3mo MA)', mode='lines',
            line=dict(color='#003865', width=2.5)
        ),
        secondary_y=True
    )
    
    fig.update_layout(
        height=450,
        hovermode='x unified',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0),
        margin=dict(l=60, r=60, t=80, b=60)
    )
    
    fig.update_xaxes(dtick='M12', tickformat='%Y')
    fig.update_yaxes(title_text='% Change Year-Over-Year', ticksuffix='%', tickfont=dict(size=10), title_font=dict(size=12), secondary_y=False)
    fig.update_yaxes(title_text='Index', tickfont=dict(size=10), title_font=dict(size=12), secondary_y=True)
    
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

    st.markdown("**Core Goods Inflation vs Supply Chain Stress Index**")
    st.plotly_chart(create_core_goods_chart(), use_container_width=True)