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
    fig.update_yaxes(title_text='Ratio', tickfont=dict(size=10), title_font=dict(size=12), secondary_y=False, showgrid=False)
    fig.update_yaxes(title_text='% Change Year-Over-Year', ticksuffix='%', tickfont=dict(size=10), title_font=dict(size=12), secondary_y=True, showgrid=False)
    
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
    fig.update_yaxes(title_text='% Change Year-Over-Year', ticksuffix='%', tickfont=dict(size=10), title_font=dict(size=12), secondary_y=False, showgrid=False)
    fig.update_yaxes(title_text='Index', tickfont=dict(size=10), title_font=dict(size=12), secondary_y=True, showgrid=False)
    
    return fig

def create_housing_chart():
    # Quarterly data - New Tenant Rent Index and All Tenant Repeat Rent Index
    quarterly_data = """date,new_tenant_rent,all_tenant_rent
3/31/05,0.0012086,0.001970177
6/30/05,0.002019093,0.002173821
9/30/05,0.002320162,0.002393245
12/31/05,0.002975809,0.002547339
3/31/06,0.003858711,0.002822438
6/30/06,0.002903192,0.002838594
9/30/06,0.003402274,0.002765868
12/31/06,0.003225493,0.002676901
3/31/07,0.002587847,0.002336408
6/30/07,0.003112817,0.00223889
9/30/07,0.002790116,0.001953866
12/31/07,0.00234453,0.001823271
3/31/08,0.001896754,0.001880429
6/30/08,0.001700669,0.001618842
9/30/08,0.001405747,0.00150416
12/31/08,0.000457182,0.001561519
3/31/09,-0.000359041,0.001331867
6/30/09,-0.002569332,0.001659764
9/30/09,-0.002509344,0.001413952
12/31/09,-0.00287847,0.001651581
3/31/10,-0.001835075,0.001545135
6/30/10,0.000149876,0.00158609
9/30/10,0.00044889,0.001676128
12/31/10,0.001831439,0.001667947
3/31/11,0.001667947,0.001970177
6/30/11,0.002149407,0.002198227
9/30/11,0.002604045,0.002482489
12/31/11,0.002214495,0.002693083
3/31/12,0.002741613,0.002555442
6/30/12,0.002409478,0.00237701
9/30/12,0.002279536,0.002587847
12/31/12,0.001872266,0.002181957
3/31/13,0.002108704,0.002441935
6/30/13,0.002352651,0.002425708
9/30/13,0.001880429,0.002157546
12/31/13,0.002409478,0.001962022
3/31/14,0.002450047,0.001545135
6/30/14,0.002076128,0.000655962
9/30/14,0.003048368,0.00019978
12/31/14,0.002749699,-0.000308857
3/31/15,0.002903192,-0.000576826
6/30/15,0.003169172,-0.000610378
9/30/15,0.003690816,-0.000191869
12/31/15,0.003522611,0.000108269
3/31/16,0.002506814,0.000556626
6/30/16,0.0032657,0.0012086
9/30/16,0.002027243,0.001454967
12/31/16,0.001864102,0.001627028
3/31/17,0.002612143,0.001782422
6/30/17,0.002830517,0.001880429
9/30/17,0.002198227,0.001904915
12/31/17,0.003008064,0.001782422
3/31/18,0.002181957,0.001888592
6/30/18,0.001978332,0.002002791
9/30/18,0.002935473,0.001921235
12/31/18,0.001798764,0.00237701
3/31/19,0.003024188,0.002222627
6/30/19,0.002425708,0.002247021
9/30/19,0.002991938,0.002360771
12/31/19,0.003241578,0.002409478
3/31/20,0.002838594,0.002482489
6/30/20,0.001446765,0.002612143
9/30/20,0.000299506,0.002401362
12/31/20,0.001397541,0.002587847
3/31/21,0.001068696,0.002441935
6/30/21,0.003410301,0.002595947
9/30/21,0.005740259,0.002555442
12/31/21,0.008734594,0.002523026
3/31/22,0.009293297,0.002458159
6/30/22,0.009833654,0.002303914
9/30/22,0.008065729,0.002433822
12/31/22,0.004193578,0.002198227
3/31/23,0.003650795,0.002450047
6/30/23,0.002668809,0.00246627
9/30/23,0.002401362,0.002531131
12/31/23,0.001872266,0.00234453
3/31/24,0.002206361,0.002676901
6/30/24,0.001872266,0.002295788
9/30/24,0.001454967,0.002644528
12/31/24,0.000804762,0.002701173
3/31/25,0.000870817,0.002401362
6/30/25,-0.008083183,0.00237701"""

    # Monthly PCE Housing data
    monthly_data = """date,pce_housing
1/31/05,0.002095028
2/28/05,0.002522833
3/31/05,0.002476913
4/30/05,0.002118032
5/31/05,0.001959047
6/30/05,0.001833862
7/31/05,0.002082191
8/31/05,0.001920767
9/30/05,0.002163647
10/31/05,0.002006777
11/30/05,0.002288129
12/31/05,0.002211868
1/31/06,0.002504425
2/28/06,0.002634868
3/31/06,0.003061144
4/30/06,0.00334229
5/31/06,0.003813465
6/30/06,0.004201469
7/31/06,0.004220842
8/31/06,0.003904302
9/30/06,0.003447796
10/31/06,0.003439306
11/30/06,0.003290642
12/31/06,0.003314482
1/31/07,0.003009128
2/28/07,0.003033968
3/31/07,0.002838453
4/30/07,0.002514635
5/31/07,0.001947627
6/30/07,0.001749385
7/31/07,0.001846784
8/31/07,0.002245648
9/30/07,0.002495535
10/31/07,0.002694096
11/30/07,0.002931872
12/31/07,0.002932045
1/31/08,0.002718006
2/29/08,0.002198871
3/31/08,0.001942249
4/30/08,0.001999737
5/31/08,0.001828563
6/30/08,0.002004634
7/31/08,0.001886082
8/31/08,0.002078959
9/30/08,0.002029054
10/31/08,0.001976653
11/30/08,0.002231811
12/31/08,0.001726497
1/31/09,0.001876625
2/28/09,0.001461747
3/31/09,0.001861949
4/30/09,0.001637561
5/31/09,0.001719384
6/30/09,0.001116724
7/31/09,0.000571999
8/31/09,0.000247617
9/30/09,-0.000167523
10/31/09,-0.000255436
11/30/09,-0.000822798
12/31/09,-0.000551541
1/31/10,-0.000583773
2/28/10,-0.000272128
3/31/10,-0.000368276
4/30/10,-0.000196196
5/31/10,4.41318E-05
6/30/10,0.000432512
7/31/10,0.000580534
8/31/10,0.000396253
9/30/10,0.000387974
10/31/10,0.000399781
11/30/10,0.000847129
12/31/10,0.001006103
1/31/11,0.001300428
2/28/11,0.001350394
3/31/11,0.001324995
4/30/11,0.001196174
5/31/11,0.001115091
6/30/11,0.00128414
7/31/11,0.001670267
8/31/11,0.002011352
9/30/11,0.001964871
10/31/11,0.001956717
11/30/11,0.001732658
12/31/11,0.001902422
1/31/12,0.001808503
2/29/12,0.001782196
3/31/12,0.001821551
4/30/12,0.001958097
5/31/12,0.001912454
6/30/12,0.001602593
7/31/12,0.001421066
8/31/12,0.001627586
9/30/12,0.001991783
10/31/12,0.002161387
11/30/12,0.002041965
12/31/12,0.001838256
1/31/13,0.00168506
2/28/13,0.001827316
3/31/13,0.001881479
4/30/13,0.001954209
5/31/13,0.001950152
6/30/13,0.002029974
7/31/13,0.001855907
8/31/13,0.002059617
9/30/13,0.001995232
10/31/13,0.002206444
11/30/13,0.002118082
12/31/13,0.002443282
1/31/14,0.002475476
2/28/14,0.002383939
3/31/14,0.002321564
4/30/14,0.002394265
5/31/14,0.002433489
6/30/14,0.002283371
7/31/14,0.002310993
8/31/14,0.002269104
9/30/14,0.002322759
10/31/14,0.002254902
11/30/14,0.002421382
12/31/14,0.002284901
1/31/15,0.002413911
2/28/15,0.002498028
3/31/15,0.002767395
4/30/15,0.002792406
5/31/15,0.002622942
6/30/15,0.002809074
7/31/15,0.002851331
8/31/15,0.002858816
9/30/15,0.002718145
10/31/15,0.002547591
11/30/15,0.002474433
12/31/15,0.002322935
1/31/16,0.002475589
2/29/16,0.002627897
3/31/16,0.002632056
4/30/16,0.002666495
5/31/16,0.002906305
6/30/16,0.003114078
7/31/16,0.003087745
8/31/16,0.002888011
9/30/16,0.003047536
10/31/16,0.003151851
11/30/16,0.003334271
12/31/16,0.003142335
1/31/17,0.0029392
2/28/17,0.002740153
3/31/17,0.00249937
4/30/17,0.002375238
5/31/17,0.002281528
6/30/17,0.002500885
7/31/17,0.002649104
8/31/17,0.002979174
9/30/17,0.002907523
10/31/17,0.003072114
11/30/17,0.002833919
12/31/17,0.003093017
1/31/18,0.002991572
2/28/18,0.002721821
3/31/18,0.002568794
4/30/18,0.002688564
5/31/18,0.002907415
6/30/18,0.002688352
7/31/18,0.002621737
8/31/18,0.002763194
9/30/18,0.0027054
10/31/18,0.002652551
11/30/18,0.002743724
12/31/18,0.002814226
1/31/19,0.002726937
2/28/19,0.002645256
3/31/19,0.002942243
4/30/19,0.003243328
5/31/19,0.003089316
6/30/19,0.00306363
7/31/19,0.002775445
8/31/19,0.002765319
9/30/19,0.002675839
10/31/19,0.002443015
11/30/19,0.002483245
12/31/19,0.002352785
1/31/20,0.002818535
2/29/20,0.00270089
3/31/20,0.002715479
4/30/20,0.002279842
5/31/20,0.002280754
6/30/20,0.001780161
7/31/20,0.001843345
8/31/20,0.00158502
9/30/20,0.001477593
10/31/20,0.001420056
11/30/20,0.001077504
12/31/20,0.00127844
1/31/21,0.001053367
2/28/21,0.001656478
3/31/21,0.001956858
4/30/21,0.002295071
5/31/21,0.002462827
6/30/21,0.002760663
7/31/21,0.002938056
8/31/21,0.002997898
9/30/21,0.003256742
10/31/21,0.003778873
11/30/21,0.004223731
12/31/21,0.004213108
1/31/22,0.004223554
2/28/22,0.004393134
3/31/22,0.004663428
4/30/22,0.004900088
5/31/22,0.005277778
6/30/22,0.006138219
7/31/22,0.006626114
8/31/22,0.007061274
9/30/22,0.007171111
10/31/22,0.007137877
11/30/22,0.007013136
12/31/22,0.00699842
1/31/23,0.006964015
2/28/23,0.0069943
3/31/23,0.006140287
4/30/23,0.005903092
5/31/23,0.005287199
6/30/23,0.005250536
7/31/23,0.004950312
8/31/23,0.004630172
9/30/23,0.004674703
10/31/23,0.004524119
11/30/23,0.004814399
12/31/23,0.004538896
1/31/24,0.004673523
2/29/24,0.004468946
3/31/24,0.004535539
4/30/24,0.004306908
5/31/24,0.00421099
6/30/24,0.003734666
7/31/24,0.003652851
8/31/24,0.003746044
9/30/24,0.003854907
10/31/24,0.003854818
11/30/24,0.003223098
12/31/24,0.003151913
1/31/25,0.002939272
2/28/25,0.003035507
3/31/25,0.003292825
4/30/25,0.003394787
5/31/25,0.003325719
6/30/25,0.002984741
7/31/25,0.002720537
8/31/25,0.003061196
9/30/25,0.002616571"""

    from io import StringIO
    
    # Load quarterly data
    df_q = pd.read_csv(StringIO(quarterly_data))
    df_q['date'] = pd.to_datetime(df_q['date'])
    df_q = df_q.set_index('date')
    # Convert to percentage
    df_q['new_tenant_rent'] = df_q['new_tenant_rent'] * 100
    df_q['all_tenant_rent'] = df_q['all_tenant_rent'] * 100
    
    # Load monthly data
    df_m = pd.read_csv(StringIO(monthly_data))
    df_m['date'] = pd.to_datetime(df_m['date'])
    df_m = df_m.set_index('date')
    # Convert to percentage
    df_m['pce_housing'] = df_m['pce_housing'] * 100
    
    # Filter from 2010 onwards
    df_q = df_q[df_q.index >= '2010-01-01']
    df_m = df_m[df_m.index >= '2010-01-01']
    
    fig = go.Figure()
    
    # New Tenant Rent Index (quarterly)
    fig.add_trace(
        go.Scatter(
            x=df_q.index, y=df_q['new_tenant_rent'],
            name='New Tenant Rent Index', mode='lines',
            line=dict(color='#D29F2A', width=2.5)
        )
    )
    
    # PCE Housing (monthly)
    fig.add_trace(
        go.Scatter(
            x=df_m.index, y=df_m['pce_housing'],
            name='PCE Housing', mode='lines',
            line=dict(color='#00B6B2', width=2.5)
        )
    )
    
    fig.update_layout(
        height=450,
        hovermode='x unified',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0),
        margin=dict(l=60, r=40, t=80, b=60),
        yaxis_title='% Change'
    )
    
    fig.update_xaxes(dtick='M12', tickformat='%Y')
    fig.update_yaxes(ticksuffix='%', tickfont=dict(size=10), title_font=dict(size=12), showgrid=False)
    
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

# Preset dropdown
preset = st.sidebar.selectbox(
    "Base Rate Scenario",
    ["Trailing 3-Month Avg", "Pre-Pandemic (2014-19)", "Fed Target (~2% YoY)", "Custom"],
    index=0
)

# Define presets
presets = {
    "Trailing 3-Month Avg": {"housing": 0.26, "non_housing": 0.29, "goods": 0.02},
    "Pre-Pandemic (2014-19)": {"housing": 0.27, "non_housing": 0.17, "goods": -0.06},
    "Fed Target (~2% YoY)": {"housing": 0.17, "non_housing": 0.17, "goods": 0.00},
    "Custom": {"housing": 0.20, "non_housing": 0.20, "goods": 0.00}
}

defaults = presets[preset]

st.sidebar.caption("Select a scenario above, then adjust sliders to customize.")

# Sliders
housing_pace = st.sidebar.slider("🏠 Housing", 0.06, 0.75, defaults["housing"], 0.01, format="%.2f")
non_housing_pace = st.sidebar.slider("💼 Services ex. Housing", 0.11, 0.61, defaults["non_housing"], 0.01, format="%.2f")
goods_pace = st.sidebar.slider("📦 Core Goods", -0.16, 0.61, defaults["goods"], 0.01, format="%.2f")

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
    st.markdown("**Housing Rent Indices vs PCE Housing**")
    st.plotly_chart(create_housing_chart(), use_container_width=True)

    st.markdown("**V/U Ratio vs Non-Housing Services Inflation**")
    st.plotly_chart(create_vu_ratio_chart(), use_container_width=True)

    st.markdown("**Core Goods Inflation vs Supply Chain Stress Index**")
    st.plotly_chart(create_core_goods_chart(), use_container_width=True)