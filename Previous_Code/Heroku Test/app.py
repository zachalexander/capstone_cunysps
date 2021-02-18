import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import pandas as pd
import numpy as np
import math
from modsim import *

#From https://callmepower.com/faq/electricity-prices/electricity-prices-per-sq-ft
cost = pd.read_csv("https://raw.githubusercontent.com/mkollontai/DATA608/master/Final/Monthly_cost_per_sf_States.csv", sep = ',', header = 0)
cost.columns = ['State','kWh per Month','Price per kWh', 'Monthly Bill', 'House Sq Ft', 'Monthly Bill per Sq Ft','Region']

States = list(cost['State'])

markdown_text = '''
# Long term financial projection of Solar Panel array  
Solar panel arrays are often marketed by the number of kW they can produce. This number is actually what the system would produce assuming they receive full sunlight for at least 8 hours in a day. In reality the amount of kW they produce depends largely on the sunlight your rooftop is exposed to.  
  
The following calculator is meant to take into account the sunlight in your area and provide estimates of certain panel system sizes (since you are often limited by the size of your roof) as well as their efficiencies.  
  
* Current efficiencies (r) range from 17.5% to slightly above 20%.
* Installation costs (I) can range anywhere from $10,000 to $25,000 depending on various factors. 
'''

app = dash.Dash(__name__)
server = app.server

app.layout = html.Div([
    dcc.Markdown(children = markdown_text),
    html.Div([dcc.Input(id='lat-input', value=29.42412, type='number')," : Latitude"]),
    html.Br(),
    html.Div([dcc.Input(id='long-input', value=-98.49363, type='number')," : Longitude"]),
    html.Br(),
    html.Button(id='get_solar', n_clicks=0, children='Update Solar Data'),    
    html.Br(),
    html.Br(),
    dcc.Dropdown(
        id = 'St',
        options=[{'label':St, 'value':St} for St in States],
        value = 'Texas'
    ),
    html.Br(),
    html.Div(["Home sq. ft. : ",
        dcc.Input(id='home_sf', value=2300, type='number')]),
    html.Br(),
    html.Div(["Roof sq. ft. : ",
        dcc.Input(id='roof_sf', value=850, type='number')]),
    
    html.Br(),
    html.Button(id='submit_details', n_clicks=0, children='Recalculate'),
    html.Br(),
    html.Div([
    dcc.Graph(id = 'US_prices'),
    dcc.Graph(id = 'your-home')
    ], style={'columnCount':2}),
    dcc.Graph(id = 'Projection'),
    html.H3('Fun Facts:'),
    #dcc.Markdown(id='simple_facts'),
    dcc.Markdown(id='calc_facts'),
    #Hidden Div for storing Solar API Data
    html.Div(id='Solar-data', style={'display':'none'})

])

@app.callback(Output('Solar-data', 'children'), 
    Input('get_solar','n_clicks'),
    dash.dependencies.State('lat-input', 'value'),
    dash.dependencies.State('long-input', 'value'))
def collect_ghi(solar_clicks,la, lo):
    api_key = 'BUnBQIpFlpJZcCcqO2VeYuUMXjX7zCSGiVBNIIdH'
    attributes = 'ghi'
    year = '2019'
    lat, lon = la, lo
    leap_year = 'false'
    interval = '60'
    utc = 'false'
    name = 'Misha+Kollontai'
    reason= 'school_project'
    affiliation = 'CUNY+SPS'
    email = 'mkollontai@gmail.com'
    mailing_list = 'false'
    
    #combine all of the relevant information into the API-specified URL
    url = 'https://developer.nrel.gov/api/solar/nsrdb_psm3_download.csv?wkt=POINT({lon}%20{lat})&names={year}&leap_day={leap}&interval={interval}&utc={utc}&full_name={name}&email={email}&affiliation={affiliation}&mailing_list={mailing_list}&reason={reason}&api_key={api}&attributes={attr}'.format(year=year, lat=lat, lon=lon, leap=leap_year, interval=interval, utc=utc, name=name, email=email, mailing_list=mailing_list, affiliation=affiliation, reason=reason, api=api_key, attr=attributes)
    
    GHI_raw = pd.read_csv(url,skiprows = 2)
    #Set the index to the proper timestamps
    GHI_raw = GHI_raw.set_index(pd.date_range('1/1/{yr}'.format(yr=year), freq=interval+'Min', periods=525600/int(interval)))
    temp = GHI_raw[['Month','Day','GHI']]
    daily = temp.groupby(['Month','Day']).sum()
    monthly_mean = daily.groupby(['Month']).mean()
    monthly_sd = daily.groupby(['Month']).std()
    monthly_ghi = pd.DataFrame(monthly_mean)
    monthly_ghi['STD'] = monthly_sd['GHI']
    return monthly_ghi.to_json()

@app.callback(
    Output('US_prices','figure'),
    Input('St', 'value')
)
def adjust_state_fig(st):
    state_fig = px.scatter(cost, x="kWh per Month", y="Price per kWh", color= "Region",
                size = np.where(cost['State'] == st, 5, 1), hover_data=['State'],
                opacity = np.where(cost['State'] == st, 1, 0.5))
    state_fig.update(layout_coloraxis_showscale=False)
    state_fig.update_layout(
        title={
            'text':'Distribution of Energy Use and Prices  in the US by State',
            'x':0.5,
            'xanchor':'center',
            'yanchor':'top'
        },
        xaxis_title='Average Energy Use (kWh/Month)',
        yaxis_range=[0,35],
        yaxis_title='Average price (¢/kWh)',
    )
    return state_fig

@app.callback(
    Output('your-home','figure'),
    [Input('home_sf', 'value'),
    Input('St', 'value')]
)
def home_graph_update(h_sf,st):
    
    us_monthly_use = 887
    avg_home_sf = int(cost.loc[cost['State'] == st, 'House Sq Ft'].iloc[0])
    avg_monthly_use = int(cost.loc[cost['State'] == st, 'kWh per Month'].iloc[0])
    our_monthly_use = int(float(h_sf) / avg_home_sf * avg_monthly_use)
    
    d = {'Location':['US',st,'Your Home'], 
        'Monthly Use':[us_monthly_use,avg_monthly_use,our_monthly_use],
        'Region':['USA',cost.loc[cost['State'] == st, 'Region'].iloc[0],'Home']}
    df = pd.DataFrame(d)
    home_fig = px.bar(df, 
                    x='Location', 
                    y = 'Monthly Use', 
                    color = 'Region',
                    color_discrete_map={
                        "South": "#636EFA",
                        "Northeast": "#AB63FA",
                        "Pacific": "#EF553B",
                        "West": "#00CC96",
                        "Midwest": "#FFA15A",
                        "USA":"#FF6692",
                        "Home":"#19D3F3"},
                    opacity = 0.6)
    home_fig.update_layout(
        title={
            'text':'Monthly Energy Use Comparison',
            'x':0.5,
            'xanchor':'center',
            'yanchor':'top'
        },
        yaxis_range=[0,2000],
        yaxis_title='Average Energy Use (kWh/Month)',
        showlegend=False
    )
    return home_fig

@app.callback(
    [Output('Projection','figure'),
    Output('calc_facts','children')],
    [Input('submit_details','n_clicks'),
    Input('Solar-data', 'children')],
    dash.dependencies.State('lat-input', 'value'),
    dash.dependencies.State('long-input', 'value'),
    dash.dependencies.State('home_sf', 'value'),
    dash.dependencies.State('roof_sf', 'value'),
    dash.dependencies.State('St', 'value')
)

def update_output_div(n_clicks,solar,la, lo, sf, rf, st):    

    monthly_ghi = pd.read_json(solar)
    avg_bill = float(cost.loc[cost['State'] == st, 'Monthly Bill'].iloc[0])
    avg_home_sf = int(cost.loc[cost['State'] == st, 'House Sq Ft'].iloc[0])
    monthly_per_sf =  avg_bill / avg_home_sf
    prices = pd.read_csv("https://raw.githubusercontent.com/mkollontai/DATA608/master/Final/Average_US_Electricity_Price.csv", sep = '\t', header = None)
    prices.columns = ['State','Avg_Rate_2019','Avg_Rate_2018','%_change','Monthly_cost']


    #####   Define a system describing our solar panels and location ############################
    def define_system(A=80,r=0.175,PR=0.8,lat=29.42412,long=-98.49363,state='Texas',initial_cost=20000):
        '''Create a system object defining our solar panel system
        '''
        start = State(P=0, N=0, PB=0, MP = -initial_cost, C = 0)
        t0 = 0
        '''15 years worth of operation'''
        t_end = 15*12
        
        return System(start=start, t0=t0, t_end=t_end, A=A, r=r, PR=PR, state = state, lat=lat, long=long)
    
    #############################################################################################
    ####   We must calculate the amount of power generated on on a given day by the panels. 
    ####   This number is influenced by the surface area of the panels, their efficiency, 
    ####   performance ratio and amount of exposure to sun they receive on that day. In our 
    ####   estimation of GHI on a given day, we will assume a normal distribution given the 
    ####   mean and stDev from the table we pulled from the NSRDB. The formula used below to 
    ####   calculate the actual yield is taken from 
    # (https://photovoltaic-software.com/principle-ressources/how-calculate-solar-energy-power-pv-systems) 
    ####   with the 'Annual average' value replaced with the GHI per day value calculated from the NSRDB data. 
    
    ####   Function to determine the daily yield of the panels   ################################
    ###      system - pre-defined system defining the panels
    ###      month - the month (1-12) for which the GHI is to be estimated
    def days_yield(system,month):
        month = month
        ghi_day = np.random.normal(monthly_ghi.iloc[month-1]['GHI'],monthly_ghi.iloc[month-1]['STD'])
        ghi_day = float(ghi_day)
        if ghi_day < 0:
            ghi_day = 0
        return (system.A*system.r*ghi_day*system.PR)/1000
    #############################################################################################

    ####   Function generating a value for the demand on our system in a month. 
    def month_demand_norm(per_sf = 0.06):
        tot_monthly = per_sf * float(sf)
        std_d = tot_monthly * 0.15
        demand_month = np.random.normal(tot_monthly,std_d)
        if demand_month < 0:
            demand_month = 0
        return demand_month
    #############################################################################################

    ####    Function calculating the balance at the end of a month ##############################
    def calc_month(system, month):
        #2% yearly increase in electricity rates
        yearly_increase = 1.02
        year = math.floor(month % 12)

        month_mod = (month % 12)+1
        if month_mod in [1,3,5,7,8,10,12]:
            days = 31
        elif month_mod in [4,6,9,11]:
            days = 30
        elif month_mod == 2:
            days = 28
        else:
            print("Not a valid month number")
            return None
        loss = month_demand_norm(monthly_per_sf * yearly_increase**year)
        p = 0
        n = 0
        balance = 0
        gain = 0

        price = prices.loc[prices['State'] == system.state, 'Avg_Rate_2019'].iloc[0]
        price = price/100 * yearly_increase**year

        for day in range(1,days+1):
            gain  = gain + days_yield(system,month_mod)
        balance = gain*price - loss
        if balance >= 0:
            p = 1
        else:
            n = 1
        
        this_month = State(P=p, N=n, B=balance, C = loss)
        return this_month
    #############################################################################################

    def update_fxn(state,system,month):
        '''Update the pos/neg/balance model.
        
        state: State with variables P, N, PB, FB, C
        system: System with relevant info
        '''
        p, n, pb, fb, c = state
        
        month_result = calc_month(system, month)
        
        p += month_result.P
        n += month_result.N
        pb += month_result.B
        fb += month_result.B
        c += month_result.C
            
        return State(P=int(p), N=int(n), PB=pb, FB = fb, C = c)

    ####   The function below generates three TimeSeries objects over the time interval specified 
    ####   within the provided time interval. The TimeSeries track number of months with a positive 
    ####   balance, number of months with a negative balance and the overall balance throughout 
    ####   the interval
    def run_simulation(system,upd_fxn):
        """Take a system as input and unpdate it based on the update function.
        
        system - system object defining panels
        update_fxn - function describing change to system 
        
        returns - Timeseries
        """
        P = TimeSeries()
        N = TimeSeries()
        PB = TimeSeries()
        FB = TimeSeries()
        C = TimeSeries()
        
        state = system.start
        t0 = system.t0
        P[t0], N[t0], PB[t0], FB[t0], C[t0] = state
        
        for t in linrange(system.t0, system.t_end):
            state = upd_fxn(state,system,t)
            P[t+1], N[t+1], PB[t+1], FB[t+1], C[t+1] = state
        
        #return P, N, PB, FB, -C
        return FB, -C


    roof_A = float(rf) * 0.092903
    system = define_system(A=roof_A, lat=la, long=lo, state=st, initial_cost = 25000, r =.175)
    FB, C = run_simulation(system,update_fxn)

    system2 = define_system(A=roof_A, lat=la, long=lo, state=st, initial_cost = 25000, r =.2)
    FB2, C2 = run_simulation(system2,update_fxn) 

    system3 = define_system(A=roof_A, lat=la, long=lo, state=st, initial_cost = 15000, r =.175)
    FB3, C3 = run_simulation(system3,update_fxn) 

    system4 = define_system(A=roof_A, lat=la, long=lo, state=st, initial_cost = 15000, r =.2)
    FB4, C4 = run_simulation(system4,update_fxn) 

    projection = pd.concat([FB,FB2,FB3,FB4,C], axis =1)
    projection.columns = ['I=$25k, r=.175','I=$25k, r=.2','I=$15k, r=.175','I=$15k, r=.2','Regular Grid Service']
    
    intersect = []
    test1 = 1
    test2 = 1
    test3 = 1
    test4 = 1
    for i,r in projection.iterrows():
        if r['I=$25k, r=.175'] > r['Regular Grid Service'] and test1:
            intersect.append(i)
            test1 = 0
        if r['I=$25k, r=.2'] > r['Regular Grid Service'] and test2:
            intersect.append(i)
            test2 = 0
        if r['I=$15k, r=.175'] > r['Regular Grid Service'] and test3:
            intersect.append(i)
            test3 = 0
        if r['I=$15k, r=.2'] > r['Regular Grid Service'] and test4:
            intersect.append(i)
            test4 = 0

    
    
    fig = px.line(projection,
        color_discrete_map={
            'I=$25k, r=.175': 'blue',
            'I=$25k, r=.2':'green',
            'I=$15k, r=.175':'aqua',
            'I=$15k, r=.2':'purple',
            'Regular Grid Service':'red'
        })

    fig.update_layout(
        title={
            'text':'Cost of regular grid power -vs- solar panel array',
            'x':0.5,
            'xanchor':'center',
            'yanchor':'top'
        },
        xaxis_title='# of Months',
        yaxis_title='Projected Cost ($)'
    )
    even_pt_lo = math.ceil(min(intersect)/12)
    even_pt_hi = math.ceil(max(intersect)/12)
    
    fig.add_vrect(
        x0=min(intersect), x1=max(intersect),
        fillcolor='rgb(179,226,205)', opacity=0.5,
        layer="below", line_width=0,
    )

    max_earn = int(projection['I=$15k, r=.2'].max())
    ceiling = max(max_earn,0) - 2500
    fig.update(layout=dict(
        annotations=[
            go.layout.Annotation(x=(min(intersect)+max(intersect))/2, 
                y=ceiling,
                text="Likely Break-Even Range",
                showarrow=False
            )
        ]
    ))
      
    c_over_13 = -(projection.iloc[13*12]['Regular Grid Service'])
    c_over_13 = int(round(c_over_13/100,0)*100)
    yearly_over_13 = int(round(c_over_13/13/10,0)*10)
    

    fun_fcts = '''
    * Over 13 years (median home ownership) with a home like yours, _**you**_ are estimated to pay around **${}** in utility bills or about **${}** per year.
    * If you spend $15,000 to install it, a 20% efficiency system would start being profitable (over the regular utility bills you would have amassed) within approximately **{} years**.
    * For comparison, a $20,000/17.5% efficiency system would take around **{} years** to become profitable.

    
    '''.format(c_over_13, yearly_over_13,even_pt_lo,even_pt_hi)


    return fig, fun_fcts



if __name__ == '__main__':
    app.run_server(debug=True)