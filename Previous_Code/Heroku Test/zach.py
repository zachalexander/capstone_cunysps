import pandas as pd 
import numpy as np
import math
# from modsim import *
import importlib

moduleName = input('modsim.py')
importlib.import_module(moduleName)

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
    GHI_raw.to_csv('C:/Users/zalexander/Desktop/ghi_raw.csv')
    #Set the index to the proper timestamps
    GHI_raw = GHI_raw.set_index(pd.date_range('1/1/{yr}'.format(yr=year), freq=interval+'Min', periods=525600/int(interval)))
    temp = GHI_raw[['Month','Day','GHI']]
    daily = temp.groupby(['Month','Day']).sum()
    monthly_mean = daily.groupby(['Month']).mean()
    monthly_sd = daily.groupby(['Month']).std()
    monthly_ghi = pd.DataFrame(monthly_mean)
    monthly_ghi['STD'] = monthly_sd['GHI']
    return monthly_ghi
    # monthly_ghi.to_csv('C:/Users/zalexander/Desktop/monthly_ghi.csv')
    # print(monthly_ghi.to_json())
    # return monthly_ghi.to_json()

monthly_ghi = collect_ghi('test', '29.42412', '-98.49363')

#####   Define a system describing our solar panels and location ############################
def define_system(A=80,r=0.175,PR=0.8,lat=29.42412,long=-98.49363,state='Texas',initial_cost=20000):
    '''Create a system object defining our solar panel system
    '''
    start = State(P=0, N=0, PB=0, MP = -initial_cost, C = 0)
    t0 = 0
    '''15 years worth of operation'''
    t_end = 15*12
    
    return System(start=start, t0=t0, t_end=t_end, A=A, r=r, PR=PR, state = state, lat=lat, long=long)


system = define_system(A=80, r=0.175, PR=0.8, lat=29.42412,long=-98.49363,state='Texas',initial_cost=20000)


def days_yield(system,month):
    month = month
    ghi_day = np.random.normal(monthly_ghi.iloc[month-1]['GHI'],monthly_ghi.iloc[month-1]['STD'])
    ghi_day = float(ghi_day)
    if ghi_day < 0:
        ghi_day = 0
    return (system.A*system.r*ghi_day*system.PR)/1000

days_yielding = days_yield(system, 1)
print(days_yielding)
#############################################################################################



# ####   Function generating a value for the demand on our system in a month. 
# def month_demand_norm(per_sf = 0.06):
#     tot_monthly = per_sf * float(sf)
#     std_d = tot_monthly * 0.15
#     demand_month = np.random.normal(tot_monthly,std_d)
#     if demand_month < 0:
#         demand_month = 0
#     return demand_month

# #############################################################################################

# ####    Function calculating the balance at the end of a month ##############################
# def calc_month(system, month):
#     #2% yearly increase in electricity rates
#     yearly_increase = 1.02
#     year = math.floor(month % 12)

#     month_mod = (month % 12)+1
#     if month_mod in [1,3,5,7,8,10,12]:
#         days = 31
#     elif month_mod in [4,6,9,11]:
#         days = 30
#     elif month_mod == 2:
#         days = 28
#     else:
#         print("Not a valid month number")
#         return None
#     loss = month_demand_norm(monthly_per_sf * yearly_increase**year)
#     p = 0
#     n = 0
#     balance = 0
#     gain = 0

#     price = prices.loc[prices['State'] == system.state, 'Avg_Rate_2019'].iloc[0]
#     price = price/100 * yearly_increase**year

#     for day in range(1,days+1):
#         gain  = gain + days_yield(system,month_mod)
#     balance = gain*price - loss
#     if balance >= 0:
#         p = 1
#     else:
#         n = 1
    
#     this_month = State(P=p, N=n, B=balance, C = loss)
#     return this_month


# #############################################################################################

# def update_fxn(state,system,month):
#     '''Update the pos/neg/balance model.
    
#     state: State with variables P, N, PB, FB, C
#     system: System with relevant info
#     '''
#     p, n, pb, fb, c = state
    
#     month_result = calc_month(system, month)
    
#     p += month_result.P
#     n += month_result.N
#     pb += month_result.B
#     fb += month_result.B
#     c += month_result.C
        
#     return State(P=int(p), N=int(n), PB=pb, FB = fb, C = c)


# ####   The function below generates three TimeSeries objects over the time interval specified 
# ####   within the provided time interval. The TimeSeries track number of months with a positive 
# ####   balance, number of months with a negative balance and the overall balance throughout 
# ####   the interval
# def run_simulation(system,upd_fxn):
#     """Take a system as input and unpdate it based on the update function.
    
#     system - system object defining panels
#     update_fxn - function describing change to system 
    
#     returns - Timeseries
#     """
#     P = TimeSeries()
#     N = TimeSeries()
#     PB = TimeSeries()
#     FB = TimeSeries()
#     C = TimeSeries()
    
#     state = system.start
#     t0 = system.t0
#     P[t0], N[t0], PB[t0], FB[t0], C[t0] = state
    
#     for t in linrange(system.t0, system.t_end):
#         state = upd_fxn(state,system,t)
#         P[t+1], N[t+1], PB[t+1], FB[t+1], C[t+1] = state
    
#     #return P, N, PB, FB, -C
#     return FB, -C