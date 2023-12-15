import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path

dirpath = Path(__file__).resolve().parent
file = dirpath/'trains_weather.csv'


def sep_temp_air(df) :
    upper = 200
    lower = -15 # Determine if it is an overflow (and should be removed) or not
    high = (df["RS_E_InAirTemp_PC1"] > upper) | (df['RS_E_InAirTemp_PC2'] > upper)
    higher = df.copy()[high] #Getting only rows that show an upper error
    higher['hta'] = True #Adding a column to indicate if there's an upper error

    low = (df["RS_E_InAirTemp_PC1"] < lower) | (df['RS_E_InAirTemp_PC2'] < lower)
    lower = df.copy()[low]
    lower['lta'] = True

    fitting = df[~(high | low)]
    return fitting, higher, lower

def sep_temp_water(df) :
    upper = 100
    lower = -5 # Determine if it is an overflow (and should be removed) or not
    high = (df["RS_E_WatTemp_PC1"] > upper) | (df['RS_E_WatTemp_PC2'] > upper)
    higher = df.copy()[high]
    higher['htw'] = True
    low = (df["RS_E_WatTemp_PC1"] <= lower) | (df['RS_E_WatTemp_PC2'] <= lower)
    lower = df.copy()[low]
    lower['ltw'] = True
    fitting = df[~(high | low)]
    return fitting, higher, lower

def sep_temp_oil(df) :
    upper = 200
    lower = -100 # Determine if it is an overflow (and should be removed) or not
    high = (df["RS_T_OilTemp_PC1"] > upper) | (df['RS_T_OilTemp_PC2'] > upper)
    higher = df.copy()[high]
    higher['hto'] = True
    low = (df["RS_T_OilTemp_PC1"] < lower) | (df['RS_T_OilTemp_PC2'] < lower)
    lower = df.copy()[low]
    lower['lto'] = True
    fitting = df[~(high | low)]
    return fitting, higher, lower

def sep_rpm(df) :
    upper = 3000
    lower = -1
    high = (df["RS_E_RPM_PC1"] > upper) | (df['RS_E_RPM_PC2'] > upper)
    higher = df.copy()[high]
    higher['hrpm'] = True
    low = (df["RS_E_RPM_PC1"] < lower) | (df['RS_E_RPM_PC2'] < lower)
    lower = df.copy()[low]
    lower['lrpm'] = True
    fitting = df[~(high | low)]
    return fitting, higher, lower

def sep_press(df) :
    upper = 689
    lower = 1 
    high = (df["RS_E_OilPress_PC1"] > upper) | (df['RS_E_OilPress_PC2'] > upper)
    higher = df.copy()[high]
    higher['hpress'] = True
    low = (df["RS_E_OilPress_PC1"] < lower) | (df['RS_E_OilPress_PC2'] < lower)
    lower = df.copy()[low]
    lower['lpress'] = True
    fitting = df[~(high | low)]
    return fitting, higher, lower

'''
def sep_cap_diff(df) :
    column_name=df.columns[4:]
    sensor_columns = column_name
    sensor_columnsPC1 = sensor_columns[::2]
    sensor_columnsPC2 = sensor_columns[1::2]
    reldiff = 0.5
    diff = pd.DataFrame()
    allzero = pd.DataFrame()
    fitting = df.copy()
    for i in range(len(sensor_columnsPC1)):    
        close = (np.isclose(fitting[sensor_columnsPC1[i]], \
                           fitting[sensor_columnsPC2[i]], rtol=reldiff) & (fitting[sensor_columnsPC1[i]] != 0) & (fitting[sensor_columnsPC2[i]] != 0))
        invalid = ((fitting[sensor_columnsPC1[i]] == 0) & (fitting[sensor_columnsPC2[i]] == 0))
        diff = diff._append(fitting[~close])
        allzero = allzero._append(fitting[invalid])
        fitting = fitting[close & ~invalid]
        print("dshape : ",  diff.shape)
    diff['different'] = True
    allzero['allzero'] = True
    return fitting, diff, allzero
'''

df = pd.read_csv(file,sep=";",index_col=[0])

### Cleaning DF

df = df.drop(['lat_y','lon_y','closest','time'],axis=1)
df = df.dropna()
df['mapped_veh_id'] = df['mapped_veh_id'].astype(int)
df['temp'] = df['temp'].astype(float)
df['wind'] = df['wind'].astype(float)
df['humidity'] = df['humidity'].astype(float)
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]


fit,h,l = sep_temp_air(df)
print("Temperature air : ", fit.shape, h.shape, l.shape)
ruled_out = h
ruled_out = ruled_out._append(l)

fit,h,l = sep_temp_oil(fit)
print("Temperature oil : ", fit.shape, h.shape, l.shape)
ruled_out = ruled_out._append(h)
ruled_out = ruled_out._append(l)

fit,h,l = sep_temp_water(fit)
print("Temperature water : ", fit.shape, h.shape, l.shape)
ruled_out = ruled_out._append(h)
ruled_out = ruled_out._append(l)

fit,h,l = sep_press(fit)
print("Pressure : ", fit.shape, h.shape, l.shape)
ruled_out = ruled_out._append(h)
ruled_out = ruled_out._append(l)

fit,h,l = sep_rpm(fit)
print("RPM : ", fit.shape, h.shape, l.shape)
ruled_out = ruled_out._append(h)
ruled_out = ruled_out._append(l)

'''
fit, diff, az = sep_cap_diff(fit)
print("Sensor diff : ",fit.shape, diff.shape, az.shape)
ruled_out= ruled_out._append(diff)
ruled_out= ruled_out._append(az)
ruled_out = ruled_out[~ruled_out.index.duplicated(keep='first')]
print("ruled out df : ",ruled_out.shape)
'''
fit = fit.fillna(False)
ruled_out = ruled_out.fillna(False)
print(ruled_out.head(10))
fit.to_csv(dirpath/'fittingtst.csv',sep=';',index=True)
ruled_out.to_csv(dirpath/'ruled_outtst.csv',sep=';',index=True)

