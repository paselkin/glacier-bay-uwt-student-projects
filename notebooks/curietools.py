# -*- coding: utf-8 -*-
"""
Created on Thu Aug 21 08:35:00 2025

@author: paselkin
"""
## Packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import scipy.interpolate as spi
import scipy.optimize as spo
import scipy.stats as sps
import scipy.signal as spsig
import scipy.special as spspecial
import altair as alt
import statsmodels.api as sm
from pathlib import Path
from scipy.special import gamma
import re

_DEBUG = False

## Functions
curie_columns=['temperature','raw_susceptibility','corrected_susceptibility','normalized_susceptibility','bulk_susceptibility','ferrt','ferrb','time','holder_susceptibility']

def read_curie(filename="",
              path=""):
    """
    curie_data = read_curie(filename,path)
        Read Kappabridge Curie temperature file and output PANDAS DataFrame
    """
    curie_data=pd.read_table(path/filename,sep='\s+',header=None,skiprows=1,names=curie_columns)
       # Read in data from fipd.Series.reverse = pd.DataFrame.reverse = lambda self: self[::-1]le
    curie_data['curve_type']='cooling'
    change_point=curie_data['temperature'].idxmax()
    curie_data.loc[0:change_point,'curve_type']='heating'
    return(curie_data)

pd.Series.reverse = pd.DataFrame.reverse = lambda self: self[::-1]
pd.Series.mirror = pd.DataFrame.mirror = lambda self: pd.concat([self, self[::-1]],axis=1)

def process_curie(curie_data,
                  name=None,
                  smooth=10):
    """
    results = process_curie(data,
                            name=None,
                            smooth=10):
        Smooth Kappabridge Curie temperature curve and extract Tc (possibly multiple) using heating and cooling curves, 
        derivative and inverse methods. Smooth is the window length for the moving average used in smoothing; curve is 
        mirrored before smoothing to remove edge effects. 
    """    
    P=alt.Chart(curie_data).mark_circle(size=10).encode(
        alt.X('temperature:Q', axis=alt.Axis(title='T (C)')),
        alt.Y('corrected_susceptibility:Q', axis=alt.Axis(title='K (SI)')),
        alt.Color('curve_type:N', axis=alt.Axis(title='Curve Type')),
        alt.Tooltip(['temperature:Q','curve_type:N'])
    ).interactive()

    susc_heating=curie_data.loc[(curie_data['curve_type']=='heating'),'corrected_susceptibility'].copy()
    susc_cooling=curie_data.loc[(curie_data['curve_type']=='cooling'),'corrected_susceptibility'].copy()
    temp_heating=curie_data.loc[(curie_data['curve_type']=='heating'),'temperature'].copy()
    temp_cooling=curie_data.loc[(curie_data['curve_type']=='cooling'),'temperature'].copy()
    n_heating=susc_heating.size
    n_cooling=susc_cooling.size
    susc_heating_mirrored=pd.concat([susc_heating[::-1],susc_heating, susc_heating[::-1]],axis=0).reset_index(drop=True)
    susc_heating_smooth=susc_heating_mirrored.rolling(smooth,center=True).mean()[n_heating:(2*n_heating)].reset_index(drop=True)
    susc_cooling_mirrored=pd.concat([susc_cooling[::-1],susc_cooling, susc_cooling[::-1]],axis=0).reset_index(drop=True)
    susc_cooling_smooth=susc_cooling_mirrored.rolling(smooth,center=True).mean()[n_cooling:(2*n_cooling)].reset_index(drop=True)
    curie_data['smoothed_susceptibility']=pd.concat([susc_heating_smooth,susc_cooling_smooth],axis=0).reset_index(drop=True)
    S=alt.Chart(curie_data).mark_line().encode(
        alt.X('temperature:Q', axis=alt.Axis(title='T (C)')),
        alt.Y('smoothed_susceptibility:Q', axis=alt.Axis(title='K (SI)')),
        alt.Color('curve_type:N', axis=alt.Axis(title='Curve Type'))
    )
    susc_heating_gradient=pd.Series(np.gradient(susc_heating_smooth.to_numpy(), temp_heating.to_numpy())).reset_index(drop=True)
    susc_cooling_gradient=pd.Series(np.gradient(susc_cooling_smooth.to_numpy(), temp_cooling.to_numpy())).reset_index(drop=True)
    susc_cooling_gradient.index=susc_cooling_gradient.index+n_heating
    curie_data['susceptibility_gradient']=pd.concat([susc_heating_gradient,susc_cooling_gradient],axis=0)
    G=alt.Chart(curie_data).mark_circle(size=10).encode(
        alt.X('temperature:Q', axis=alt.Axis(title='T (C)')),
        alt.Y('susceptibility_gradient:Q', axis=alt.Axis(title='dK/dT (SI/C)')),
        alt.Color('curve_type:N', axis=alt.Axis(title='Curve Type')),
        alt.Tooltip(['temperature:Q','curve_type:N'])
    ).interactive()
    susc_heating_gradient_mirrored=pd.concat([susc_heating_gradient[::-1],susc_heating_gradient,susc_heating_gradient[::-1]],\
                                             axis=0).reset_index(drop=True)
    susc_heating_gradient_smooth=susc_heating_gradient_mirrored.rolling(smooth,center=True).mean()[n_heating:\
                                                                                               (2*n_heating)].reset_index(drop=True)
    susc_cooling_gradient_mirrored=pd.concat([susc_cooling_gradient[::-1],susc_cooling_gradient,susc_cooling_gradient[::-1]],\
                                             axis=0).reset_index(drop=True)
    susc_cooling_gradient_smooth=susc_cooling_gradient_mirrored.rolling(smooth,center=True).mean()[n_cooling:\
                                                                                               (2*n_cooling)].reset_index(drop=True)
    curie_data['smoothed_gradient']=pd.concat([susc_heating_gradient_smooth,susc_cooling_gradient_smooth],axis=0).reset_index(drop=True)
    SG=alt.Chart(curie_data).mark_line().encode(
        alt.X('temperature:Q'),
        alt.Y('smoothed_gradient:Q'),
        alt.Color('curve_type:N', axis=alt.Axis(title='Curve Type'))
    )
    
    susc_heating_gradient2=pd.Series(np.gradient(susc_heating_gradient_smooth.to_numpy(), temp_heating.to_numpy())).reset_index(drop=True)
    susc_cooling_gradient2=pd.Series(np.gradient(susc_cooling_gradient_smooth.to_numpy(), temp_cooling.to_numpy())).reset_index(drop=True)
    susc_cooling_gradient2.index=susc_cooling_gradient2.index+n_heating
    curie_data['susceptibility_gradient2']=pd.concat([susc_heating_gradient2,susc_cooling_gradient2],axis=0)
    G2=alt.Chart(curie_data).mark_circle(size=10).encode(
        alt.X('temperature:Q',axis=alt.Axis(title='T (C)')),
        alt.Y('susceptibility_gradient2:Q', axis=alt.Axis(title='d2K/dT2 (SI2/C2)')),
        alt.Color('curve_type:N', axis=alt.Axis(title='Curve Type')),
        alt.Tooltip(['temperature:Q','curve_type:N'])
    ).interactive()
    susc_heating_gradient2_mirrored=pd.concat([susc_heating_gradient2[::-1],susc_heating_gradient2, susc_heating_gradient2[::-1]],\
                                              axis=0).reset_index(drop=True)
    susc_heating_gradient2_smooth=susc_heating_gradient2_mirrored.rolling(smooth,center=True).mean()[n_heating:\
                                                                                                 (2*n_heating)].reset_index(drop=True)
    susc_cooling_gradient2_mirrored=pd.concat([susc_cooling_gradient2[::-1],susc_cooling_gradient2, susc_cooling_gradient2[::-1]],\
                                              axis=0).reset_index(drop=True)
    susc_cooling_gradient2_smooth=susc_cooling_gradient2_mirrored.rolling(smooth,center=True).mean()[n_cooling:\
                                                                                                 (2*n_cooling)].reset_index(drop=True)
    curie_data['smoothed_gradient2']=pd.concat([susc_heating_gradient2_smooth,susc_cooling_gradient2_smooth],axis=0).reset_index(drop=True)
    SG2=alt.Chart(curie_data).mark_line().encode(
        alt.X('temperature:Q'),
        alt.Y('smoothed_gradient2:Q'),
        alt.Color('curve_type:N')
    )
    susc_heating_inverse=1./susc_heating_smooth.reset_index(drop=True)
    susc_cooling_inverse=1./susc_cooling_smooth.reset_index(drop=True)
    susc_cooling_inverse.index=susc_cooling_inverse.index+n_heating
    curie_data['susceptibility_inverse']=pd.concat([susc_heating_inverse,susc_cooling_inverse],axis=0)
    PI=alt.Chart(curie_data).mark_circle(size=10).encode(
        alt.X('temperature:Q', axis=alt.Axis(title='T (C)')),
        alt.Y('susceptibility_inverse:Q', axis=alt.Axis(title='1/K (1/SI)')),
        alt.Color('curve_type:N'),
        alt.Tooltip(['temperature:Q','curve_type:N'])
    ).interactive()
    susc_heating_inverse_gradient=pd.Series(np.gradient(susc_heating_inverse.to_numpy(), temp_heating.to_numpy())).reset_index(drop=True)
    susc_cooling_inverse_gradient=pd.Series(np.gradient(susc_cooling_inverse.to_numpy(), temp_cooling.to_numpy())).reset_index(drop=True)
    susc_cooling_inverse_gradient.index=susc_cooling_inverse_gradient.index+n_heating
    curie_data['susceptibility_inverse_gradient']=pd.concat([susc_heating_inverse_gradient,susc_cooling_inverse_gradient],axis=0)
    PIG=alt.Chart(curie_data).mark_circle(size=2).encode(
        alt.X('temperature:Q', axis=alt.Axis(title='T (C)')),
        alt.Y('susceptibility_inverse_gradient:Q', axis=alt.Axis(title='d/dT[1/K] (1/C*SI)')),
        alt.Color('curve_type:N')
    )
    
    susc_heating_inverse_gradient_smooth=susc_heating_inverse_gradient.rolling(smooth).mean().reset_index(drop=True)
    susc_cooling_inverse_gradient_smooth=susc_cooling_inverse_gradient.rolling(smooth).mean().reset_index(drop=True)
    susc_cooling_inverse_gradient_smooth.index=susc_cooling_inverse_gradient_smooth.index+n_heating
    curie_data['susceptibility_inverse_gradient_smooth']=pd.concat([susc_heating_inverse_gradient_smooth,\
                                                                    susc_cooling_inverse_gradient_smooth],axis=0)
    PIGS=alt.Chart(curie_data).mark_line().encode(
        alt.X('temperature:Q'),
        alt.Y('susceptibility_inverse_gradient_smooth:Q'),
        alt.Color('curve_type:N')
    )
    
    display(PI+PIG+PIGS)

    s_heating=input('Heating min T, max T:')
    heating_limits = s_heating.split(',')
    heating_limits = [float(x) for x in heating_limits]
    min_heating=min(heating_limits)
    max_heating=max(heating_limits)
    s_cooling=input('Cooling min T, max T:')
    cooling_limits = s_cooling.split(',')
    cooling_limits = [float(x) for x in cooling_limits]
    min_cooling=min(cooling_limits)
    max_cooling=max(cooling_limits)
    
    imax_heating=temp_heating[temp_heating>=max_heating].idxmin()
    imin_heating=temp_heating[temp_heating>=min_heating].idxmin()
    mask_heating=np.zeros(temp_heating.shape).astype(bool)
    mask_heating[imin_heating:imax_heating]=True
    x=temp_heating[mask_heating].to_numpy()
    y=susc_heating_inverse[mask_heating].to_numpy()
    result = sps.linregress(x, y)
    susc_heating_linear=np.zeros(susc_heating.shape)
    susc_heating_linear[mask_heating]=(result.slope*temp_heating.to_numpy()[mask_heating])+result.intercept
    susc_heating_linear=pd.Series(susc_heating_linear)
    susc_heating_linear[~mask_heating]=np.nan
    susc_heating_inverse_Tc=-result.intercept/result.slope
    
    imax_cooling=temp_cooling[temp_cooling>=min_cooling].idxmin()-n_heating
    imin_cooling=temp_cooling[temp_cooling>=max_cooling].idxmin()-n_heating
    mask_cooling=np.zeros(temp_cooling.shape).astype('bool')
    mask_cooling[imin_cooling:imax_cooling]=True
    x=temp_cooling.loc[mask_cooling].to_numpy()
    y=susc_cooling_inverse.loc[mask_cooling].to_numpy()
    result = sps.linregress(x, y)
    susc_cooling_linear=np.zeros(susc_cooling.shape)
    susc_cooling_linear[mask_cooling]=(result.slope*temp_cooling.to_numpy()[mask_cooling])+result.intercept
    susc_cooling_linear=pd.Series(susc_cooling_linear)
    susc_cooling_linear[~mask_cooling]=np.nan
    susc_cooling_linear.index=susc_cooling_linear.index+n_heating
    susc_cooling_inverse_Tc=-result.intercept/result.slope
    curie_data['susceptibility_inverse_linear']=pd.concat([susc_heating_linear,susc_cooling_linear],axis=0)
    
    PIL=alt.Chart(curie_data).mark_line().encode(
        alt.X('temperature:Q'),
        alt.Y('susceptibility_inverse_linear:Q'),
        alt.Color('curve_type:N')
    )
    
    compound_chart=(P+S|G+SG|G2+SG2|PI+PIG+PIL).properties(
        title=name
    )
    display(compound_chart)
    s_derivative=input('Tc based on dK/dT heating, cooling:')
    derivative_Tc = s_derivative.split(',')
    derivative_Tc = [float(x) for x in derivative_Tc]
    susc_heating_derivative_Tc=derivative_Tc[0]
    susc_cooling_derivative_Tc=derivative_Tc[1]
    
    results=pd.Series({'Tc_inverse_heating':susc_heating_inverse_Tc,
                       'Tc_inverse_cooling':susc_cooling_inverse_Tc,
                       'Tc_derivative_heating':susc_heating_derivative_Tc,
                       'Tc_derivative_cooling':susc_cooling_derivative_Tc})
    return(results)