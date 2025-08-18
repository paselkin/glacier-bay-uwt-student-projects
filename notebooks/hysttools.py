# -*- coding: utf-8 -*-
"""
Created on Tue Oct 26 10:41:04 2021

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

def read_hyst(filename="",
              path=""):
    """
    s,d = read_hyst(filename,path)
        Read hysteresis file and output PANDAS DataFrame
    """
    if (path != ''):
        path=path.replace("\\","/")
        if (path[-1]!="/"): path=path+"/"
        filename=path+filename
    with open(filename,"r") as hystfile: 
        # One does not simply read_csv a MicroMag file 
        hystlines = hystfile.readlines()
    hystfile.close()
    start_segments=[i for i,l in enumerate(hystlines) if re.search('Segment',l)][0]+3
    end_segments=[i for i,l in enumerate(hystlines) if bool(re.search('^\s$',l))&(i>start_segments)][0]
    segments=pd.DataFrame([l.rstrip().split(",") for l in hystlines[start_segments:end_segments]])
    segments.columns=['segment','averaging_time','initial_field','field_increment','final_field','pause','final_index']
    start_data=[i for i,l in enumerate(hystlines) if re.search('^\s+Field\s+Moment',l)][0]+2
    end_data=[i for i,l in enumerate(hystlines) if re.search('MicroMag 2900/3900 Data File ends',l)][0]
    data=pd.DataFrame([l.rstrip().split(",") for l in hystlines[start_data:end_data] if not(bool(re.search('^\s$',l)))])
    if (len(data.columns)==2):
        data.columns=['field','moment']
    else:
        data.columns=['field','moment','temperature']
    data['field']=data['field'].apply(float)
    data['moment']=data['moment'].apply(float)
    #data['temperature']=data['temperature'].apply(float)
    segments['final_index']=segments['final_index'].apply(int)
    return (segments,data)

def read_dcd(filename="",
              path=""):
    """
    s,d = read_dcd(filename,path)
        Read dc demag file and output PANDAS DataFrame
    """
    if (path != ''):
        path=path.replace("\\","/")
        if (path[-1]!="/"): path=path+"/"
        filename=path+filename
    with open(filename,"r") as dcdfile: 
        # One does not simply read_csv a MicroMag file 
        dcdlines = dcdfile.readlines()
    dcdfile.close()
    unit_system=[re.split("\s{2,}",line)[1] for line in dcdlines if line.startswith('Units of measure')][0].rstrip()
    if (unit_system == 'Hybrid SI'):
        conversion_field = 1.0
        conversion_mag = 1.0
    elif (unit_system == 'cgs'):
        conversion_field = 1.0E-4
        conversion_mag = 1.0E-3
    if _DEBUG:
        print(unit_system)
    start_segments=[i for i,l in enumerate(dcdlines) if re.search('Segment',l)][0]+3
    end_segments=[i for i,l in enumerate(dcdlines) if bool(re.search('^\s$',l))&(i>start_segments)][0]
    segments=pd.DataFrame([l.rstrip().split(",") for l in dcdlines[start_segments:end_segments]])
    segments.columns=['segment','averaging_time','initial_field','field_increment','final_field','pause','final_index']
    start_data=[i for i,l in enumerate(dcdlines) if re.search('^\s+Field\s+Remanence',l)][0]+2
    end_data=[i for i,l in enumerate(dcdlines) if re.search('MicroMag 2900/3900 Data File ends',l)][0]
    if _DEBUG:
        print("Data block: {}-{}\n Segments:\n{}".format(start_data,end_data,segments))
    data=pd.DataFrame([l.rstrip().split(",") for l in dcdlines[start_data:end_data] if not(bool(re.search('^\s$',l)))])
    if (len(data.columns)==2):
        data.columns=['field','remanence']
    elif (len(data.columns)==3):
        data.columns=['field','remanence','temperature']
    elif (len(data.columns)==5):
        data.columns=['field','remanence','temperature','adjusted_field','adjusted_remanence']
    data['field']=data['field'].apply(float)*conversion_field
    data['remanence']=data['remanence'].apply(float)*conversion_mag
    data['file']=filename
    data['id']=data['file'].apply(lambda x: re.sub('-.*','',Path(x).stem))
    #data['temperature']=data['temperature'].apply(float)
    segments['final_index']=segments['final_index'].apply(int)
    segments['file']=filename
    segments['id']=segments['file'].apply(lambda x: re.sub('-.*','',Path(x).stem))
    segments['type']=segments.apply(lambda x: 'irm' if float(x['final_field'])>0. else 'dcd',axis=1)
    segments['start_index']=segments['final_index'].shift(1)+1
    segments.loc[0,'start_index']=0
    segments['end_index']=segments['final_index']
    
    segments.index = pd.IntervalIndex.from_arrays(segments['start_index'],segments['end_index'],closed='both')
    data=data.reset_index()
    data['type'] = data['index'].apply(lambda x : segments.iloc[segments.index.get_loc(x)]['type'])
    segments=segments.reset_index(drop=True)
    data=data.drop('index',axis=1)
    return (segments,data)

def process_dcd(segments,
                data,
                Hgrid=None,
                lam=0.5,
                smooth=None):
    """
    results = process_dcd(segments,
                                    data,
                                    Hgrid=None,
                                    lam=0.5)
        Smooth dc demag curve and extract useful parameters: Hcr, IRM_80mT
    """    
    dcd_curve=data.loc[(data['type']=='dcd')].copy()
    dcd_curve['field']=-1.*dcd_curve['field']
    dcd_curve=dcd_curve.sort_values(by='field').reset_index(drop=True)
    Hmax=np.max(np.abs(dcd_curve['field'].values))
    if (Hgrid is None):
        N=len(dcd_curve)
        Hgrid=make_hgrid2(N,Hmax)
    if (smooth is None):
        smooth = 10**(2.0*np.ceil(np.log10(0.1*np.min(np.abs(dcd_curve.remanence)))))
    dcd_sp=spi.UnivariateSpline(dcd_curve['field'].values,dcd_curve['remanence'].values,s=smooth,ext='extrapolate')
    lowess = sm.nonparametric.lowess
    predicted=lowess(dcd_curve['remanence'].to_numpy(),dcd_curve['field'].to_numpy(),xvals=Hgrid,frac=1./10.)
    dcd_interpolated=pd.DataFrame({'field':Hgrid,
                                     'remanence':dcd_sp(Hgrid),
                                  'predicted':predicted})
    #dcd_interpolated=dcd_interpolated.sort_values(by='field').reset_index(drop=True)
    #dcd_interpolated.drop_duplicates(subset='remanence',inplace=True)
    #dcd_interpolated.drop_duplicates(subset='field',inplace=True)
    try:
        Hcr=dcd_sp.roots()[0]
    except:
        Hcr=np.nan
    try:
        IRM_80mT=dcd_sp(80.E-3)
    except:
        IRM_80mT=np.nan
    return {'Hcr':Hcr,
            'IRM_m80mT':IRM_80mT,
            'dcd_curve':dcd_curve,
            'dcd_interpolated':dcd_interpolated}
             
def collapse_dcd(data):
    """
    results = collapse_dcd(data)
        Create single IRM and DCD curve from (potentially) multiple segments
    """    
    type_list = segments['type'].unique()
    segments_grouped = segments.groupby('type')
    data_grouped = data.groupby('type')
    new_data=pd.DataFrame([])
    for this_type in type_list:
        data_list = data_grouped.get_group(this_type)
        if (this_type=='irm'):
            data_list = data_list.sort_values('field',inplace=True,ascending=True)
        else:
            data_list = data_list.sort_values('field',inplace=True,ascending=False)
        new_data=pd.concat([new_data,data_list],axis=0)
    return (new_data)

def eliminate_outliers(irm_sp,irm_curve,smooth,Hgrid,nb):
    mask=np.array([False]*len(irm_curve.index))
    for i in np.linspace(0,nb,num=nb):
        irm_smaller=irm_curve[~mask]
        fields=irm_smaller['field'].values
        actual=irm_smaller['remanence'].values
        predicted=irm_sp(fields)
        sq_difference=np.array([x**2 for x in (actual-predicted)])
        confidence_interval=1.96*np.mean(sq_difference)
        fields_remove=fields[sq_difference>confidence_interval]
        condition_remove=np.array(irm_curve['field'].isin(fields_remove))
        if any(condition_remove):
            mask=mask|condition_remove 
            if _DEBUG:
                print(f'Iteration: {i}\nRemoving: {[j for j, x in enumerate(mask) if x]}')
            irm_smaller=irm_curve[~mask]
            irm_sp=spi.UnivariateSpline(irm_smaller['field'].values,
                                    irm_smaller['remanence'].values,s=smooth,ext='extrapolate')
        if _DEBUG:
            print(f'Outliers: {[j for j, x in enumerate(mask) if x]}')
        
    if _DEBUG:
        print(f'Final: removing {len([j for j, x in enumerate(mask) if x])} steps.')
    return(irm_sp)

#def adjust_knots(irm_sp,irm_curve,smooth,t,Hgrid,ni=5):
#    print(f'Knots: {t}')
#    for i in np.linspace(0,ni,num=ni):
#        fields=irm_curve['field'].values
#        actual=irm_curve['remanence'].values
#        predicted=irm_sp(fields)
#        dirm_sp=irm_sp.derivative()
#        derivative=dirm_sp(Hgrid)
#        negative_derivative=Hgrid[j for j,x in enumerate(derivative<0.) if x]
#        for k in negative_derivative:
            #not finished
#    return(irm_sp)
        
        

def process_irm(segments,
                data,
                Hgrid=None,
                lam=0.5,
                k=3,
                smooth=None,
                nb=0,
                adjust=False):
    """
    results = process_irm(segments,
                                    data,
                                    Hgrid=None,
                                    lam=0.5,
                                    k=3,
                                    smooth=None,
                                    nb=0)
        Smooth irm curve and extract useful parameters: Hcr, IRM_80mT
    """    
    irm_curve=data.loc[(data['type']=='irm')].copy()
    irm_curve=irm_curve[irm_curve['field']>=0]
    fields=irm_curve['field'].values
    if _DEBUG:
        print(fields)
        print(np.diff(fields) >= 0)
    if (np.any(np.diff(fields) <= 0)):
        is_sorted=np.concatenate(([True],(np.diff(fields) <= 0)),axis=0)
        irm_curve.drop(irm_curve[is_sorted].index,inplace=True)
    #irm_curve=irm_curve.sort_values(by='field').reset_index(drop=True)
    Hmax=np.max(np.abs(irm_curve['field'].values))
    if (Hgrid is None):
        N=len(irm_curve)
        Hgrid=make_hgrid2(N,Hmax)
    if (smooth is None):
        smooth = 10**(2.0*np.ceil(np.log10(0.1*np.min(np.abs(irm_curve.remanence)))))
    max_field=irm_curve['field'].max()
    irm_periodic=pd.DataFrame({'field':np.concatenate((np.flip(-irm_curve['field'].values),irm_curve['field'].values,
                                                       np.flip((2*max_field)-irm_curve['field'].values))),
                               'remanence':np.concatenate((np.flip(irm_curve['remanence'].values),irm_curve['remanence'].values,
                                                           np.flip(irm_curve['remanence'].values)))})
    irm_sp=spi.UnivariateSpline(irm_curve['field'].values,irm_curve['remanence'].values,s=smooth,ext='extrapolate')
    knots=irm_sp.get_knots()
    #print(knots)
    #irm_sp=spi.LSQUnivariateSpline(irm_curve['field'].values,irm_curve['remanence'].values,t=knots,ext='extrapolate')
#    if adjust:
#        irm_sp=adjust_knots(irm_sp,irm_curve,smooth,t,Hgrid,nb)
    if (nb>0):
        # Iterate to eliminate outliers
        irm_sp=eliminate_outliers(irm_sp,irm_curve,smooth,Hgrid,nb)
    irm_interpolated=pd.DataFrame({'field':Hgrid,
                                     'remanence':irm_sp(Hgrid)})
    irm_normalized=irm_interpolated.copy()
    #irm_normalized['remanence']=irm_normalized['predicted']/irm_normalized['predicted'].max()
    #irm_derivative=irm_normalized.copy()
    #irm_derivative['gradient']=pd.Series(np.gradient(irm_derivative['predicted'],irm_derivative['field']))
    irm_normalized['remanence']=irm_normalized['remanence']/irm_normalized['remanence'].max()
    irm_derivative=irm_normalized.copy()
    irm_derivative['gradient']=pd.Series(np.gradient(irm_derivative['remanence'],irm_derivative['field']))
    #dirm_sp=irm_sp.derivative()
    #irm_derivative['gradient']=dirm_sp(Hgrid)
    irm_derivative=irm_derivative[irm_derivative['gradient']>=0]
    try:
    #    SIRM=irm_interpolated['predicted'].max()
        SIRM=irm_interpolated['remanence'].max()
    except:
        SIRM=np.nan
    try:
        IRM_50mT=irm_sp(50.E-3)
    except:
        IRM_50mT=np.nan
    try:
        IRM_80mT=irm_sp(80.E-3)
    except:
        IRM_80mT=np.nan
    try:
        IRM_100mT=irm_sp(100.E-3)
    except:
        IRM_100mT=np.nan
    try:
        IRM_300mT=irm_sp(300.E-3)
    except:
        IRM_300mT=np.nan
    #irm_interpolated=irm_interpolated.sort_values(by='field').reset_index(drop=True)
    #irm_interpolated.drop_duplicates(subset='remanence',inplace=True)
    #irm_interpolated.drop_duplicates(subset='field',inplace=True)
    return {'irm_curve':irm_curve,
            'irm_interpolated':irm_interpolated,
            'irm_normalized':irm_normalized,
            'irm_derivative':irm_derivative,
            'SIRM':SIRM,
            'IRM_50mT':IRM_50mT,
            'IRM_80mT':IRM_80mT,
            'IRM_100mT':IRM_100mT,
            'IRM_300mT':IRM_300mT}

def test_plot(result_irm,name):
    #id=result_irm['irm_curve']['id'][0]
    field1=result_irm['irm_curve']['field'].to_numpy()
    remanence1=result_irm['irm_curve']['remanence'].to_numpy()
    df1=pd.DataFrame({'field':field1,'remanence':remanence1/np.max(remanence1)})
    #print(df1)
    p1=alt.Chart(df1).mark_point(color='black').encode(
        x=alt.X('field:Q'),
        y=alt.Y('remanence:Q')
    ).properties(
        title=f'{name}'
    )
    field2=result_irm['irm_interpolated']['field'].to_numpy()
    remanence2=result_irm['irm_interpolated']['remanence'].to_numpy()
    df2=pd.DataFrame({'field':field2,'remanence':remanence2/np.max(remanence2)})
    #print(df2)
    p2=alt.Chart(df2).mark_point(color='gray').encode(
        x=alt.X('field:Q'),
        y=alt.Y('remanence:Q')
    )
    field3=result_irm['irm_normalized']['field'].to_numpy()
    remanence3=result_irm['irm_normalized']['remanence'].to_numpy()
    df3=pd.DataFrame({'field':field3,'remanence':remanence3})
    #print(df2)
    p3=alt.Chart(df3).mark_circle(color='steelblue').encode(
        x=alt.X('field:Q'),
        y=alt.Y('remanence:Q')
    )
    field4=result_irm['irm_derivative']['field'].to_numpy()
    remanence4=result_irm['irm_derivative']['gradient'].to_numpy()
    df4=pd.DataFrame({'field':field4,'remanence':remanence4})
    #print(df2)
    p4=alt.Chart(df4).mark_circle(color='salmon').encode(
        x=alt.X('field:Q',scale=alt.Scale(type='log')),
        y=alt.Y('remanence:Q')
    )
    display(p1+p2+p3|p4)
def irm_fit(data, guess=None):
    if (guess is None):
        v=list(data['gradient'].values())
        f=list(data['field'].values())
        peaks,_=spsig.find_peaks(v)
        guess=[1., f[peaks[0]], 1.0, 0.1, 1.0]
    popt, pcov=spo.curve_fit(sgg,guess,bounds=([0, 0, -1., 0],[inf, inf, 1., inf]))
    return popt

def corr_loop(Hoff,upper,lower,return_params=False):
    # Function to minimize in process_hyst to find horizontal and vertical offset
    lower_t=-lower.copy(deep=True)
    lower_t['field']=lower_t['field']-(2*Hoff)
    lower_t=lower_t.sort_values(by='field')
    hu=upper['field']
    sp=spi.splrep(lower_t['field'],lower_t['moment'])
    ml=spi.splev(hu,sp)
    lrs=sps.stats.linregress(upper['moment'].values,ml)
    if return_params:
        return lrs
    else:
        return -(lrs[2]**2)    

def make_hgrid(N,Hmax,lam):
    i=[x for x in np.arange(-N+1,N) if x != 0]
    Hgrid=(np.abs(i)/i)*(Hmax/lam)*(((lam+1)**(np.abs(i)/N))-1)     # From Von Dobeneck 1996
    Hgrid=Hgrid[~np.isnan(Hgrid)]
    return Hgrid

def make_hgrid2(N,Hmax):
    i=np.arange(0,N)
    Hgrid=np.concatenate([np.array([0.]),np.geomspace(Hmax/N, Hmax, num=N)])
    return Hgrid

def process_hyst(segments,
                 data,
                 Hgrid=None,
                 lam=0.5,
                 offset_correct=True,
                 fraction_saturation=0.7,
                 smooth=None,
                 initial_susceptibiity_field=15.E-3):
    """
    results = process_hyst(segments,
                                    data,
                                    lam=0.5,
                                    offsset_correct=True,
                                    fraction_saturation=0.7,
                                    initial_susceptibiity_field=15.E-3)
        Smooth hysteresis loop and extract useful parameters: Mrs, Ms, Ki, Khf, etc.
    """
    # Step 1 of approach: split loop into initial, upper, and lower branches
    initial_start=0
    initial_end=segments['final_index'][0]
    upper_start=segments['final_index'][0]
    upper_end=segments['final_index'][1]
    lower_start=segments['final_index'][1]
    lower_end=segments['final_index'][2]
    initial_loop=data.iloc[initial_start:initial_end,0:2]
    upper_loop=data.iloc[upper_start:upper_end,0:2]
    lower_loop=data.iloc[lower_start:lower_end,0:2]
    # Calculate initial susceptibility from initial loop at fields below initial_susceptibiity_field
    init_linear = initial_loop.loc[(initial_loop['field']<=initial_susceptibiity_field),:]
    print('init_linear',init_linear)
    init_params=sps.stats.linregress(init_linear['field'].values,init_linear['moment'].values)
    Xinit=init_params[0]
    #except:
    #    print('Xinit failed. Xinit = np.nan')
    #    Xinit=np.nan
    # Iteratively solve the following problem:
    #   For each Hoff from -Hmax to Hmax
    #   Invert lower loop through Hoff,0
    #   Spline interpolate at fields used in upper loop
    #   Calculate R^2 between lower and upper loops
    #   Maximize R^2 (note: using scipy.optimize.brent, so minimize -R^2)
    Hmax=np.max(np.abs(upper_loop['field'].values))
    try:
        Hoff=spo.brent(corr_loop,args=(upper_loop,lower_loop),brack=(-Hmax,0,Hmax))
    except:
        print('Minimization failed. Hoff=0')
        Hoff=0
    
    # Calculate error estimates R^2 and Q and vertical offset M0
    lrs=corr_loop(Hoff,upper_loop,lower_loop,True)
    r2=(lrs[2]**2)
    M0=lrs[1]/2.
    Q=np.log10(1/(1-r2))
    
    if ((offset_correct==True)|(offset_correct=='H')):
        # Correct loop for horizontal offset
        upper_loop['field']=upper_loop['field']-Hoff
        lower_loop['field']=lower_loop['field']-Hoff
    if ((offset_correct==True)|(offset_correct=='M')):
        # Correct loop for vertical offset
        upper_loop['moment']=upper_loop['moment']-M0
        lower_loop['moment']=lower_loop['moment']-M0
    
    # "Grid" and interpolate both half loops
    if (Hgrid is None):
        N=len(upper_loop)
        Hgrid=make_hgrid(N,Hmax,lam)
    
    upper_loop=upper_loop.sort_values(by='field').reset_index(drop=True)
    upper_loop.drop_duplicates(subset='moment',inplace=True)
    upper_loop.drop_duplicates(subset='field',inplace=True)
    try:
        upper_sp=spi.splrep(upper_loop['field'].values,upper_loop['moment'].values)
        upper_interpolated=pd.DataFrame({'field':Hgrid,
                                     'moment':spi.splev(Hgrid,upper_sp)})
    except TypeError:
        print('Degree too high. rying smoothing spline.\n')
        upper_sp=spi.UnivariateSpline(upper_loop['field'].values,upper_loop['moment'].values)
        upper_interpolated=pd.DataFrame({'field':Hgrid,
                                     'moment':upper_sp(Hgrid)})
    lower_loop=lower_loop.sort_values(by='field').reset_index(drop=True)
    lower_loop.drop_duplicates(subset='moment',inplace=True)
    lower_loop.drop_duplicates(subset='field',inplace=True)
    try:
        lower_sp=spi.splrep(lower_loop['field'].values,lower_loop['moment'].values)
        lower_interpolated=pd.DataFrame({'field':Hgrid,
                                     'moment':spi.splev(Hgrid,lower_sp)})
    except TypeError:
        print('Degree too high. trying smoothing spline.\n')
        lower_sp=spi.UnivariateSpline(lower_loop['field'].values,lower_loop['moment'].values)
        lower_interpolated=pd.DataFrame({'field':Hgrid,
                                     'moment':lower_sp(Hgrid)})
    lower_inverted=(-lower_interpolated).sort_values(by='field').reset_index(drop=True)
    
    # Calculate error estimate err(H)
    err=upper_interpolated['moment']-lower_inverted['moment']
    
    #calculate Mrh and Mih
    reversible=pd.DataFrame({'field':upper_interpolated['field'],
                'moment':(upper_interpolated['moment']-lower_interpolated['moment'])/2.})
    irreversible=pd.DataFrame({'field':upper_interpolated['field'],
                'moment':(upper_interpolated['moment']+lower_interpolated['moment'])/2.})
    
    # Calculate Mrsp (Mrs from reversible moment function), Ms, Xhf
    Mrsp = reversible['moment'].max()
    top_linear=irreversible.loc[irreversible['field']>=(fraction_saturation*Hmax),:]
    bottom_linear=irreversible.loc[irreversible['field']<=(-fraction_saturation*Hmax),:]
    bottom_inverted=(-bottom_linear).sort_values(by='field').reset_index(drop=True)
    all_linear=pd.concat([top_linear,bottom_inverted],axis=0)
    hfs=sps.stats.linregress(all_linear['field'].values,all_linear['moment'].values)
    Ms=hfs[1]
    Xhf=hfs[0]

    # Detrend loop and calculate new Qeff
    upper_detrended=upper_interpolated.copy(deep=True)
    upper_detrended['moment']= upper_detrended['moment']-(upper_detrended['field']*Xhf)
    lower_detrended=lower_interpolated.copy(deep=True)
    lower_detrended['moment']= lower_detrended['moment']-(lower_detrended['field']*Xhf)
    irreversible_detrended=irreversible.copy(deep=True)
    irreversible_detrended['moment']= irreversible_detrended['moment']-(irreversible_detrended['field']*Xhf)
    lrs=corr_loop(0,upper_detrended,lower_detrended,True)
    r2=(lrs[2]**2)
    Qeff=np.log10(1/(1-r2))
    
    if (smooth is None):
        smooth = 0.5*np.sqrt(np.mean(err**2.))  # Smooth within half of MSE
    
    # Calculate Hc, Hcp (Hc from reversible and irreversible loops),Mrs
    #lower_spline=spi.interp1d(lower_detrended['field'],lower_detrended['moment'],fill_value="extrapolate")
    lower_spline=spi.UnivariateSpline(lower_detrended['field'],lower_detrended['moment'],s=smooth)
    print(lower_spline.roots())
    try:
        Hc=lower_spline.roots()[0]
    except:
        Hc=0    
    if (Hc<=0.):
        print ("Warning: Hc<0. Trying upper branch.\n")
        upper_spline=spi.interp1d(upper_detrended['field'],upper_detrended['moment'],fill_value="extrapolate")
        Hc=-spo.fsolve(upper_spline,0.)[0]
        print ("Upper Hc: ",Hc)
    try:
        distance=np.abs(reversible['moment']-irreversible['moment'])
        distance_spline=spi.interp1d(irreversible['field'],distance)
        Hcp=spo.minimize(distance_spline,0,method='CG')['x']
    except ValueError:
        print('An error occurred in finding Hcp.')
        Hcp=None
    upper_search=upper_detrended.loc[abs(upper_detrended['field'])<Hc,:]
    upper_rspline=spi.interp1d(upper_detrended['moment'],upper_detrended['field'],fill_value="extrapolate")
    Mrsu=spo.fsolve(upper_rspline,0,)[0]
    lower_search=lower_detrended.loc[abs(lower_detrended['field'])<Hc,:]
    lower_rspline=spi.interp1d(lower_detrended['moment'],lower_detrended['field'],fill_value="extrapolate")
    Mrsl=spo.fsolve(lower_rspline,0)[0]
    Mrs=0.5*(Mrsu-Mrsl)
    
    return {'upper':upper_loop,
            'upper_interpolated':upper_interpolated,
            'upper_detrended':upper_detrended,
            'lower':lower_loop,
            'lower_interpolated':lower_interpolated,
            'lower_inverted': lower_inverted,
            'lower_detrended':lower_detrended,
            'reversible': reversible,
            'irreversible': irreversible,
            'irreversible_detrended':irreversible_detrended,
            'horizontal_offset':Hoff,
            'vertical_offset':M0,
            'Mrs':Mrs,
            'Ms':Ms,
            'Xhf':Xhf,
            'Xinit':Xinit,
            'Mrsp':Mrsp,
            'Hc':Hc,
            'Hcp':Hcp,
            'Q':Q,
            'Qeff':Qeff,
            'err':err}

def plot_hyst(result,ax,plot_detrended=True,title=None,xlim=None,ylim=None):
    ax.plot(result['upper_interpolated']['field'],result['upper_interpolated']['moment'],'-',c='#666666')
    ax.plot(result['lower_interpolated']['field'],result['lower_interpolated']['moment'],'-',c='#666666')

    ax.plot(result['upper']['field'],result['upper']['moment'],'.',c='#000000')
    ax.plot(result['lower']['field'],result['lower']['moment'],'.',c='#000000')

    ax.plot(result['reversible']['field'],result['reversible']['moment'],'-',c='g',linewidth=1)
    ax.plot(result['irreversible']['field'],result['irreversible']['moment'],'-',c='r',linewidth=1)

    if (plot_detrended):
        ax.plot(result['upper_detrended']['field'],result['upper_detrended']['moment'],'-',c='k')
        ax.plot(result['lower_detrended']['field'],result['lower_detrended']['moment'],'-',c='k')

    # Move left y-axis and bottim x-axis to centre, passing through (0,0)
    ax.spines['left'].set_position('center')
    ax.spines['bottom'].set_position('center')

    # Eliminate upper and right axes
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')

    # Show ticks in the left and lower axes only
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    
    if (xlim is not None):
        ax.set_xlim(xlim)
    if (ylim is not None):
        ax.set_ylim(ylim)
    
    if (title is not None):
        bbox_props=dict(boxstyle='square',fc='w',ec='w')
        xb=ax.get_xlim()[0]
        yb=ax.get_ylim()[1]
        ax.text(xb,yb,title,ha='left',va='top',fontsize=12,bbox=bbox_props,fontweight='bold')
    return ax

def plot_err(result,ax,xlim=None,ylim=None):
    ax.plot(result['upper_interpolated']['field'],result['err'],'r-')
    if (xlim is not None):
        ax.set_xlim(xlim)
    if (ylim is not None):
        ax.set_ylim(ylim)
    return ax

def plot_hyst_report(result,fig,id):
    ax_hyst_main=plt.subplot2grid((3,2),(0,0),rowspan=2)
    ax_hyst_inset=plt.subplot2grid((3,2),(0,1),rowspan=2)
    ax_err=plt.subplot2grid((3,2),(2,0),rowspan=1)
    #ax_table=plt.subplot2grid((3,2),(2,1),rowspan=1)
    
    ax_hyst_main=plot_hyst(result,ax_hyst_main,title=id)
    ax_hyst_inset=plot_hyst(result,ax_hyst_inset,xlim=(-10.*result['Hc'],10.*result['Hc']),ylim=(-2.*result['Ms'],2.*result['Ms']))
    ax_err=plot_err(result,ax_err)
    
    #result_table=list(map(lambda x: "{}".format(x),[result['Mrs'],result['Ms'],result['Hc'],result['Xhf'],result['Q']]))
    #result_labels=('Mrs','Ms','Hc','Xhf','Q')
    #ax_table.xaxis.set_visible(False)
    #ax_table.yaxis.set_visible(False)
    #ax_table.table(cellText=result_table,loc='center')
    
    plt.tight_layout()
    
    return (ax_hyst_main,
            ax_hyst_inset,
            ax_err)

def sgg(x, *params):
    # Skewed Gaussian using skewnorm (similar to Egli, 2004a eq. 1, but without kurtosis parameter)
    # Note that shape parameter alpha here is not traditional skewness. See https://en.wikipedia.org/wiki/Skew_normal_distribution
    fx=np.zeros_like(x)
    N_dist=int(len(params)/4)
    param_array = np.reshape(list(params),(N_dist,4))
    for row in param_array:
        A,mu,sigma,p=tuple(row)
        fx=fx+sps.skewnorm.cdf(x,a=p,loc=mu,scale=sigma)
    return fx

def fit_sgg(data,N_dist,guess=None,log_transform=True,max_nfev=None,ftol=1E-6,debug=False):
    # Fit multiple skewed Gaussian distributions to IRM data
    # If guess is none, makes a first guess
    # Returns optimum parameters (popt) and covariance matrix (pcov), prediction (pred) 
    # and statistics for each distribution (sgg_params).
    if (type(data['remanence'])==dict):
        remanence=list(data['remanence'].values())
    else:
        remanence=data['remanence'].tolist()
    if (type(data['field'])==dict):
        field=list(data['field'].values())
    else:
        field=data['field'].tolist()
    if (log_transform):
        if (any([x==0 for x in field])):
            field=[x for x in field if (x!=0)]
            remanence=[x[1] for x in zip(field,remanence) if (x[0]!=0)]
        field=np.log(field)
    if max_nfev is None:
        max_nfev=100*len(field)
    if guess is None:
        A=1/N_dist
        n,minmax,d_50,v,s,k=sps.describe(remanence)
        d_iqr=sps.iqr(remanence)
        d_25=d_50-(0.5*d_iqr)
        if N_dist == 1.:
            guess=[A, d_50, d_iqr, 0.1]
        else:
            guess=[A, d_50, d_iqr, 0.1]*N_dist
            mu_base=(d_iqr)/N_dist
            for i in range(N_dist):
                guess[(4*i)+1]=(mu_base*i)+d_25
    lower_bound=[0,0, 0, -1.]*N_dist
    upper_bound=[np.inf,np.inf, np.inf, np.inf]*N_dist
    if (debug):
        print(guess)
        print(lower_bound)
        print(upper_bound)
    popt, pcov=spo.curve_fit(sgg,field,remanence,p0=guess,method='trf',
                             bounds=(lower_bound,upper_bound),max_nfev=max_nfev,ftol=ftol)
    pred=sgg(field,*popt)
    sgg_params=pd.DataFrame(columns=['component','mean','variance','skewness','kurtosis','rmse'])
    for i in range(N_dist):
        m,v,s,k=sps.skewnorm.stats(a=popt[(4*i)+3],loc=popt[(4*i)+1],scale=[(4*i)+2],moments='mvsk')
        rmse=np.sqrt(np.sum([x**2 for x in (pred-remanence)])/len(pred))
        sgg_params=sgg_params.append({'component':i, 'mean':m, 'variance':v, 'skewness':s, 'kurtosis':k, 'rmse':rmse},ignore_index=True)
    return(popt,pcov,pred,sgg_params)
    
## Data
dunlop_SDMD1=pd.DataFrame({'soft_component':[0,10,20,30,40,50,60,70,80,90,100],'HcrHc':[1.4193,1.492,1.5859,1.6859,1.8121,1.9695,2.1766,2.4731,2.9053,3.6082,4.9541],'MrMs':[0.3767,0.337,0.3066,0.2683,0.2321,0.1975,0.1608,0.1245,0.0886,0.0531,0.0181]})

dunlop_SDMD2=pd.DataFrame({'soft_component':[0,10,20,30,40,50,60,70,80,90,100],'HcrHc':[1.2422,1.2915,1.3205,1.3806,1.4595,1.5514,1.677,1.895,2.2635,2.9878,5.3536],'MrMs':[0.4948,0.4426,0.4005,0.3524,0.3016,0.2581,0.2101,0.1608,0.1139,0.0668,0.0186]})

dunlop_SP10SD=pd.DataFrame({'soft_component':[5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85],'HcrHc':[1.256,1.4111,1.6209,1.9355,2.324,2.6993,3.1006,3.5617,4.1369,4.8051,5.5811,6.4827,7.7415,9.245,11.2878,14.17,18.3904],'MrMs':[0.4975,0.4731,0.4425,0.4231,0.4001,0.37,0.3498,0.3235,0.296,0.2723,0.2476,0.2215,0.197,0.1705,0.1491,0.1241,0.0982]})

dunlop_SP15SD=pd.DataFrame({'soft_component':[5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80],'HcrHc':[1.256,1.6118,1.9679,2.4701,3.7433,7.3197,9.0864,11.0323,13.2473,15.9072,19.2076,23.4512,28.7924,35.5468,45.3712,59.5404],'MrMs':[0.5003,0.4474,0.4231,0.3978,0.3719,0.3494,0.3249,0.3005,0.2748,0.2499,0.2236,0.2,0.1739,0.1505,0.1252,0.1002]})