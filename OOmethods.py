#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 15:04:46 2017

@author: lb16998
"""
import xray
import numpy as np
import pandas as pd
import pickle
import os
from scipy.odr import Model, Data, ODR, RealData
from scipy.stats import linregress
from datetime import datetime
import matplotlib.pyplot as plt
from math import radians, cos, sin, asin, sqrt, log
from scipy.spatial.distance import pdist, squareform
import itertools as it
import calendar

def filterByLandPct(df,filPct=5):
    """ Filters data for to only include periods when the percentage footprint 
    over land is less than the number specified in filPct."""
    landpct = df['landpc']*100
    mask = np.ma.masked_where(landpct<filPct,landpct)
    mask = mask.mask
    df = df[mask==True]
    
    return df

def filterBySeaInfluence(df,filPct=95):
    """ Filters data for to only include periods when the estimated percentage 
    atmopsheric concentration derived from marine sources, as estimated by the 
    NAME model and source datasets is greater than the percentage specified in 
    filPct""" 
    frac = df['Ocean']/df['Combined']*100
    mask = np.ma.masked_where(frac<filPct,frac)
    mask = mask.mask
    df = df[mask==False]
    
    return df

def listFilesInDirectory(path_name,search_string):
    """ Lists all files in directory 'path_name' containing the string 
    'search_string'"""
    content = []
    for dirname, dirnames, filenames in os.walk(path_name):
        for filename in filenames:
            if search_string in filename:
                content.append(os.path.join(dirname,filename))
    return(content)

def get_stdev(df):
    """ Calculates standard deviation of 5 data cycles centred on measurement.
    First two and last two cycles ignored and removed from dataframe."""
    stdev = np.zeros(len(df))
    for i in range(0,len(df)):        
        try:
            stdev[i] = np.std(df['Ocean'][i-2:i+2])
        except:
            stdev[i] = 0
            
    df['Ocean stdev'] = stdev
    df = df[df['Ocean stdev']>0]

    return(df)

class dataAll(object):
    """ Data class. Stores the filtered data before the regression"""
    def __init__(self,site,gas_ex):
        # Reads in data
        data = None
        path = '/home/lb16998/work/N2O/datamerge/'
        content  = listFilesInDirectory(path,site+'_merged_landmask_added_filt_'+gas_ex) 
        for fil in content:  
            dataT = pickle.load(open(fil, "rb" ) )
            if data is None:
                data = dataT
            else:
                data = pd.concat([data,dataT])
        # Subtract NAME boundary conditions from measurements
        data['Agage_corr'] = data['Agage'] - data['Boundary']
        # Store variables             
        self.site = site
        self.gas_ex = gas_ex
        self.filter_method = 'unfiltered'
        self.data = data.sort_index()
            
    def filter_data(self,filter_method,filter_val):
        """ Calculates standard deviation of measurements and filters by choosen 
        method""" 
        data_stdev = get_stdev(self.data)
        self.filter_method =  filter_method
        if filter_method == 'landpct':
            self.data = filterByLandPct(data_stdev,filPct=filter_val)            
        elif filter_method == 'sea_influence':
            self.data = filterBySeaInfluence(data_stdev,filPct=filter_val)
        
class OZ(dataAll):
    """ subclass of data class for Cape Grim data, which requires extra 
    filtering to remove periods when NAME model is playing up."""
    def __init__(self,site,gas_ex):
        dataAll.__init__(self,site,gas_ex)
    
    def remove_bad_data(self):        
        # Removes periods with 0 in Boundary files and strange period of data
        # in November 2013.
        df = self.data[self.data.Boundary != 0]
        msk = np.zeros(len(df))
        time = df.index
        tIdx = np.where(np.logical_and(time>='2013-11-18', time<='2013-11-29'))
        msk[tIdx] = 1
        self.data = df[msk==0]

def f(p, x):
    """Basic linear regression 'model' for use with ODR"""
    return (p[0] * x) + p[1]

def odreg(x,y,sx,sy):
    """ Runs ODR functions"""
    # Creates linear regression model
    linreg = linregress(x, y)
    linear = Model(f)
    # Runs ODR
    mydata = RealData(x, y, sx=sx, sy=sy)
    myodr = ODR(mydata, linear, beta0=linreg[0:2])
    myoutput = myodr.run()
    
    # Create dataframe of the input data. Used for plotting later.
    data = pd.DataFrame({'measurements':y,'predictions':x})

    return myoutput,data

def filt(df,year=2006,month=1):
    """ Creates a mask which can be used to filter the data frame to retreive
    data related to a specific month"""
    msk = np.zeros(len(df))   
    idx = np.where(np.logical_and(df.index.year==year, df.index.month==month))
    msk[idx]=1
    msk = msk.astype(bool)

    return(msk)

def haversine(lat1, lon1, lat2, lon2):
    """ Uses the haversine formula to calculate the distance between two 
    positions (in m)""" 
    R = 6372800 # Earth radius in metres
 
    dLat = radians(lat2 - lat1)
    dLon = radians(lon2 - lon1)
    lat1 = radians(lat1)
    lat2 = radians(lat2)
 
    a = sin(dLat/2)**2 + cos(lat1)*cos(lat2)*sin(dLon/2)**2
    c = 2*asin(sqrt(a))
 
    return R * c

def calc_flux(years,fils,domain,labo=[-180,180],lobo=[-360,360]):
    """The total flux within the specified area for each month is calculated."""
    
    # Checks that all years of data are in the ODR dataframe. If dataframe
    # only contains subset of all years then missing years are excluded.
    fils_filt = []
    for year in years:
        for fil in fils:
            if str(year) in fil and fil not in fils_filt:
                fils_filt.append(fil)                   
    fils_filt.sort()

    # Use haversine function to calculate area of each grid cell
    dsT = xray.open_dataset(fils_filt[0])
    area = np.zeros(dsT.flux.values[:,:,0].shape)
    for la,latT in enumerate(dsT.lat.values):
        for lo,lonT in enumerate(dsT.lon.values):
            latDist = haversine(latT,lonT,latT+.234,lonT)
            lonDist = haversine(latT,lonT,latT,lonT+.352)
            area[la,lo] = latDist*lonDist
    # Create meshgrid of lat/lons and convert to vector 
    xg,yg=np.meshgrid(dsT.lon.values,dsT.lat.values)
    xv = np.ravel(xg)
    yv = np.ravel(yg)
    # Create mask using latitude/longitude boundaries provides in function input
    mask = np.zeros(len(xv))
    xmask = np.where(np.logical_and(xv>=lobo[0], xv<=lobo[1]))
    ymask = np.where(np.logical_and(yv>=labo[0], yv<=labo[1]))
    mask[np.intersect1d(xmask,ymask)] = 1
    
    # Adds gridcells over land to mask  
    surfmaskds = xray.open_dataset('/home/lb16998/work/N2O/AIRSEA/N2OFLXS_'+domain+'_Surace_Mask_2006_2013.nc')
    surfmask = np.ravel(surfmaskds.surfacemask.values[:,:,0])
    zmask = np.ma.masked_where(surfmask==1,surfmask)
    mask = mask*zmask.mask
    mask = mask.astype(bool)
    
    # Calculate total area of gridcells within box which are over sea.
    totarea = np.sum(np.ravel(area)[mask])
    
    # Loops through each month of data, multiply flux /m2/s with area then
    # applies mask. 
    sumcol = []
    for fi,fil in enumerate(fils_filt):
        ds = xray.open_dataset(fil) 
        for t in range(len(ds.time.values)):
            arr = ds.flux.values[:,:,t]*area        
            sumcol.append(np.sum(np.ravel(arr)[mask]))
    sumcol = np.asarray(sumcol)
    
    # Convert seconds into years, mols into kg to give
    # flux in kg per month
    sumcol = sumcol*2628000
    sumcol=(sumcol*43.013)/1000
    
    return sumcol,totarea


class ODR_run(object):
    """ ODR class. Stores the results of the orthogonal distance regression in
    raw and subsetted form, and contains various plotting functions."""
    def __init__(self,data):
        df = data.data
        # Create empty dataframe to store results
        columns = ['gradient','observations','ocean_mf','se','res_var']
        index = np.arange('2006-01', '2014-01', dtype='datetime64[M]')
        odr_df = pd.DataFrame(index=index, columns=columns)
        output = {}
        data_all = {}
        for idx,month in enumerate(odr_df.index):
            # Mask all data but one month
            msk = filt(df,year=month.year,month=month.month)
            if np.count_nonzero(msk)>0:  
                # Performs regression and stores results in dataframe
                minVal = min(df['Agage_corr'])    
                odr_output,odr_data_df = odreg(df['Ocean'][msk],df['Agage_corr'][msk],
                                   df['Ocean stdev'][msk],
                                   df['Agage_error'][msk] + (minVal*-1))
                odr_df.gradient[idx]= odr_output.beta[0]
                odr_df.observations[idx] = len(odr_output.y)       
                odr_df.ocean_mf[idx] = np.mean(df['Ocean'][msk])
                odr_df.se[idx] = odr_output.sd_beta[0]+odr_output.sd_beta[1]
                odr_df.res_var[idx] = odr_output.res_var
                
                # Stores data dataframe and full ODR output in dictionaries
                datestr = month.strftime('%Y-%m')
                output[datestr]=odr_output 
                data_all[datestr]=odr_data_df
        # Store results in object
        self.output = odr_df
        self.output_raw = output
        self.data = data_all
        self.years = np.unique(self.output.index.year)
        
        if data.site == 'MHD':
            self.domain = 'EUROPE'
        elif data.site == 'CGO':
            self.domain = 'AUSTRALIA'
        elif data.site == 'THD':
            self.domain = 'WESTUSA'
        elif data.site == 'RPB':
            self.domain = 'CARIBBEAN'
        elif data.site == 'SMO':
            self.domain = 'PACIFIC'
        
    def add_scaling_factors(self,labo,lobo):
        """ Calculates the sum of the flux within the sampled area and applies
        the scaling factor calculated by the regression to the flux"""
        
        # Locates the relevant data files
        pathName = '/shared_data/air/shared/NAME/emissions/'+self.domain
        searchString = 'n2o-ocean-tot-wk92'
        fils = listFilesInDirectory(pathName,searchString)
        
        # Calculates the total flux in the sampled area /km2/month. Lat/Lon
        # limits are supplied to reduce box to area sampled by site.
        sumcol,totarea = calc_flux(self.years,fils,self.domain,labo,lobo)
        # Divides total flux by area to give flux /km2/month
        sumcol=sumcol/(totarea/1000)

        # Adds flux to ODR dataframe and applies scaling factor
        sumdf = pd.DataFrame(sumcol,index=self.output.index.values,columns=['sumcol'])
        self.output['old_ocean'] = sumdf['sumcol'].values
        self.output['new_ocean'] = self.output['gradient'].values*sumdf['sumcol'].values
        
        
    def plot_scatter(self,save=True):
        """Plots scatter for all months"""
        for month in self.output_raw.keys():
            data = self.data[month]
            fitx = self.output_raw[month].xplus
            fity = self.output_raw[month].y
            
            fig = data.plot(kind='scatter', x='predictions', y='measurements')
            fig.set_ylabel(r'atmospheric N$_2$O baseline removed (mf)')
            fig.set_xlabel(r'predicted atmospheric N$_2$O from ocean model (mf)')
            fig = plt.plot(fitx, fity,c='green', linewidth=2)
            plt.title('scatter showing results of ODR for '+month)
            fignam = '/home/lb16998/work/thesis_images/scatters/'+month+'.png'
            if save == True:
                plt.savefig(fignam)

    def plot_whisker(self,field='gradient'):
        """ Plots box and whisker plot of results. Two options, 'gradient', which
        plots scaling factors and 'new_ocean' which plots applied scaling factors
        with overlayed scatter of original values."""
        # Converts dataframe to list of gradients required for boxplot input.
        final_list = []
        for mm in range(1,13):
            final_list.append(self.output[self.output.index.month==mm][field].values.tolist())    
            # Create a figure instance
        fig = plt.figure(figsize=(8, 6))
            
        plt.grid(True)
        # Plots gradient
        if field=='gradient':
            plt.yticks(np.arange(-11, 11, 2.0))
            plt.title('Box and Whisker plot of regression slopes aranged by month')
            ax = fig.add_subplot(111)
            ax.set_xlabel('months')
            ax.set_ylabel('regression slope')
            bp = ax.boxplot(final_list)  
            ax.set_ylim([-6,15]) 
            #x1,x2,y1,y2 = plt.axis()
            #plt.axis([x1,x2,-6,15])
        
        # Plots new ocean
        elif field=='new_ocean':
            # Calculate average of flux aves to overly onto box/whisker plot.
            ave = []
            for mnth in range(1,13):
                ave.append(np.mean(self.output['old_ocean'][self.output.index.month==mnth].values))
            ave = np.asarray(ave)
            #ave = ave*1000
            
            #plt.yticks(np.arange(-0.015,0.01, 0.005))
            plt.title('Box and Whisker plot of total flux with scaling factor applied arranged by month')
            ax = fig.add_subplot(111)
            ax.set_xlabel('months')
            ax.set_ylabel('n$_2$o flux $kg$ $km^-$$^1$ $month^-$$^1$')
            #ax.set_ylim([-0.05,0.05])
            bp = ax.boxplot(final_list)  
            plt.scatter(np.arange(1,13,1),ave) 
            
    def plot_sf_with_ebars(self,month_range):   
        """Creates a plot with four graphs displaying the scaling factor for 
        each year with the error bars."""
        years = self.years
        
        fig,axs = plt.subplots(2,2)
        fig.set_figheight(10)
        fig.set_figwidth(10) 
        
        for month in range(month_range[0],month_range[-1]):
            ii = 0
            arrT = self.output[self.output.index.month == month]
            arr = np.zeros((len(years),3))
            for year in range(np.min(years),np.max(years)+1):
                gradient = arrT[arrT.index.year==year]['gradient'].values[0]
                SE = arrT[arrT.index.year==year]['se'].values[0]                
                arr[ii,0] = gradient-(2*SE)
                arr[ii,1] = gradient
                arr[ii,2] = gradient+(2*SE)
                ii = ii+1
            month_string = calendar.month_name[month]
            if month - month_range[0] == 0:
                pidx = [0,0]
            elif month - month_range[0] == 1:
                pidx = [0,1]
            elif month - month_range[0] == 2:
                pidx = [1,0]
            elif month - month_range[0] == 3:
                pidx = [1,1]
                
            axs[pidx[0],pidx[1]].scatter(years,arr[:,1])
            for yri,yr in enumerate(years):
                axs[pidx[0],pidx[1]].plot((yr,yr),(arr[yri,0],arr[yri,2]),'-',color='royalblue')
                axs[pidx[0],pidx[1]].plot((yr-0.05,yr+0.05),(arr[yri,2],arr[yri,2]),'-',color='royalblue')
                axs[pidx[0],pidx[1]].plot((yr-0.05,yr+0.05),(arr[yri,0],arr[yri,0]),'-',color='royalblue')
            axs[pidx[0],pidx[1]].set_title(month_string)
            axs[pidx[0],pidx[1]].set_ylabel('scaling factor')
            axs[pidx[0],pidx[1]].set_xlabel('year')
        
    def plot_flux_bar(self,labo,lobo,month_range):
        """ Plots bar graphs (4 per chart) of total n2o flux per month."""
        pathName = '/shared_data/air/shared/NAME/emissions/AUSTRALIA/'
        searchString = 'n2o-ocean-tot-wk92'
        fils = listFilesInDirectory(pathName,searchString)
        years = self.years
        sumcol,totarea = calc_flux(years,fils,labo,lobo)
        
        fig,axs = plt.subplots(2,2)
        fig.set_figheight(12)
        fig.set_figwidth(12)
        width = 0.8
        
        for i, month in enumerate(range(month_range[0],month_range[-1])):
            month_string = calendar.month_name[month]
            if month - month_range[0] == 0:
                pidx = [0,0]
            elif month - month_range[0] == 1:
                pidx = [0,1]
            elif month - month_range[0] == 2:
                pidx = [1,0]
            elif month - month_range[0] == 3:
                pidx = [1,1]
                
            idx =  np.where(self.ODR_output.index.month==month)
            sumarr = sumcol[idx]
        
            axs[pidx[0],pidx[1]].bar(years,sumarr[:,i],width,color='royalblue')
            axs[pidx[0],pidx[1]].set_title(month_string)
            axs[pidx[0],pidx[1]].set_ylabel('$kg$ $month^-$$^1$')
            axs[pidx[0],pidx[1]].set_xlabel('year')
            
            
class OZODR(ODR_run):
    def __init__(self,data):
        ODR_run.__init__(self,data)
        self.output = self.output[self.output.index.year>=2008]
        self.years = np.unique(self.output.index.year)
        #odr.add_scaling_factors([-180,-30],[-360,165])
        


#EU odr.add_scaling_factors([27.5,180],[-360,-5])
    