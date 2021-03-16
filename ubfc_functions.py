from __future__ import division, print_function
import numpy
import scipy.sparse as sp
from scipy.sparse import linalg
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.signal import butter, lfilter
from scipy import signal
from scipy.stats import pearsonr, skew
from sklearn.metrics import mean_squared_error
import pandas as pd
import xlsxwriter
from openpyxl import load_workbook
from statsmodels.tsa.stattools import ccovf
from scipy.interpolate import interp1d

def plotMetrics(train_df,val_df,savingPath,fold):
    '''
    This function plot the Histograms of time and SNR of a train and validation datasets
    Parameters
    ----------
    train_df : pandas data frame
        train dataset.
    val_df : pandas data frame
        validation dataset.
    savingPath : string
        Path to save the results.
    fold : int
        Current fold.
    Returns
    -------
    None.
    '''
    PathFold = os.path.join(savingPath,str(fold))
    if not os.path.exists(PathFold): os.makedirs(PathFold)#Folder where data will be saved
    
    train_df = train_df.reset_index().copy()
    val_df = val_df.reset_index().copy()
    ####################
    # HISTOGRAM OF TIME OF SIGNALS
    #######
    # TRAINING SET    
    tr_time_list = []
    for i in range(0,len(train_df)):
        tr_time_list.append(len(train_df['rppg'][i])/train_df['rppg_fr'][i]) 
        
    # VALIDATION SET
    val_time_list = []
    for i in range(0,len(val_df)):
        val_time_list.append(len(val_df['rppg'][i])/val_df['rppg_fr'][i]) 

    # SAVING
    fig1, ax1 = plt.subplots(2)
    ax1[0].figure,ax1[0].hist(tr_time_list, bins=10),ax1[0].grid(True)
    ax1[0].set_title('Training: Time histogram, Total: %0.2fs, skewness: %0.2f'%(np.sum(tr_time_list),skew(tr_time_list))) 
    ax1[1].figure,ax1[1].hist(val_time_list, bins=10),ax1[1].grid(True)
    ax1[1].set_title('Validation: Time histogram, Total: %0.2fs, skewness: %0.2f'%(np.sum(val_time_list),skew(val_time_list)))
    manager = plt.get_current_fig_manager()
    manager.window.showMaximized()
    plt.show(),plt.pause(0.1)  
    plt.savefig(os.path.join(PathFold,'Hist_time_'+str(fold)),bbox_inches = 'tight')
    
    ####################
    # HISTOGRAM OF NOISE
    ####################
    # TRAINING SET - rPPG & gt
    tr_rppg_SNR = [];tr_gt_SNR = [] 
    for i in range(0,len(train_df)):
        for j in train_df['rppg_SNR'][i]:
            tr_rppg_SNR.append(j)
        for j in train_df['gt_SNR'][i]:
            tr_gt_SNR.append(j)   
            
    # VALIDATION SET - rPPG & gt
    val_rppg_SNR = [];val_gt_SNR = [] 
    for i in range(0,len(val_df)):
        for j in val_df['rppg_SNR'][i]:
            val_rppg_SNR.append(j)
        for j in val_df['gt_SNR'][i]:
            val_gt_SNR.append(j)   

    # SAVING
    fig2, ax2 = plt.subplots(2,2)
    ax2[0,0].figure,ax2[0,0].hist(tr_rppg_SNR, bins=10),ax2[0,0].grid(True)
    ax2[0,0].set_title('Tr: rPPG SNR hist, mean: %0.2fdb, std: %0.2f'%(np.mean(tr_rppg_SNR),np.std(tr_rppg_SNR))) 
    ax2[0,1].figure,ax2[0,1].hist(tr_gt_SNR, bins=10),ax2[0,1].grid(True)
    ax2[0,1].set_title('Tr: gt SNR hist, mean: %0.2fdb, std: %0.2f'%(np.mean(tr_gt_SNR),np.std(tr_gt_SNR))) 
    ax2[1,0].figure,ax2[1,0].hist(val_rppg_SNR, bins=10),ax2[1,0].grid(True)
    ax2[1,0].set_title('Val: rPPG SNR hist, mean: %0.2fdb, std: %0.2f'%(np.mean(val_rppg_SNR),np.std(val_rppg_SNR))) 
    ax2[1,1].figure,ax2[1,1].hist(val_gt_SNR, bins=10),ax2[1,1].grid(True)
    ax2[1,1].set_title('Val: gt SNR hist, mean: %0.2fdb, std: %0.2f'%(np.mean(val_gt_SNR),np.std(val_gt_SNR))) 
    manager = plt.get_current_fig_manager()
    manager.window.showMaximized()
    plt.show(),plt.pause(0.1)  
    plt.savefig(os.path.join(PathFold,'Hist_SNR_'+str(fold)),bbox_inches = 'tight')
    
    ####################
    # HISTOGRAM OF MSE
    ####################
    # TRAINING SET    
    tr_MSE = [] 
    for i in range(0,len(train_df)):
        for j in train_df['MSE'][i]:
            tr_MSE.append(j) 
            
    # VALIDATION SET 
    val_MSE = []  
    for i in range(0,len(val_df)):
        for j in val_df['MSE'][i]:
            val_MSE.append(j)   

    # SAVING
    fig3, ax3 = plt.subplots(2)
    ax3[0].figure,ax3[0].hist(tr_MSE, bins=10),ax3[0].grid(True)
    ax3[0].set_title('Training: MSE histogram, skewness: %0.2f'%skew(tr_MSE))
    ax3[1].figure,ax3[1].hist(val_MSE, bins=10),ax3[1].grid(True)
    ax3[1].set_title('Validation: MSE histogram, skewness: %0.2f'%skew(val_MSE))
    manager = plt.get_current_fig_manager()
    manager.window.showMaximized()
    plt.show(),plt.pause(0.1)  
    plt.savefig(os.path.join(PathFold,'Hist_MSE_'+str(fold)),bbox_inches = 'tight')  

    ####################
    # HISTOGRAM OF r (pearson's correlation between PPG and rPPG)
    ####################
    # TRAINING SET    
    tr_r = [] 
    for i in range(0,len(train_df)):
        for j in train_df['r'][i]:
            tr_r.append(j) 
            
    # VALIDATION SET 
    val_r = []  
    for i in range(0,len(val_df)):
        for j in val_df['r'][i]:
            val_r.append(j)   

    # SAVING
    fig4, ax4 = plt.subplots(2)
    ax4[0].figure,ax4[0].hist(tr_r, bins=10),ax4[0].grid(True)
    ax4[0].set_title('Training: r histogram, skewness: %0.2f'%skew(tr_r))
    ax4[1].figure,ax4[1].hist(val_r, bins=10),ax4[1].grid(True)
    ax4[1].set_title('Validation: r histogram, skewness: %0.2f'%skew(val_r))
    manager = plt.get_current_fig_manager()
    manager.window.showMaximized()
    plt.show(),plt.pause(0.1)  
    plt.savefig(os.path.join(PathFold,'Hist_r_'+str(fold)),bbox_inches = 'tight') 
    
    ###################
    # SAVE EXCEL FILE
    datos = pd.DataFrame(columns=['METHOD',str(fold)+'_tr',str(fold)+'_val'])
    datos = datos.append({'METHOD': 'MSE',str(fold)+'_tr': str(np.round(np.mean(tr_MSE),2)) + u"\u00B1" + str(np.round(np.std(tr_MSE),2)),
                          str(fold) +'_val': str(np.round(np.mean(val_MSE),2)) + u"\u00B1" + str(np.round(np.std(val_MSE),2))},ignore_index=True)
    datos = datos.append({'METHOD': 'r',str(fold)+'_tr': str(np.round(np.mean(tr_r),2)) + u"\u00B1" + str(np.round(np.std(tr_r),2)),
                          str(fold) +'_val': str(np.round(np.mean(val_r),2)) + u"\u00B1" + str(np.round(np.std(val_r),2))},ignore_index=True)
    datos = datos.append({'METHOD': 'gtSNR',str(fold)+'_tr': str(np.round(np.mean(tr_gt_SNR),2)) + u"\u00B1" + str(np.round(np.std(tr_gt_SNR),2)),
                          str(fold) +'_val': str(np.round(np.mean(val_gt_SNR),2)) + u"\u00B1" +  str(np.round(np.std(val_gt_SNR),2))},ignore_index=True)
    datos = datos.append({'METHOD': 'rppgSNR',str(fold)+'_tr': str(np.round(np.mean(tr_rppg_SNR),2)) + u"\u00B1" + str(np.round(np.std(tr_rppg_SNR),2)),
                          str(fold) +'_val': str(np.round(np.mean(val_rppg_SNR),2)) + u"\u00B1" + str(np.round(np.std(val_rppg_SNR)))},ignore_index=True)
    datos = datos.append({'METHOD': 'Time[m]',str(fold)+'_tr': str(np.round(np.sum(tr_time_list)/60,2)),
                          str(fold) +'_val': str(np.round(np.sum(val_time_list)/60,2))},ignore_index=True)      
    
    savingPath = os.path.join(savingPath,'datainfo.xlsx')
    if not os.path.exists(savingPath):       
        writer = pd.ExcelWriter(savingPath, engine = 'xlsxwriter')
        datos.to_excel(writer, sheet_name = str(fold),index=False,float_format = "%0.2f")
        writer.save()
        writer.close()
    else:
        book = load_workbook(savingPath)        
        writer = pd.ExcelWriter(savingPath, engine = 'openpyxl')
        writer.book = book        
        datos.to_excel(writer, sheet_name = str(fold),index=False,float_format = "%0.2f")
        writer.save()
        writer.close()
    
        

def get_HR_SNR_r_MSE(pulseTrace,gtTrace,Fs=25,winLengthSec=15,stepSec=0.5,lowF=0.7,upF=3.5,VERBOSE=0):
    '''
    Parameters
    ----------
    pulseTrace : array
        rppg traces.
    gtTrace : array
        ground truth traces.
    Fs : int, optional
        Frequency of rppg and gt. The default is 25.
    winLengthSec : int, optional
        Length of the sliding windows in seconds. The default is 15.        
    stepSec : int, optional
        length of the step to take between windows, in seconds. The default is 0.5. 
    lowF : int, optional
        low frequency for HR measurement. The default is 0.7
    upF : int, optinal
        up frequency for HR measurement. The defaul is 3.5
    Returns
    -------
    rPPG_HR: Heart rate of rPPG traces per window.
    PPG_HR: Heart rate of PPG traces per window.
    rPPG_SNR: Signal to noise Ratio of rPPG.
    PPG_SNR: Signal to noise Ratio of PPG
    r: Pearson's correlation between rPPG and PPG

    '''
    #IF rppg is exactly winLengthSec, add one more value to get into the function
    if np.size(pulseTrace)<=winLengthSec*Fs:
        if np.size(pulseTrace)<winLengthSec*Fs:
            print('Can not measure metrics because signals is shorter than %i seconds'%winLengthSec)
        elif np.size(pulseTrace)==winLengthSec*Fs:
            pulseTrace = np.append(pulseTrace,pulseTrace[-1]).copy()
            gtTrace = np.append(gtTrace,gtTrace[-1]).copy()
            
    
    pulseTrace = np.asarray(pulseTrace).copy()
    gtTrace = np.asarray(gtTrace).copy()
    # CALCULE Timetrace of rPPG with its frequency 
    timeTrace = np.zeros(pulseTrace.size)
    for j in range(0,len(timeTrace)):
        timeTrace[j] = j*(1/Fs)
    
    # Calculate timeTrace of PPG with its frequency
    gtTime = timeTrace
    
    traceSize = len(pulseTrace);
    winLength = round(winLengthSec*Fs)# length of the sliding window for FFT
    step = round(stepSec*Fs);# length of the steps for FFT
    halfWin = (winLength/2);
    
    show1window = True
    rPPG_SNR = []; PPG_SNR = []
    rPPG_HR = []; PPG_HR = []
    Pearsonsr = []
    MSE = []
    cont=0
    for i in range(int(halfWin),int(traceSize-halfWin),int(step)):#for i=halfWin:step:traceSize-halfWin
        #Uncomment next three lines just to debug
        #if cont == 90:
        #    print('error')
        #print(cont);cont=cont+1
        
        ###
        # GET CURRENT WINDOW
        ## get start/end index of current window
        startInd = int(i-halfWin) #startInd = i-halfWin+1;
        endInd = int(i+halfWin) # endInd = i+halfWin;
        startTime = int(timeTrace[startInd]) # startTime = timeTrace(startInd);
        endTime = int(timeTrace[endInd]) #timeTrace(endInd);
        # get current pulse window
        crtPulseWin = pulseTrace[startInd:endInd]# crtPulseWin = pulseTrace(startInd:endInd);
        crtTimeWin = timeTrace[startInd:endInd]# crtTimeWin = timeTrace(startInd:endInd);
        # get current PPG window
        startIndGt = startInd # [~, startIndGt] = min(abs(gtTime-startTime));
        endIndGt = endInd # [~, endIndGt] = min(abs(gtTime-endTime));
        crtPPGWin = gtTrace[startIndGt:endIndGt]
        crtTimePPGWin = gtTime[startIndGt:endIndGt]
        # get exact PPG Fs
        # Fs_PPG = 1/mean(diff(crtTimePPGWin));       
        if VERBOSE>0 and show1window==True: pltnow(crtPulseWin,crtPPGWin,val=2,fr=Fs)
        
        #########################
        # rPPG: SPECTRAL ANALYSIS
        ### rPPG: Get spectrum by Welch
        # Get power spectral density in Frequency of HR in humans [0.7-3.5]
        rppg_freq_w, rppg_power_w = signal.welch(crtPulseWin, fs=Fs)
        rppg_freq2 = [item1 for item1 in rppg_freq_w if item1 > lowF and item1 < upF]
        rppg_power2 = [item2 for item1,item2 in zip(rppg_freq_w,rppg_power_w) if item1 > lowF and item1 < upF]
        rppg_freq_w = rppg_freq2.copy();rppg_power_w = rppg_power2.copy()
        # Find highest peak in the spectral density and take its frequency value
        loc = detect_peaks(np.asarray(rppg_power_w), mpd=1, edge='rising',show=False)
        if loc.size == 0 :# If no peak was found
            loc = np.array([0])
        loc = loc[np.argmax(np.array(rppg_power_w)[loc])]#just highest peak

        rPPG_peaksLoc = np.asarray(rppg_freq_w)[loc]
        if VERBOSE>0 and show1window==True: 
            plt.figure(),plt.title('rPPG Spectrum,welch'),plt.plot(rppg_freq_w,rppg_power_w)
            plt.axvline(x=rPPG_peaksLoc,ymin=0,ymax=1,c='r'),plt.show(),plt.pause(1)
                
        # YB: SNR is more intersting with FFT so we get spectra again (I don't care about processing cost :)        
        width = 0.4
        # rPPG: Get spectrum by FFT        
        N = len(crtPulseWin)*3
        rppg_freq = np.arange(0,N,1)*Fs/N#freq=[0 : N-1]*Fs/N;
        rppg_power = np.abs(np.fft.fft(crtPulseWin,N))**2#power = abs(fft(x,N)).^2;
        rppg_freq2 = [item1 for item1 in rppg_freq if item1 > lowF and item1 < upF]
        rppg_power2 = [item2 for item1,item2 in zip(rppg_freq,rppg_power) if item1 > lowF and item1 < upF]
        rppg_freq = rppg_freq2.copy();rppg_power = rppg_power2.copy()
        if VERBOSE>0 and show1window==True: plt.figure(),plt.title('rPPG Spectrum,FFT'),plt.plot(rppg_freq,rppg_power),plt.show(),plt.pause(1)        
        
        #rPPG: SNR
        range1 = [((i>(rPPG_peaksLoc-width/2))and(i<(rPPG_peaksLoc+width/2))) for i in rppg_freq]
        #if VERBOSE>0 and show1window==True: plt.figure(),plt.title('rPPG:range 1'),plt.plot(rppg_freq,range1),plt.show(),plt.pause(1)
        range2 = [((i>((rPPG_peaksLoc*2)-(width/2)))and(i<((rPPG_peaksLoc*2)+(width/2)))) for i in rppg_freq]
        #if VERBOSE>0 and show1window==True: plt.figure(),plt.title('rPPG:range 2'),plt.plot(rppg_freq,range2),plt.show(),plt.pause(1)
        rango = np.logical_or(range1, range2)
        #if VERBOSE>0 and show1window==True: plt.figure(),plt.title('rPPG:range'),plt.plot(rppg_freq,rango),plt.show(),plt.pause(1)
        Signal = rppg_power*rango # signal = rPPG_power.*range;
        Noise = rppg_power*~rango #noise = rPPG_power.*(~range);
        if VERBOSE>0 and show1window==True: plt.figure(),plt.title('rPPG:Signal'),plt.plot(rppg_freq,Signal),plt.show(),plt.pause(1)
        if VERBOSE>0 and show1window==True: plt.figure(),plt.title('rPPG:Noise'),plt.plot(rppg_freq,Noise),plt.show(),plt.pause(1)
        n = np.sum(Noise) # n = sum(noise);
        s = np.sum(Signal) # s = sum(signal);
        snr = 10*np.log10(s/n) # snr(ind) = 10*log10(s/n);
        rPPG_SNR.append(snr)
        
        #rPPG: HR
        rPPG_HR.append(rPPG_peaksLoc*60)#rPPG_peaksLoc(1)*60;
        
        #########################
        # PPG: SPECTRAL ANALYSIS
        ### PPG: Get spectrum by Welch       
        # Get power spectral density in Frequency of HR in humans [0.7-3.5]
        ppg_freq_w, ppg_power_w = signal.welch(crtPPGWin, fs=Fs)
        ppg_freq2 = [item1 for item1 in ppg_freq_w if item1 > lowF and item1 < upF]
        ppg_power2 = [item2 for item1,item2 in zip(ppg_freq_w,ppg_power_w) if item1 > lowF and item1 < upF]
        ppg_freq_w = ppg_freq2.copy();ppg_power_w = ppg_power2.copy()
        # Find highest peak in the spectral density and take its frequency value
        loc = detect_peaks(np.asarray(ppg_power_w), mpd=1, edge='rising',show=False)
        if loc.size == 0 :# If no peak was found
            loc = np.array([0])
        loc = loc[np.argmax(np.array(ppg_power_w)[loc])]#just highest peak
        PPG_peaksLoc = np.asarray(ppg_freq_w)[loc]
        if VERBOSE>0 and show1window==True:
            plt.figure(),plt.title('PPG Spectrum,welch'),plt.plot(ppg_freq_w,ppg_power_w)
            plt.axvline(x=PPG_peaksLoc,ymin=0,ymax=1,c='r'),plt.show(),plt.pause(1)
        
        # YB: SNR is more intersting with FFT so we get spectra again (I don't care about processing cost :)        
        width = 0.4      
        # PPG: get spectrum by FFT
        N = len(crtPPGWin)*3;
        ppg_freq = np.arange(0,N,1)*Fs/N
        ppg_power = np.abs(np.fft.fft(crtPPGWin,N))**2
        ppg_freq2 = [item1 for item1 in ppg_freq if item1 > lowF and item1 < upF]
        ppg_power2 = [item2 for item1,item2 in zip(ppg_freq,ppg_power) if item1 > lowF and item1 < upF]
        ppg_freq = ppg_freq2.copy();ppg_power = ppg_power2.copy()
        if VERBOSE>0 and show1window==True: plt.figure(),plt.title('PPG Spectrum,FFT'),plt.plot(ppg_freq,ppg_power),plt.show(),plt.pause(1)        
        
        range1 = [((i>(PPG_peaksLoc-width/2))and(i<(PPG_peaksLoc+width/2))) for i in ppg_freq]
        #if VERBOSE>0 and show1window==True: plt.figure(),plt.title('PPG:range 1'),plt.plot(ppg_freq,range1),plt.show(),plt.pause(1)
        range2 = [((i>((PPG_peaksLoc*2)-(width/2)))and(i<((PPG_peaksLoc*2)+(width/2)))) for i in ppg_freq]
        #if VERBOSE>0 and show1window==True: plt.figure(),plt.title('PPG:range 2'),plt.plot(ppg_freq,range2),plt.show(),plt.pause(1)
        rango = np.logical_or(range1, range2)
        #if VERBOSE>0 and show1window==True: plt.figure(),plt.title('PPG:range'),plt.plot(ppg_freq,rango),plt.show(),plt.pause(1)
        Signal = ppg_power*rango # signal = rPPG_power.*range;
        Noise = ppg_power*~rango #noise = rPPG_power.*(~range);
        if VERBOSE>0 and show1window==True: plt.figure(),plt.title('PPG:Signal'),plt.plot(ppg_freq,Signal),plt.show(),plt.pause(1)
        if VERBOSE>0 and show1window==True: plt.figure(),plt.title('PPG:Noise'),plt.plot(ppg_freq,Noise),plt.show(),plt.pause(1)
        n = np.sum(Noise) # n = sum(noise);
        s = np.sum(Signal) # s = sum(signal);
        snr = 10*np.log10(s/n) # snr(ind) = 10*log10(s/n);
        PPG_SNR.append(snr)
        
        #PPG: HR
        PPG_HR.append(PPG_peaksLoc*60)#rPPG_peaksLoc(1)*60;

        ##############
        # rPPG vs PPG:
        # Pearson's correlation between rPPG and PPG
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.pearsonr.html)
        pearsonsr,_ = pearsonr(crtPulseWin, crtPPGWin)
        Pearsonsr.append(pearsonsr)
        
        ###########
        # Mean Square Error (MSE)
        MSE.append(mean_squared_error(crtPPGWin,crtPulseWin))
        
        show1window=False # Just plot first window

    return np.asarray(rPPG_HR),np.asarray(PPG_HR),np.asarray(rPPG_SNR),np.asarray(PPG_SNR),np.asarray(Pearsonsr),np.asarray(MSE)

# DISCLAIMER: This function is copied from https://github.com/nwhitehead/swmixer/blob/master/swmixer.py, 
#             which was released under LGPL. 
def resample_by_interpolation(signal, input_fs, output_fs):

    scale = output_fs / input_fs
    # calculate new length of sample
    n = round(len(signal) * scale)

    # use linear interpolation
    # endpoint keyword means than linspace doesn't go all the way to 1.0
    # If it did, there are some off-by-one errors
    # e.g. scale=2.0, [1,2,3] should go to [1,1.5,2,2.5,3,3]
    # but with endpoint=True, we get [1,1.4,1.8,2.2,2.6,3]
    # Both are OK, but since resampling will often involve
    # exact ratios (i.e. for 44100 to 22050 or vice versa)
    # using endpoint=False gets less noise in the resampled sound
    resampled_signal = np.interp(
        np.linspace(0.0, 1.0, n, endpoint=False),  # where to interpret
        np.linspace(0.0, 1.0, len(signal), endpoint=False),  # known positions
        signal,  # known data points
    )
    return resampled_signal

def signaltonoise(a, axis, ddof): 
    a = np.asanyarray(a) 
    m = a.mean(axis) 
    sd = a.std(axis = axis, ddof = ddof) 
    return np.where(sd == 0, 0, m / sd) 

def SNR(data):
    data['snr_rppg'] = ""
    data['snr_gt'] = ""
    for i in range(0,len(data)):        
        rppg = data['rppg'].iloc[i]
        gt = data['gt'].iloc[i]
        snr_gt = signaltonoise(gt, axis = 0, ddof = 0)
        snr_rppg = signaltonoise(rppg, axis = 0, ddof = 0)
        data['snr_rppg'].iloc[i] = snr_rppg
        data['snr_gt'].iloc[i] = snr_gt
    return data

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

# -*- coding: utf-8 -*-
def pltnow(variable,variable2=0,val=0,fr=25):
    sr = 1/fr #sf = sampling frequency
    #sr = sampling rate
    if val==0: #simpleplot
        plt.figure(),plt.plot(variable),plt.show(),plt.pause(0.05)
        plt.ylabel('Amplitude')
        #manager = plt.get_current_fig_manager()
        #manager.window.showMaximized()
    elif val==1: #generate traces
        timeTrace = np.zeros(len(variable))
        for j in range(0,len(timeTrace)):
            timeTrace[j] = j*sr#sr=0.04
        plt.figure(),plt.plot(timeTrace,variable),plt.show(),plt.pause(0.05)
        plt.xlabel('time [s]'),plt.ylabel('Amplitude')
        #manager = plt.get_current_fig_manager()
        #manager.window.showMaximized()
    elif val==2:
        timeTrace = np.zeros(len(variable))
        for j in range(0,len(timeTrace)):
            timeTrace[j] = j*sr#sr=0.04
        plt.figure()
        line1, = plt.plot(timeTrace,variable)
        line2, = plt.plot(timeTrace,variable2)
        plt.xlim(timeTrace[0], timeTrace[-1])
        plt.legend([line1, line2], ['signal 1', 'signal 2'], loc='upper right')
        plt.xlabel('time [s]'),plt.ylabel('Amplitude')        
        plt.show(),plt.pause(0.05)
    elif val==3:#plot different-length-signals
        timeTrace = np.zeros(len(variable))
        for j in range(0,len(timeTrace)):
            timeTrace[j] = j*sr#sr=0.04
        timeTrace2 = np.zeros(len(variable2))
        for j in range(0,len(timeTrace2)):
            timeTrace2[j] = j*sr#sr=0.04
        plt.figure()
        plt.plot(timeTrace,variable)
        plt.plot(timeTrace2,variable2)
        plt.xlim(timeTrace[0], np.max((timeTrace[-1],timeTrace2[-1])))
        #plt.legend([line1, line2], ['signal 1', 'signal 2'], loc='upper right')
        plt.xlabel('time [s]'),plt.ylabel('Amplitude')        
        plt.show(),plt.pause(0.05)
        #manager = plt.get_current_fig_manager()
        #manager.window.showMaximized()
        
def saveParameters(diccionario,path,name='parameters'):
    if not os.path.exists(path): os.makedirs(path)
    values = list(diccionario.values())
    variables = list(diccionario.keys())
    f = open(path+"\\"+name,"w")
    for i in range(len(values)):
        f.write(variables[i] + '\t' + str(values[i]) + '\n')
    f.close()

def smooth(x, window_len=11, window='hanning'):
    """smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.

    input:
        x: the input signal
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal

    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)

    see also:

    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter

    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """
    
    if x.ndim != 1:
        raise(ValueError, "smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise(ValueError, "Input vector needs to be bigger than window size.")

    if window_len < 3:
        return x

    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise(ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")

    s = numpy.r_[x[window_len - 1:0:-1], x, x[-2:-window_len - 1:-1]]
    # print(len(s))
    if window == 'flat':  # moving average
        w = numpy.ones(window_len, 'd')
    else:
        w = eval('numpy.' + window + '(window_len)')

    y = numpy.convolve(w / w.sum(), s, mode='valid')
    return y[round((window_len/2-1)):-round((window_len/2))]

def detrendsignal(z):
	T = len(z)
	l=500 #Regulariszer
	
	I = sp.eye(T)
	#The second order difference matrix
	ones = np.squeeze(np.ones([1,T]))
	data = np.array([ones,-2*ones,ones])
	diags = np.array([0, 1, 2])
	data.shape
	D2 = sp.spdiags(data,diags, T-2, T)
	prod = (I+l**2 *(D2.transpose()*D2)).tocsc()
	z_detrended = (I-linalg.inv(prod))*z;
	'''plt.subplot(211)
	plt.plot(z,linewidth=1.0)
	plt.subplot(212)
	plt.plot(z_detrended,linewidth=1.0)
	plt.show()'''
	return z_detrended
'''def main():
	#e.g.
	detrendsignal([125.94,126.06,125.76,125.87,125.84,125.83,125.79,125.77,125.7,125.88,125.73,126.01,125.77,125.79,125.93,125.94,126.11,126.12,126.02,126.23,126.04,125.97,125.91,125.78,125.65,125.89,125.5,125.47,125.8,126.04,125.91,126.12,125.6,125.78,126.16,125.92,125.99,125.95,126.12,125.97,125.8,125.91,125.9,125.87,125.51,125.73,125.63,125.3,125.58,125.67,125.73,125.67,125.48,125.59,125.35,125.51,125.65,125.67,125.64,125.65,126.01,125.78,125.68,125.38,125.56,125.37,125.43,125.32,125.55,125.27,125.11,125.1,125.43,125.66,125.48,125.31,125.55,125.31,125.67,125.21,125.23,125.43,125.27,124.97,125.06,125.27,125.15,125.22,125.21,125.2,125.22,125.28,125.5,125.24,125.37,125.37,125.36,125.24,125.2,125.51,125.23,125.3,125.32,125.19,125.37,125.37,125.57,125.56,125.39,125.39,125.54,125.63,125.96,125.77,125.69,125.71,125.71,125.55,125.65,125.25,125.31,125.45,125.52,125.54,125.47,125.3,125.11,125.48,125.3,125.56,125.49,125.74,125.81,126,125.71,125.55,125.54,125.4,125.64,125.59,125.47,125.63,125.63,125.82,125.26,125.34,125.48,125.61,125.79,125.83,125.97,125.98,125.7,125.81,125.63,125.57,125.72,125.81,125.89,125.46,125.76,125.83,125.86,125.99,125.97,126.17,126.37,126.32,126.15,126.02,125.92,125.91,125.81,126.18,126.45,126.2,126.1,126,125.75,125.75,125.57,125.83,125.93,125.8,125.77,125.52,125.92,126,125.72,125.75,125.86,125.6,125.78,125.7,125.5,125.33,125.17,125.28,125.31,125.85,125.45,125.61,125.41,125.5,125.65,125.59,125.66,125.36,125.24,125.28,125.31,125.22,125.4,125.15,125.23,125.21,125.48,125.29,125.25,125.33,125.54,125.71,125.54,125.35,125.15,125.46,125.1,124.89,124.94,124.98,125.17,125.29,125.16,125.31,125.08,125.01,124.95,125.04,125.18,125.11,125.4,125.41,125.64,125.27,125.36,125.24,124.87,125.07,124.95,125.04,124.97,125.2,125.09,125.36,125.32,125.5,125.25,125.56,125.41,125.26,125.31,125.01,125.04,125.17,125.13,125.19,124.94,125.08,125.22,125.26,125.53,125.43,125.12,125.16,125.4,125.36,125.43,125.17,125.28,125.26,125.05,125.15,125.17,125.01,125.33,125.37,125.35,125.27,125.08,125.31,125.18,125.45,125.51,125.36,125.45,125.25,125.32,125,125.07,125.32,125.21,125.19,125.31,125.17,125.39,125.57,125.35,125.61,125.42,125.36,125.3,125.38,125.13,125.44,125.52,125.44,125.28,125.53,125.54,125.62,125.76,125.44,125.75,125.52,125.8,125.86,125.62,125.44,125.64,125.3,125.66,125.3,125.59,125.94,125.43,125.63,125.73,125.83,125.77,125.64,126.1,125.55,125.8,125.62,125.61,125.57,125.35,125.82,125.64,125.52,125.62,125.45,125.68,125.84,125.75,125.9,125.92,125.96,125.61,125.88,125.64,125.5,125.84,125.8,125.76,125.93,125.81,125.96,125.88,126.09,125.83,125.74,125.87,125.73,125.41,126.05,125.62,125.9,125.74,125.91,125.59,126.08,125.79,125.9,125.83,125.69,125.97,125.69,125.77,125.43,125.48,125.62,125.55,125.34,125.45,125.4,125.46,125.63,125.4,125.13,125.51,125.35,125.45,125.27,125.66,125.16,125.29,125.05,124.88,125.48,125.72,125.5,125.45,125.5,125.24,125.31,125.29,125.23,125.41,125.61,125.52,125.74,125.88,125.22,125.29,125.24,125.42,125.02,125.21,125.14,125,125.21,125.25,125.12,125.42,125.33,125.47,125.28,125.37,125.53,125.21,125.23,125.07,125.22,125.09,125.02,125.19,125.15,125.31,125.21,124.99,124.97,125.41,125.33,125.18,125.3,125.28,125.26,125.3,125.24,125.48,125.23,125.26,125.27,125.43,125.27,125.56,125.26,125.45,125.6,125.56,125.57,125.31,125.35,125.29,124.96,125.23,125.21,125.31,125.16,125.41,125.42,125.27,125.32,125.66,125.79,125.63,125.81,125.81,125.66,125.47,125.5,125.37,125.41,125.07,125.16,125.17,125.29,125.39,125.24,125.1,125.4,125.6,125.63,125.87,125.36,125.4,125.32,125.21,125.13,125.15,125.05])

main()'''

# %load ./../functions/detect_peaks.py
"""Detect peaks in data based on their amplitude and other features."""


import numpy as np

__author__ = "Marcos Duarte, https://github.com/demotu/BMC"
__version__ = "1.0.5"
__license__ = "MIT"


def detect_peaks(x, mph=None, mpd=1, threshold=0, edge='rising',
                 kpsh=False, valley=False, show=False, ax=None):
    """Detect peaks in data based on their amplitude and other features.

    Parameters
    ----------
    x : 1D array_like
        data.
    mph : {None, number}, optional (default = None)
        detect peaks that are greater than minimum peak height (if parameter
        `valley` is False) or peaks that are smaller than maximum peak height
         (if parameter `valley` is True).
    mpd : positive integer, optional (default = 1)
        detect peaks that are at least separated by minimum peak distance (in
        number of data).
    threshold : positive number, optional (default = 0)
        detect peaks (valleys) that are greater (smaller) than `threshold`
        in relation to their immediate neighbors.
    edge : {None, 'rising', 'falling', 'both'}, optional (default = 'rising')
        for a flat peak, keep only the rising edge ('rising'), only the
        falling edge ('falling'), both edges ('both'), or don't detect a
        flat peak (None).
    kpsh : bool, optional (default = False)
        keep peaks with same height even if they are closer than `mpd`.
    valley : bool, optional (default = False)
        if True (1), detect valleys (local minima) instead of peaks.
    show : bool, optional (default = False)
        if True (1), plot data in matplotlib figure.
    ax : a matplotlib.axes.Axes instance, optional (default = None).

    Returns
    -------
    ind : 1D array_like
        indeces of the peaks in `x`.

    Notes
    -----
    The detection of valleys instead of peaks is performed internally by simply
    negating the data: `ind_valleys = detect_peaks(-x)`

    The function can handle NaN's

    See this IPython Notebook [1]_.

    References
    ----------
    .. [1] http://nbviewer.ipython.org/github/demotu/BMC/blob/master/notebooks/DetectPeaks.ipynb

    Examples
    --------

    >>> from detect_peaks import detect_peaks
    >>> x = np.random.randn(100)
    >>> x[60:81] = np.nan
    >>> # detect all peaks and plot data
    >>> ind = detect_peaks(x, show=True)
    >>> print(ind)

    >>> x = np.sin(2*np.pi*5*np.linspace(0, 1, 200)) + np.random.randn(200)/5
    >>> # set minimum peak height = 0 and minimum peak distance = 20
    >>> detect_peaks(x, mph=0, mpd=20, show=True)

    >>> x = [0, 1, 0, 2, 0, 3, 0, 2, 0, 1, 0]
    >>> # set minimum peak distance = 2
    >>> detect_peaks(x, mpd=2, show=True)

    >>> x = np.sin(2*np.pi*5*np.linspace(0, 1, 200)) + np.random.randn(200)/5
    >>> # detection of valleys instead of peaks
    >>> detect_peaks(x, mph=-1.2, mpd=20, valley=True, show=True)

    >>> x = [0, 1, 1, 0, 1, 1, 0]
    >>> # detect both edges
    >>> detect_peaks(x, edge='both', show=True)

    >>> x = [-2, 1, -2, 2, 1, 1, 3, 0]
    >>> # set threshold = 2
    >>> detect_peaks(x, threshold = 2, show=True)

    Version history
    ---------------
    '1.0.5':
        The sign of `mph` is inverted if parameter `valley` is True

    """

    x = np.atleast_1d(x).astype('float64')
    if x.size < 3:
        return np.array([], dtype=int)
    if valley:
        x = -x
        if mph is not None:
            mph = -mph
    # find indices of all peaks
    dx = x[1:] - x[:-1]
    # handle NaN's
    indnan = np.where(np.isnan(x))[0]
    if indnan.size:
        x[indnan] = np.inf
        dx[np.where(np.isnan(dx))[0]] = np.inf
    ine, ire, ife = np.array([[], [], []], dtype=int)
    if not edge:
        ine = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) > 0))[0]
    else:
        if edge.lower() in ['rising', 'both']:
            ire = np.where((np.hstack((dx, 0)) <= 0) & (np.hstack((0, dx)) > 0))[0]
        if edge.lower() in ['falling', 'both']:
            ife = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) >= 0))[0]
    ind = np.unique(np.hstack((ine, ire, ife)))
    # handle NaN's
    if ind.size and indnan.size:
        # NaN's and values close to NaN's cannot be peaks
        ind = ind[np.in1d(ind, np.unique(np.hstack((indnan, indnan - 1, indnan + 1))), invert=True)]
    # first and last values of x cannot be peaks
    if ind.size and ind[0] == 0:
        ind = ind[1:]
    if ind.size and ind[-1] == x.size - 1:
        ind = ind[:-1]
    # remove peaks < minimum peak height
    if ind.size and mph is not None:
        ind = ind[x[ind] >= mph]
    # remove peaks - neighbors < threshold
    if ind.size and threshold > 0:
        dx = np.min(np.vstack([x[ind] - x[ind - 1], x[ind] - x[ind + 1]]), axis=0)
        ind = np.delete(ind, np.where(dx < threshold)[0])
    # detect small peaks closer than minimum peak distance
    if ind.size and mpd > 1:
        ind = ind[np.argsort(x[ind])][::-1]  # sort ind by peak height
        idel = np.zeros(ind.size, dtype=bool)
        for i in range(ind.size):
            if not idel[i]:
                # keep peaks with the same height if kpsh is True
                idel = idel | (ind >= ind[i] - mpd) & (ind <= ind[i] + mpd) \
                       & (x[ind[i]] > x[ind] if kpsh else True)
                idel[i] = 0  # Keep current peak
        # remove the small peaks and sort back the indices by their occurrence
        ind = np.sort(ind[~idel])

    if show:
        if indnan.size:
            x[indnan] = np.nan
        if valley:
            x = -x
            if mph is not None:
                mph = -mph
        _plot(x, mph, mpd, threshold, edge, valley, ax, ind)

    return ind


def _plot(x, mph, mpd, threshold, edge, valley, ax, ind):
    """Plot results of the detect_peaks function, see its help."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print('matplotlib is not available.')
    else:
        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=(8, 4))

        ax.plot(x, 'b', lw=1)
        y=0
        for i in range(len(ind) - 1):
            m = (ind[(len(ind)) - (i + 1)]) - (ind[(len(ind) - (i + 2))])
            y += m
        y = y / (len((ind)) - 1)

        if ind.size:
            label = 'valley' if valley else 'peak'
            label = label + 's' if ind.size > 1 else label
            ax.plot(ind, x[ind], '+', mfc=None, mec='r', mew=2, ms=8,
                    label='%d %s' % (ind.size, label))
            ax.legend(loc='best', framealpha=.5, numpoints=1)
        ax.set_xlim(-.02 * x.size, x.size * 1.02 - 1)
        ymin, ymax = x[np.isfinite(x)].min(), x[np.isfinite(x)].max()
        yrange = ymax - ymin if ymax > ymin else 1
        ax.set_ylim(ymin - 0.1 * yrange, ymax + 0.1 * yrange)
        ax.set_xlabel('Number of Samples', fontsize=14)
        ax.set_ylabel('Amplitude', fontsize=14)
        mode = 'Valley detection' if valley else 'Peak detection'
        ax.set_title("%s (mph=%s, mpd=%d, threshold=%s, edge='%s', average distance='%s')"
                     % (mode, str(mph), mpd, str(threshold), edge, y))
        # plt.grid()
        plt.show()

#Fonction normalisation
def normalize(data):
    
    from sklearn.preprocessing import MinMaxScaler
    x = np.asarray(data)
    x = x.reshape(len(x), 1)
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler = scaler.fit(x)
    scaled_x = scaler.transform(x)
    return scaled_x

def phase_align(reference, target, roi, res=100):
    '''
    Cross-correlate data within region of interest at a precision of 1./res
    if data is cross-correlated at native resolution (i.e. res=1) this function
    can only achieve integer precision 
    Args:
        reference (1d array/list): signal that won't be shifted
        target (1d array/list): signal to be shifted to reference
        roi (tuple): region of interest to compute chi-squared
        res (int): factor to increase resolution of data via linear interpolation
    
    Returns:
        shift (float): offset between target and reference signal 
    '''
    # convert to int to avoid indexing issues
    ROI = slice(int(roi[0]), int(roi[1]), 1)

    # interpolate data onto a higher resolution grid 
    x,r1 = highres(reference[ROI],kind='linear',res=res)
    x,r2 = highres(target[ROI],kind='linear',res=res)

    # subtract mean
    r1 -= r1.mean()
    r2 -= r2.mean()

    # compute cross covariance 
    cc = ccovf(r1,r2,demean=False,unbiased=False)

    # determine if shift if positive/negative 
    if np.argmax(cc) == 0:
        cc = ccovf(r2,r1,demean=False,unbiased=False)
        mod = -1
    else:
        mod = 1

    # often found this method to be more accurate then the way below
    return np.argmax(cc)*mod*(1./res)

    # interpolate data onto a higher resolution grid 
    x,r1 = highres(reference[ROI],kind='linear',res=res)
    x,r2 = highres(target[ROI],kind='linear',res=res)

    # subtract off mean 
    r1 -= r1.mean()
    r1 -= r2.mean()

    # compute the phase-only correlation function
    product = np.fft.fft(r1) * np.fft.fft(r2).conj()
    cc = np.fft.fftshift(np.fft.ifft(product))

    # manipulate the output from np.fft
    l = reference[ROI].shape[0]
    shifts = np.linspace(-0.5*l,0.5*l,l*res)

    # plt.plot(shifts,cc,'k-'); plt.show()
    return shifts[np.argmax(cc.real)]

def highres(y,kind='cubic',res=100):
    '''
    Interpolate data onto a higher resolution grid by a factor of *res*
    Args:
        y (1d array/list): signal to be interpolated
        kind (str): order of interpolation (see docs for scipy.interpolate.interp1d)
        res (int): factor to increase resolution of data via linear interpolation
    
    Returns:
        shift (float): offset between target and reference signal 
    '''
    y = np.array(y)
    x = np.arange(0, y.shape[0])
    f = interp1d(x, y,kind='cubic')
    xnew = np.linspace(0, x.shape[0]-1, x.shape[0]*res)
    ynew = f(xnew)
    return xnew,ynew