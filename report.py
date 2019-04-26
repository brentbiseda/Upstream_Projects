import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns; sns.set()
import matplotlib.ticker as mtick
from numpy.polynomial.polynomial import polyfit
from numpy import inf
import os
import errno
from fpdf import FPDF
from PIL import Image

class Report:
    def GeneratePlots(self, well):
        self.RemoveFiles()
        self.SilentPlot(self.ProductionPlot, well.df)
        self.SilentPlot(self.PZPlot, well.df)
        self.SilentPlot(self.SemilogRateCumPlot, well.df)
        self.SilentPlot(self.SemilogRateTimePlot, well.df)
        self.SilentPlot(self.PressureDerivative, well.df)
        self.SilentPlot(self.LinearSuperpositionTimePlot, well.df)
        self.SilentPlot(self.LiquidLoadingPlot, well.df)
        self.SilentPlot(self.SquareRootTimePlot, well.df)
        self.SilentPlot(self.RateTimePlot, well.df)
        self.MakePDF('figures', well)
    
    def RemoveFiles(self):
        self.SilentRemove('figures/pz.jpg')
        self.SilentRemove('figures/semilogratecum.jpg')
        self.SilentRemove('figures/semilogratetime.jpg')
        self.SilentRemove('figures/productionhistory.jpg')
        self.SilentRemove('figures/linearsuperpositiontime.jpg')
        self.SilentRemove('figures/ratetime.jpg')
        self.SilentRemove('figures/squareroottime.jpg')
        self.SilentRemove('figures/pressurederivative.jpg')
        self.SilentRemove('figures/liquidloading.jpg')
    
    def MakePDF(self, path, well):
    #try:
        self.SilentRemove(path+'/pdfs/'+well.df.WELL_NAME_OPS[0]+'.pdf')

        fileNames = os.listdir(path)
        #fileNames = [file for file in filenames if '.jpg' in file]
        fileNames = [path+'/'+file for file in fileNames if '.jpg' in file]

        pdf = FPDF()

        for file in fileNames:
            pdf.add_page()
            pdf.image(file, x=0, y=0, w=200, h=250)

        pdf.output(path+'/pdfs/'+well.df.WELL_NAME_OPS[0]+'.pdf', 'F')
        #except:
         #   pass
        
    def SilentRemove(self, filename):
        try:
            os.remove(filename)
            #print("removed: " + filename)
        except OSError as e: # this would be "except OSError, e:" before Python 2.6
            if e.errno != errno.ENOENT: # errno.ENOENT = no such file or directory
                raise # re-raise exception if a different error occurred
    
    def SilentPlot(self, fun, *args):
        try:
            fun(*args)
        except:
            pass
        
    def PZPlot(self, df):
        #plt.clf()
        sns.set(style='white')
        fig = plt.figure(figsize=(10,10))
        ax = fig.add_subplot(1,1,1)
        ax.scatter(x='GAS_CUMM_MMCF', y='P_Z', data=df.reset_index(), color='red', label='P/Z',s=5)
        ax.set_ylabel('P/Z [MCFD]')
        ax.set_xlabel('Cumulative Production [MMCF]')
        ax.set_title('P/Z')
        ax.set_ylim(0,)
        ax.set_xlim(0,10000)
        ax.grid(b=True, which='major', color='black', linewidth=0.5)
        ax.legend()
        plt.savefig('figures/pz.jpg', dpi=300)
        pass

    def SemilogRateCumPlot(self, df):
        #plt.clf()
        sns.set(style='white')
        fig = plt.figure(figsize=(10,10))
        ax = fig.add_subplot(1,1,1)
        ax.scatter(x='GAS_CUMM_MMCF', y='Q', data=df.reset_index(), color='red', label='Gas Rate',s=5)
        ax.scatter(x='GAS_CUMM_MMCF', y='AOF', data=df.reset_index(), color='black', label='AOF', s=5)
        ax.scatter(x='GAS_CUMM_MMCF', y='WATER_BBL', data=df.reset_index(), color='blue', label='Water', s=5)
        ax.set_ylabel('Rate [MCFD or BBLD]')
        ax.set_xlabel('Cumulative Production [MMCF]')
        ax.set_title('Semilog Rate Cumulative Plot')
        ax.set(yscale='log')
        ax.set_ylim(1,)
        ax.set_xlim(0,10000)
        ax.grid(b=True, which='major', color='black', linewidth=0.5)
        #ax.grid(b=True, which='minor', color='black', linewidth=0.2)
        ax.legend()
        plt.savefig('figures/semilogratecum.jpg', dpi=300)
        pass

    def SemilogRateTimePlot(self, df):
        #plt.clf()
        sns.set(style='white')
        fig = plt.figure(figsize=(10,10))
        ax = fig.add_subplot(1,1,1)
        ax.scatter(x='PRODUCTION_DAY_GAS_COUNTER', y='Q', data=df.reset_index(), color='red', label='Gas Rate',s=5)
        ax.scatter(x='PRODUCTION_DAY_GAS_COUNTER', y='AOF', data=df.reset_index(), color='black', label='AOF', s=5)
        ax.scatter(x='PRODUCTION_DAY_GAS_COUNTER', y='WATER_BBL', data=df.reset_index(), color='blue', label='Water', s=5)
        ax.set_ylabel('Rate [MCFD or BBLD]')
        ax.set_xlabel('Time [Days]')
        ax.set_title('Semilog Rate Time Plot')
        ax.set(yscale='log')
        ax.set_ylim(1,)
        ax.set_xlim(0,10000)
        ax.grid(b=True, which='major', color='black', linewidth=0.5)
        #ax.grid(b=True, which='minor', color='black', linewidth=0.2)
        ax.legend()
        plt.savefig('figures/semilogratetime.jpg', dpi=300)
        pass

    def RateTimePlot(self, df):
        #plt.clf()
        sns.set(style='white')
        fig = plt.figure(figsize=(10,10))
        ax = fig.add_subplot(1,1,1)
        ax.scatter(x='PRODUCTION_DAY_GAS_COUNTER', y='Q', data=df.reset_index(), color='red', label='Gas Rate',s=5)
        ax.scatter(x='PRODUCTION_DAY_GAS_COUNTER', y='AOF', data=df.reset_index(), color='blue', label='AOF', s=5)
        ax.set_ylabel('Rate [MCFD]')
        ax.set_xlabel('Time [Days]')
        ax.set_title('Rate Time Plot')
        ax.set(xscale='log', yscale='log')
        ax.set_ylim(100,)
        ax.grid(b=True, which='major', color='black', linewidth=0.5)
        ax.grid(b=True, which='minor', color='black', linewidth=0.2)
        ax.legend()
        plt.savefig('figures/ratetime.jpg', dpi=300)
        pass

    def MakeFitDF(self, df, tol):
        TOLERANCE = tol
        fit_df = df[(df['deltamP_q'] != inf) & (df['deltamP_q'] != np.nan) & (df['deltamP_q'] != -inf) &
                    (df['deltamP_q'] >= 0) & (df['deltamP_q'] <= TOLERANCE)] #remove inf, and nan
        
        fit_df = fit_df[fit_df['sqrt_t'] <= 20]  
        fit_df = fit_df[fit_df['sqrt_t'] >= 3]
        
        return fit_df

    def SquareRootTimePlot(self, df):
        #Determine the best fit line
        TOLERANCE = 0.3 * 10**7

        fit_df = self.MakeFitDF(df, TOLERANCE)
        b, m = polyfit(fit_df['sqrt_t'], fit_df['deltamP_q'], 1)

        #plt.clf()
        sns.set(style='white')
        fig = plt.figure(figsize=(10,10))
        ax = fig.add_subplot(1,1,1)
        ax.scatter(x='sqrt_t', y='deltamP_q', data=df.reset_index(), color='blue', label='Rate History',s=5)

        ax.plot(df['sqrt_t'], b + m*df['sqrt_t'], color='black', label='Fit')
        ax.set_ylabel('RPI [(m(Pi)-mP(Wf))/q]')
        ax.set_xlabel('Sqrt(Time) [Days^0.5]')
        ax.set_title('Square Root of Time Plot')
        ax.set_ylim(0,1.0*TOLERANCE)
        ax.set_xlim(0,100)
        ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2e'))
        ax.grid(b=True, which='major', color='black', linewidth=0.5)
        ax.legend()
        plt.savefig('figures/squareroottime.jpg', dpi=300)
        return (b, m)

    def LinearSuperpositionTimePlot(self, df):
        #Determine the best fit line
        TOLERANCE = 0.3 * 10**7

        fit_df = self.MakeFitDF(df, TOLERANCE)
        b, m = polyfit(fit_df['Linear_Superposition_Time'], fit_df['deltamP_q'], 1)

        #plt.clf()
        sns.set(style='white')
        fig = plt.figure(figsize=(10,10))
        ax = fig.add_subplot(1,1,1)
        ax.scatter(x='Linear_Superposition_Time', y='deltamP_q', data=df.reset_index(), color='blue', label='Rate History',s=5)

        ax.plot(df['Linear_Superposition_Time'], b + m*df['Linear_Superposition_Time'], color='black', label='Fit')
        ax.set_ylabel('RPI [(m(Pi)-mP(Wf))/q]')
        ax.set_xlabel('Superposition Time Function [Days^0.5]')
        ax.set_title('Linear Superposition of Time Plot')
        ax.set_ylim(0,0.5*TOLERANCE)
        ax.set_xlim(0,1000)
        ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2e'))
        ax.grid(b=True, which='major', color='black', linewidth=0.5)
        ax.legend()
        plt.savefig('figures/linearsuperpositiontime.jpg', dpi=300)
        return (b, m)

    def PressureDerivative(self, df):
        sns.set(style='white')
        fig = plt.figure(figsize=(10,10))
        ax = fig.add_subplot(1,1,1)
        ax.scatter(x='Linear_Superposition_Time', y='raw_pressure_derivative', data=df.reset_index(), color='blue', label='Raw Derivative',s=5)
        ax.scatter(x='Linear_Superposition_Time', y='Bourdet_Derivative', data=df.reset_index(), color='red', label='Bourdet Derivative',s=5)
        ax.set_ylabel('Inverse Pressure Derivative')
        ax.set_xlabel('Superposition Time Function [Days^0.5]')
        ax.set_title('Wattenbarger Pressure Derivative Plot')
        ax.set_ylim(1*10**-7,)
        ax.set_xlim(1,1000)
        ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2e'))
        ax.grid(b=True, which='major', color='black', linewidth=0.5)
        ax.grid(b=True, which='minor', color='black', linewidth=0.2)
        ax.set(xscale='log', yscale='log')
        ax.legend()
        plt.savefig('figures/pressurederivative.jpg', dpi=300)
        pass

    def LiquidLoadingPlot(self, df):
        #plt.clf()
        sns.set(style='white')
        fig = plt.figure(figsize=(10,10))
        ax = fig.add_subplot(1,1,1)
        ax2 = ax.twinx()
        ax.scatter(x='PRODUCTION_DAY_GAS_COUNTER', y='Q', data=df.reset_index(), color='red', label='Gas Rate',s=5)
        ax.scatter(x='PRODUCTION_DAY_GAS_COUNTER', y='WATER_BBL', data=df.reset_index(), color='blue', label='Water', s=5)
        #ax.plot(df['PRODUCTION_DAY_GAS_COUNTER'], df['Turner_Curve'], color='black', label='Turner Curve')# color='black', label='Turner Curve', s=5)
        ax.scatter(x='PRODUCTION_DAY_GAS_COUNTER', y='Turner_Curve', data=df.reset_index(), color='black', label='Turner Curve', s=5)
        #sns.lineplot(x='PRODUCTION_DAY_GAS_COUNTER', y='Turner_Curve', data=df.reset_index())#, color='black', label='Turner Curve', s=5)
        ax2.scatter(x='PRODUCTION_DAY_GAS_COUNTER', y='TUBING_PRESSURE_AVG', data=df.reset_index(), color='green', label='Tubing Pressure', s=5)
        ax.set_ylabel('Rate [MCFD or BWPD]')
        ax2.set_ylabel('Pressure [psig]')
        ax.set_xlabel('Time [Days^0.5]')
        ax.set_title('Liquid Loading Plot')
        #ax.set_ylim(0,)
        #ax2.set_ylim(0,)
        ax.grid(b=True, which='major', color='black', linewidth=0.5)
        fig.legend(loc='upper right', bbox_to_anchor=(0.85,0.8))
        plt.savefig('figures/liquidloading.jpg', dpi=300)
        pass

    def ProductionPlot(self, df):
        #plt.clf()
        sns.set(style='white')
        fig = plt.figure(figsize=(10,10))
        ax = fig.add_subplot(1,1,1)
        ax2 = ax.twinx()
        ax.scatter(x='PRODUCTION_DAY_GAS_COUNTER', y='Q', data=df.reset_index(), color='red', label='Gas Rate',s=5)
        ax.scatter(x='PRODUCTION_DAY_GAS_COUNTER', y='WATER_BBL', data=df.reset_index(), color='blue', label='Water', s=5)
        ax2.scatter(x='PRODUCTION_DAY_GAS_COUNTER', y='TUBING_PRESSURE_AVG', data=df.reset_index(), color='green', label='Tubing Pressure', s=5)
        ax2.scatter(x='PRODUCTION_DAY_GAS_COUNTER', y='CASING_PRESSURE_AVG', data=df.reset_index(), color='purple', label='Casing Pressure', s=5)
        ax.set_ylabel('Rate [MCFD or BWPD]')
        ax2.set_ylabel('Pressure [psig]')
        ax.set_xlabel('Time [Days]')
        ax.set_title('Production History Plot')
        ax.set_ylim(1,)
        ax2.set_ylim(1,)
        ax.grid(b=True, which='major', color='black', linewidth=0.5)
        ax.set(yscale='log')
        fig.legend(loc='upper right', bbox_to_anchor=(0.85,0.8))
        plt.savefig('figures/productionhistory.jpg', dpi=300)
        pass 