import pandas as pd
import numpy as np
from PVT import *
from numpy import inf
from numpy.polynomial.polynomial import polyfit

class Well:
    def __init__(self, well_id, production, wellHeader, wellInputs, completion, geology):
        self.df = production[production['FILENUM']==str(well_id)]
        self.wellHeader = wellHeader[wellHeader['WELLID']==str(well_id)].reset_index()
        self.wellInputs = wellInputs[wellInputs['WELLID']==str(well_id)].reset_index()
        self.completion = completion[completion['WELLID']==str(well_id)].reset_index()
        self.geology = geology[geology['WELLID']==str(well_id)].reset_index()
        self.wellDict = self.GetWellParameters()
        self.wellDict = self.CalcWellParameters(self.wellDict)
        self.df = self.PopulateWellDataSet(self.df, self.wellDict)
        
        #Create Parameters from square root of time graph
        self.MakeFitDF()
        self.CalcFracArea()
    
    def GetWellParameters(self):
        return {'depth': self.wellInputs['TVD_AVG_RESERVOIR'][0],
                'wellid': self.wellInputs['WELLID'][0],
                'presGrad': self.wellHeader['GRADIENT'][0],
                'netPayHeight': 100,
                'porosity': self.wellHeader.POROSITY[0] / 100,
                'play': self.wellHeader.PLAY[0],
                'Sw': self.wellHeader.INITIAL_WATER_SATURATION[0] / 100,
                'BHT': self.wellHeader.FORMATION_TEMPERATURE[0],
                'gasGrav': self.wellHeader.GAS_GRAVITY[0],
                'CO2': self.wellHeader.CO2[0],
                'N2': self.wellHeader.N2[0],
                'H2S': self.wellHeader.H2S[0],
                'formationCompressibility': 1.25 * 10 ** (-5),
                'surfaceTemp': 60,
                'lateralLength': self.wellHeader.TREATABLE_LENGTH[0],
                'fracStages': self.completion.STAGE[0],
                'perfClusters': self.completion.PERF_CLUSTERS_CNT[0],
                'fluidTotal': self.completion.FLUID_TOT[0],
                'sandTotal': self.completion.TOTAL_SAND[0],
                'RF': 0.5,
                'casingPressure': 400}
    
    def CalcWellParameters(self, wellDict):
        wellDict['Pi'] = wellDict['depth'] * wellDict['presGrad']
        wellDict['Cgi'] = Cg(wellDict['Pi'], wellDict['BHT'], wellDict['gasGrav'], wellDict['N2'], wellDict['CO2'], wellDict['H2S'])
        wellDict['Cti'] = (wellDict['Cgi'] * (1-wellDict['Sw'])) + wellDict['formationCompressibility']
        wellDict['Ugi'] = Ug(wellDict['Pi'], wellDict['BHT'], wellDict['gasGrav'], wellDict['N2'], wellDict['CO2'], wellDict['H2S'])
        wellDict['Pwf_Ps'] = math.exp(wellDict['Pi']/(53.34*(wellDict['BHT']+460+540)/2))
        wellDict['Zs'] = Z(wellDict['casingPressure'], wellDict['BHT'], wellDict['gasGrav'], wellDict['N2'], wellDict['CO2'], wellDict['H2S'])
        wellDict['Zwfe'] = Z(wellDict['casingPressure']*wellDict['Pwf_Ps'], wellDict['BHT'], wellDict['gasGrav'], wellDict['N2'], wellDict['CO2'], wellDict['H2S'])
        wellDict['Pwf'] = math.exp(wellDict['Pi'] / (53.34*(wellDict['BHT']+460+540)/2*(wellDict['Zs']+wellDict['Zwfe'])/2)) * wellDict['casingPressure']
        wellDict['mpi'] = mp(wellDict['Pi'], wellDict['BHT'], wellDict['gasGrav'], wellDict['N2'], wellDict['CO2'], wellDict['H2S'])
        wellDict['mpwf'] = mp(wellDict['Pwf'], wellDict['BHT'], wellDict['gasGrav'], wellDict['N2'], wellDict['CO2'], wellDict['H2S'])
        wellDict['Dd'] = (wellDict['mpi']-wellDict['mpwf'])/wellDict['mpi']
        wellDict['Fcp'] = 1-(0.0852*wellDict['Dd'])-(0.0857*wellDict['Dd']*wellDict['Dd'])
        wellDict['tbg_ID'] = 1.995
        wellDict['tbg_OD'] = 2.375
        wellDict['csg_ID'] = 4.995
        wellDict['csg_OD'] = 5.500
        wellDict['API'] = 55
        wellDict['surface_tension'] = 55
        wellDict['permeability'] = 1*10**(-4)

        return wellDict
    
    def MakeFitDF(self):
        TOLERANCE = 0.3 * 10**7
        
        df = self.df
        
        fit_df = df[(df['deltamP_q'] != inf) & (df['deltamP_q'] != np.nan) & (df['deltamP_q'] != -inf) &
                    (df['deltamP_q'] >= 0) & (df['deltamP_q'] <= TOLERANCE)] #remove inf, and nan
        
        fit_df = fit_df[fit_df['sqrt_t'] <= 20]  
        fit_df = fit_df[fit_df['sqrt_t'] >= 3]
        
        b, m = polyfit(fit_df['sqrt_t'], fit_df['deltamP_q'], 1)
        
        self.wellDict['mCPL'] = m
        self.wellDict['bCPL'] = b
        
        return (b, m)

    
    def CalcFracArea(self):       
        self.wellDict['frac_area'] = (self.wellDict['Fcp']*1262*(self.wellDict['BHT']+460)) / ((self.wellDict['porosity'] * self.wellDict['Ugi'] * self.wellDict['Cti'] * self.wellDict['permeability'])**0.5) / self.wellDict['mCPL']
    
    def PopulateWellDataSet(self, df, wellDict):
        df = df.reset_index(drop=True)
        df['Q'] = df.GAS_MCF
        df['SURF_PRESSURE'] = df.CASING_PRESSURE_AVG
        df['Pwf'] = df['SURF_PRESSURE']*wellDict['Pwf']/wellDict['casingPressure']
        df['mPwf'] = df['Pwf'].apply(lambda x: mp(abs(x)+0.00001, wellDict['BHT'], wellDict['gasGrav'], wellDict['N2'], wellDict['CO2'], wellDict['H2S']))
        df['deltamP_q'] = (wellDict['mpi']-df['mPwf'])/df['Q']
        df['sqrt_t'] = df['PRODUCTION_DAY_GAS_COUNTER'].apply(math.sqrt)
        df['AOF'] = df['Q'] * wellDict['mpi'] / (wellDict['mpi'] - df['mPwf'])
        df['deltamAOF_q'] = (wellDict['mpi'] - wellDict['mpwf']) / df['AOF']
        df['ln_t'] = df['PRODUCTION_DAY_GAS_COUNTER'].apply(math.log)
        df['raw_pressure_derivative'] = abs((df['ln_t']-df['ln_t'].shift(1)) / (df['deltamP_q']-df['deltamP_q'].shift(1)))
        df['P_Z'] = df['CASING_PRESSURE_AVG']/df['Pwf'].apply(lambda x: Z(abs(x)+0.00001, wellDict['BHT'], wellDict['gasGrav'], wellDict['N2'], wellDict['CO2'], wellDict['H2S']))
        df['Linear_Superposition_Time'] = self.Superposition(df)
        df['Linear_Superposition_Time_Squared'] = df['Linear_Superposition_Time'] ** 2
        df['Bourdet_Derivative'] = self.BourdetDerivative(df, L = 0.33)
        df['Turner_Curve'] = self.TurnerCurve(df, wellDict)

        return df
    
    def Superposition(self, df):
        """Calculated linear superposition time function"""
        q_init = max(df['GAS_MCF']) 
        #Count the number of non-zero gas rates
        non_zero_rows = sum(df['GAS_MCF']>0)
        superposition = np.zeros(shape=(df.shape[0],1))

        #Iterate through each row of the df
        for i in range(df.shape[0]):
            q_n = df['GAS_MCF'][i]
            t_n = i
            lintime = (t_n ** 0.5) * q_init / q_n
            if(q_n > 0 ):
                for j in range(i):
                    q_j1 = df['GAS_MCF'][j]
                    q_j = df['GAS_MCF'][j+1]
                    t_j1 = j
                    lintime = lintime + (q_j - q_j1) / q_n * (t_n - t_j1) ** 0.5
                superposition[i] = lintime
        return superposition

    def BourdetDerivative(self, df, L = 0.33):
        """Smooth the derivative data by making use of bourdet algorithm.
        Central derivative smoothing technique
        dP/dXi=((dP1/dX1)*dX2+(dP2/dX2)*dX1)/(dX1+dX2)
        dX12>L

        Calculate Bourdet Derivative function"""

        bourdetDerivative = np.zeros(shape=(df.shape[0],1))

        #Iterate through each row of the df
        #Ignore first row and last row
        totalRows = df.shape[0] - 1

        for i in range(1, totalRows):
            prow = i
            nrow = i
            Qc = df['deltamP_q'][i]
            Tc = df['ln_t'][i]
            pQ = Qc
            nQ = Qc
            pT = Tc
            nT = Tc

            flag = 0

            while (abs(pT - nT) < L) & (flag != -1):
                #Expand the range for the derivative
                prow = i - 1
                nrow = i + 1

                #Prevent the previous row from looking outside of the range
                if prow < 0:
                    prow = 0

                #Prevent the next row from looking outside of the range
                if nrow > totalRows:
                    nrow = totalRows

                pQ = df['deltamP_q'][prow]
                nQ = df['deltamP_q'][nrow]
                pT = df['ln_t'][prow]
                nT = df['ln_t'][nrow]

                flag = flag + 1
                #Exit if we look at too many data points
                if flag > 30:
                    flag = -1

            #Calculations for derivatives
            dp1 = (Qc - pQ)
            dp2 = (nQ - Qc)
            dx1 = (Tc - pT)
            dx2 = (nT - Tc)
            dpdx1 = dp1 / dx1
            dpdx2 = dp2 / dx2

            #Calculate the final pressure derivative
            dpdx = (dpdx1 * dx2 + dpdx2 * dx1) / (dx1 + dx2)

            #Return the inverse of the derivative as per the Wattenbarger Type Curve
            bourdetDerivative[i] = (1 / dpdx)

        return bourdetDerivative

    def TurnerCurve(self, df, wellDict):
        """Calculate the liquid loading curve for the given well"""
        turnerCurve = np.zeros(shape=(df.shape[0],1))

        for i in range(df.shape[0]):
            currentRate = df['GAS_MCF'][i]
            if currentRate > 0:
                area = math.pi * wellDict['tbg_ID'] ** 2 / 4 / 144

                z_factor = Z(df['TUBING_PRESSURE_AVG'][i], wellDict['BHT'], wellDict['gasGrav'], wellDict['N2'], wellDict['CO2'], wellDict['H2S'])

                #Air density is calculated using ideal gas laww (might want to double check this)
                air_density = (df['TUBING_PRESSURE_AVG'][i] + 14.7) / ((wellDict['surfaceTemp'] + 460) * 0.37037)
                dens_gas = air_density * wellDict['gasGrav']
                #Water density is 62.4 lb/cu-ft (assumed incompressible)
                oil_density = 141.5 / (wellDict['API'] + 131.5) * 62.4

                #Calculate the mixed liquid density
                oil_rate = 0
                water_rate = df['WATER_BBL'][i]

                if (oil_rate + water_rate) > 0:
                    dens_liq = (oil_rate * oil_density + water_rate * 62.4) / (oil_rate + water_rate)
                else:
                    dens_liq = 100

                v_t = 1.593 * wellDict['surface_tension'] ** 0.25 * (dens_liq - dens_gas) ** 0.25 / dens_gas ** 0.5

                turner = 3067 * df['TUBING_PRESSURE_AVG'][i] *v_t * area / (wellDict['surfaceTemp'] + 460) / z_factor

                turnerCurve[i] = turner

        return turnerCurve