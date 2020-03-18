import pandas as pd
import numpy as np
from scipy import optimize

class Economics(object):
    """
    This class represents the economic engine.
    Useful Methods:
    xnpv - Calculates the NPV from a dataframe
    xirr - Calculates the rate of return from a dataframe
    generate_summary_economics - Calculates the economics for the field in aggregate
    generate_field_economics - Calculates the economics for each single well in the field
    generate_well_economics - Calculates the economics for a single well in the field
    """
    def __init__(self):
        columns = ['ID', 'NPV', 'IRR', 'MOIC', 'F_D', 'EUR', 'CAPITAL', 'CUM_CASHFLOW', 'OPEX_TOTAL']
        self.single_well_df = pd.DataFrame(columns=columns)
              
    def generate_well_dict(self, well, rate_of_return = 0.1):
        capital = np.sum(well['CAPITAL'])
        cum_cashflow = np.sum(well['TOTAL_CASHFLOW'])
        eur = np.sum(well['GAS'])

        npv = self.xnpv(well, rate_of_return)
        irr = self.xirr(well, rate_of_return)
        moic = cum_cashflow / capital
        f_d = capital / eur
        total_opex = np.sum(well['OPEX_TOTAL'])

        econ_dict = {
            'ID': well['ID'][0],
            'NPV': [npv],
            'IRR': [irr],
            'MOIC': [moic],
            'F_D': [f_d],
            'EUR': [eur],
            'CAPITAL': [capital],
            'CUM_CASHFLOW': [cum_cashflow],
            'OPEX_TOTAL': [total_opex]
        }    
        return econ_dict
    
    def generate_well_economics(self, well, rate_of_return=0.1):
        econ_dict = self.generate_well_dict(well, rate_of_return)
        econ_df = pd.DataFrame.from_dict(econ_dict)
        self.single_well_df = self.single_well_df.append(econ_df, ignore_index=True)
            
    def xnpv(self, dataframe, rate_of_return = 0.1):
        #Calculate NPV for a particular well or project from a dataframe
        dates = dataframe['DATE']
        cashflow_total = dataframe['TOTAL_CASHFLOW']
        cashflows = list(zip(dates,cashflow_total))
        t0 = dates[0]
        
        calculated_npv = np.sum([cf/(1+rate_of_return)**((t-t0).days/365.0) for (t,cf) in cashflows])
        return calculated_npv
    
    def xirr(self, dataframe, rate_of_return = 0.1, guess = 0.1):
        #Calculate IRR for a particular well or project from a dataframe
        return optimize.newton(lambda r: self.xnpv(dataframe, r) , guess, tol=0.01, maxiter=250)