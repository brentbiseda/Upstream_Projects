
# coding: utf-8

# In[33]:


#Brent Biseda
#Project #1
#W200, Monday, 4 PM California Time (7 PM Eastern)
#832-766-5276
#
#This project performs economic evaluation of oil and gas properties from a set of input csv files

import Costs

import numpy as np
import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from textwrap import wrap
import random
from scipy import optimize
import csv


# In[34]:


#Custom error classes
class CalculationError(Exception):
    pass
class DataLoaderError(Exception):
    pass
class AssignmentError(Exception):
    pass
class MontecarloError(Exception):
    pass


# In[35]:


class DataLoader(object):
    """
    This class loads external files and provides the interface between the external and internal
    parts of the program.  This class is able to load in the following information:
    Drill Schedule
    Operating Cost File (OPEX)
    Price File
    Cost Information
    Type Curves
    """
    def __init__(self):
        self.drill_schedule_well_count = 0
        self.drill_schedule = []
        self.type_wells = []
    
    def load_drill_schedule(self, file_name):
        try:
            with open(file_name, 'rt') as fin:
                drill_schedule = csv.DictReader(fin)
                self.drill_schedule = [row for row in drill_schedule]
        except IOError:
            print("Error reading Drill Schedule: " + file_name)
            
        self.drill_schedule_well_count = len(self.drill_schedule)
        
        #Convert text dates to datetime
        date_columns = ["Start_Drill_Date", "Start_Frac_Date", "Start_Production_Date"]
        for column in date_columns:
            for item in self.drill_schedule:
                item[column] = pd.to_datetime(item[column])
        
        #Convert numeric columns to ints
        numeric_columns = ["Comp_Lateral_Length","Type_Curve","Well_ID"]
        for column in numeric_columns:
            for item in self.drill_schedule:
                item[column] = int(item[column])

    def load_type_curve(self, file_name, lateral_length=5000):
        try:
            current_type_well = Type_Well(pd.read_csv(file_name, sep=',',header=0), lateral_length)
            self.type_wells.append(current_type_well)
        except IOError:
            print("Error reading Type Well File: " + file_name)
    
    def load_opex(self, file_name):
        try:
            with open(file_name, 'rt') as fin:
                opex = csv.DictReader(fin)
                self.opex = [row for row in opex]
        except IOError:
            print("Error reading Opex File: " + file_name)
    
    def load_prices(self, file_name):
        try:
            self.prices = pd.read_csv(file_name, sep=',',header=0)
            self.prices['date'] = self.prices['date'].apply(pd.to_datetime)
        except IOError:
            print("Error reading Prices File: " + file_name)
        
    def load_costs(self, file_name):
        try:
            self.costs = file_name
        except IOError:
            print("Error reading Costs function: " + file_name)
    
    def get_opex(self):
        return self.opex
    
    def get_drill_schedule(self):
        return self.drill_schedule
    
    def get_drill_schedule_well_count(self):
        return self.drill_schedule_well_count
    
    def get_type_curves(self):
        return self.type_wells
    
    def get_prices(self):
        return self.prices
    
    def get_costs(self):
        return self.costs


# In[36]:


class Type_Well(object):
    """
    This class represents type wells.  Type wells are hypothetical oil and gas wells with consistent production and
    lateral lengths.  New wells that are created are assigned a type well and then scaled based off of their independent
    length.
    """
    type_well_count = 0
    well_duration = 600 #600 month economic life (50 years)
    
    def __init__(self, production, lateral_length=5000, tc_factor=1.0):
        #First type well code starts at 0
        self.__type_well_code = Type_Well.type_well_count
        self.gas = production['Gas']
        self.oil = production['Oil']
        self.water = production['Water']
        self.lateral_length = lateral_length
        Type_Well.type_well_count += 1
        
    def get_type_well_code(self):
        return self.__type_well_code
    
    def get_well_count(self):
        return Well.well_count


# In[37]:


class Well(Type_Well):
    """
    This class represents an oil and gas well.  The well holds identifying information at the time of its creation.
    Useful Methods:
    assign_type_curve: The well can be assigned a type curve, which provides production information.
    assign_opex: The well can be assigned operating cost information
    assign_capital: The well can be assigned capital cost information for each phase of its construction
    assign_prices: The well can be assigned realized pricing information for its production
    calc_all: Calculates the operating costs, the capital, revenue, and cashflow
    """
    def __init__(self, well_dict):
        #Stores the values in a dictionary which allows for additional information to be input from the csv file
        self.well_dict = well_dict
    
    def __str__(self):
        #If the well has been calculated return the well name and the EUR
        if hasattr(self, 'df'):
            return "Well name: " + self.well_dict['Well_Name'] + " Lat. Len. " + str(self.well_dict['Comp_Lateral_Length']) +         " Gas EUR: " + "{0:.2f}".format(np.sum(self.df['gas'])) + " Oil EUR: " + "{0:.2f}".format(np.sum(self.df['oil']))
        else:
            return "Well name: " + self.well_dict['Well_Name'] + "Lat. Len. " + str(self.well_dict['Comp_Lateral_Length'])
    
    def __repr__(self):
        return self.__str__()

    def assign_type_curve(self, type_curve):
        self.tc_factor = self.well_dict['Comp_Lateral_Length'] / type_curve.lateral_length
        gas = pd.DataFrame(type_curve.gas * self.tc_factor)
        oil = pd.DataFrame(type_curve.oil * self.tc_factor)
        water = pd.DataFrame(type_curve.water * self.tc_factor)
        datelist = pd.DataFrame(pd.date_range(self.well_dict['Start_Production_Date'], periods = Well.well_duration, freq='M').tolist())

        self.df = pd.concat([datelist, gas, oil, water], ignore_index=True, axis=1)
        self.df.columns = ['date', 'gas', 'oil', 'water']
        self.df = self.df[pd.notnull(self.df['date'])]
    
    def assign_opex(self, opex_dict):
        #Stores the values in a dictionary which allows for additional information to be input from the csv file
        self.opex_dict = opex_dict
    
    def assign_capital(self, well_costs):
        if self.well_dict["Formation"] == "Marcellus":
            self.cost_drill = well_costs.marcellus_total_drill(self.well_dict["Comp_Lateral_Length"])
            self.cost_complete = well_costs.marcellus_total_completion(self.well_dict["Comp_Lateral_Length"])
            self.cost_equipment = well_costs.prod_equipment(water_injection = 0)
            self.abandonment = well_costs.abandonment()
        elif self.well_dict["Formation"] == "Utica":
            self.cost_drill = well_costs.utica_total_drill(self.well_dict["Comp_Lateral_Length"])
            self.cost_complete = well_costs.utica_total_completion(self.well_dict["Comp_Lateral_Length"])
            self.cost_equipment = well_costs.prod_equipment(water_injection = 1)
            self.abandonment = well_costs.abandonment()
    
    def assign_prices(self, prices):
        self.df = pd.merge(self.df, prices, how='inner', on='date')
        self.df.fillna(0, inplace=True)
        
    def calc_opex(self):
        self.df['opex_fixed'] = self.opex_dict['wi'] * self.opex_dict['opex_fixed']
        self.df['opex_var'] = self.opex_dict['wi'] * (self.opex_dict['opex_gas'] * self.df['gas'] +                                                       self.opex_dict['opex_wtr'] * self.df['water'] +                                          self.opex_dict['opex_oil'] * self.df['oil'] + self.opex_dict['transport_gas'])
        self.df['opex_total'] = self.df['opex_fixed'] + self.df['opex_var']
        
    def calc_revenue(self):
        self.df['revenue_gas'] = self.opex_dict['nri'] * self.df['price_gas'] * self.opex_dict['btu'] *                                 self.opex_dict['shrink'] * self.df['gas']
        self.df['revenue_oil'] = self.opex_dict['nri'] * self.df['price_oil'] * self.df['oil']
        self.df['revenue_total'] = self.df['revenue_gas'] + self.df['revenue_oil']
    
    def calc_capital(self):
        #Create the list of end of month dates for capital
        capital_dates = pd.DataFrame(pd.date_range(self.well_dict["Start_Drill_Date"], periods = Well.well_duration, freq='M').tolist())
        capital_df = pd.concat([capital_dates, pd.DataFrame(np.zeros(len(capital_dates)))], ignore_index=True, axis=1)
        capital_df.columns = ['date', 'capital']
        
        #Set the capital in the correct places
        capital_df.loc[(capital_df['date'] == self.well_dict["Start_Drill_Date"]), 'capital'] = self.cost_drill
        capital_df.loc[(capital_df['date'] == self.well_dict["Start_Frac_Date"]), 'capital'] = self.cost_complete
        capital_df.loc[(capital_df['date'] == self.well_dict["Start_Production_Date"]), 'capital'] = self.cost_equipment
        

        #Combine the capital dataframe with the regular dataframe
        if 'capital' in self.df.columns:
            self.df = self.df.drop('capital',1)
            
        self.df = pd.merge(self.df, capital_df, how='outer', on='date')
        self.df.fillna(0, inplace=True)
        self.df = self.df.sort_values(by='date')
        self.df = self.df.reset_index(drop=True)
        self.df.loc[len(self.df['date'])-1,'capital'] = self.abandonment     #abandonment capital is set at end of life
                
    def calc_cashflow(self):
        #Operating cashflow is not negative.  Well is not operated if negative cashflow
        self.df['cashflow_operating'] = self.df['revenue_total'] - self.df['opex_total']
        self.df.loc[self.df['cashflow_operating'] < 0, 'cashflow_operating'] = 0
        self.df['cashflow_total'] = self.df['cashflow_operating'] - self.df['capital']
    
    def calc_all(self):
        #Recalculate all columns
        try:
            self.calc_opex()
            self.calc_revenue()
            self.calc_capital()
            self.calc_cashflow()
        except CalculationError:
            print("Calculation Error in well level math")


# In[38]:


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
        columns = ['well_id', 'well_name', 'npv', 'irr', 'moic', 'f_d', 'eur', 'capital', 'cum_cashflow', 'total_opex']
        self.single_well_df = pd.DataFrame(columns=columns)
        project_columns = ['project', 'npv', 'irr', 'moic', 'f_d', 'eur', 'capital', 'cum_cashflow', 'total_opex']
        self.summary_df = pd.DataFrame(columns=project_columns)
        
    def generate_summary_economics(self, field, rate_of_return=0.1):
        self.df = pd.DataFrame()
        #Create a summary dataframe from all wells
        for well in field.wells:
            self.df = self.df.append(well.df)  
        
        self.df = self.df.groupby(['date']).sum()
        self.df = self.df.reset_index()
        
        econ_dict = self.generate_project_dict(self.df, rate_of_return)
        econ_df = pd.DataFrame.from_dict(econ_dict)
        self.summary_df = self.summary_df.append(econ_df, ignore_index=True)
        
    def generate_project_dict(self, df, rate_of_return):
        capital = np.sum(df['capital'])
        cum_cashflow = np.sum(df['cashflow_total'])
        eur = np.sum(df['gas'] + df['oil'] * 6)

        npv = self.xnpv(df, rate_of_return)
        irr = self.xirr(df, rate_of_return)
        moic = cum_cashflow / capital
        f_d = capital / eur
        total_opex = np.sum(df['opex_total'])

        econ_dict = {
            'project': "Total Project",
            'npv': [npv],
            'irr': [irr],
            'moic': [moic],
            'f_d': [f_d],
            'eur': [eur],
            'capital': [capital],
            'cum_cashflow': [cum_cashflow],
            'total_opex': [total_opex]
        }
        return econ_dict
        
    def generate_well_dict(self, well, rate_of_return = 0.1):
        capital = np.sum(well.df['capital'])
        cum_cashflow = np.sum(well.df['cashflow_total'])
        eur = np.sum(well.df['gas'] + well.df['oil'] * 6)

        npv = self.xnpv(well.df, rate_of_return)
        irr = self.xirr(well.df, rate_of_return)
        moic = cum_cashflow / capital
        f_d = capital / eur
        total_opex = np.sum(well.df['opex_total'])

        econ_dict = {
            'well_id': [well.well_dict["Well_ID"]],
            'well_name': [well.well_dict["Well_Name"]],
            'npv': [npv],
            'irr': [irr],
            'moic': [moic],
            'f_d': [f_d],
            'eur': [eur],
            'capital': [capital],
            'cum_cashflow': [cum_cashflow],
            'total_opex': [total_opex]
        }    
        return econ_dict
    
    def generate_field_economics(self, field, rate_of_return = 0.1):
        for well in field.wells:
            self.generate_well_economics(well, rate_of_return)
    
    def generate_well_economics(self, well, rate_of_return=0.1):
        econ_dict = self.generate_well_dict(well, rate_of_return)
        econ_df = pd.DataFrame.from_dict(econ_dict)
        self.single_well_df = self.single_well_df.append(econ_df, ignore_index=True)
            
    def xnpv(self, dataframe, rate_of_return = 0.1):
        #Calculate NPV for a particular well or project from a dataframe
        dates = dataframe['date']
        cashflow_total = dataframe['cashflow_total']
        cashflows = list(zip(dates,cashflow_total))
        t0 = dates[0]
        
        calculated_npv = np.sum([cf/(1+rate_of_return)**((t-t0).days/365.0) for (t,cf) in cashflows])
        return calculated_npv
    
    def xirr(self, dataframe, rate_of_return = 0.1, guess = 0.1):
        #Calculate IRR for a particular well or project from a dataframe
        return optimize.newton(lambda r: self.xnpv(dataframe, r) , guess)


# In[39]:


class Field(object):
    """
    This class represents a field of wells. A field starts with 0 wells.
    Useful Methods:
    drill_wells: Requires a dataloader which contains a drill schedule, type curves, operating costs, prices, and costs
                Then creates a list of wells that are held within the field and lastly calculates their cashflows
    """
    def __init__(self):
        self.well_count = 0
        self.drill_schedule = []
        self.wells = []
    
    def drill_wells(self, dataloader):
        try:
            self.well_count = dataloader.get_drill_schedule_well_count()
            self.drill_schedule = dataloader.get_drill_schedule()
            self.type_curves = dataloader.get_type_curves()
            self.opex = dataloader.get_opex()
            self.prices = dataloader.get_prices()
            self.costs = dataloader.get_costs()
        except DataLoaderError:
            print("Error in incomplete definition of Data Loader to populate wells")
        
        date_columns = ["Start_Drill_Date", "Start_Frac_Date", "Start_Production_Date"]
        
        for well_dict in self.drill_schedule:
            #Set all dates to consistent end of month format
            for column in date_columns:
                well_dict[column] = well_dict[column].to_period('M').to_timestamp('M')
              
            #Create Wells and append to list of wells
            current_well = Well(well_dict)
            
            #Access the type curve to assign to the new well  
            try:
                current_well.assign_type_curve(self.type_curves[well_dict['Type_Curve']])
            except AssignmentError:
                print("Error in assignment of type curve to current well")
            
            #Access opex and assign to the well
            opex_dict = [item for item in self.opex if item['well_id'] == str(well_dict["Well_ID"])][0]
            #Convert opex_dict from strings to floats
            for k, v in opex_dict.items():
                opex_dict[k] = float(v)
            
            try:
                current_well.assign_opex(opex_dict)
                current_well.assign_prices(self.prices)
                current_well.assign_capital(dataloader.costs)
            except AssignmentError:
                print("Error in assignment of opex, prices, or capital to current well")
                
            current_well.calc_all()
                
            
            #Put well into list of wells
            self.wells.append(current_well)


# In[40]:


class Visualizer(object):
    """
    This class represents a visualization tool for wells & economics
    Useful Methods:
    plot_production - Creates a linear scale plot of production from a dataframe & associated production streams
    log_plot_production - Creates a log scale plot of production from a dataframe & associated production streams
    generate_well_report - Creates a pdf of a single property
    generate_field_report - Creates a pdf of the total field
    """
    def plot_prod(self, df):
        #Default well plotting method
        date = df['date']
        gas = df['gas']
        oil = df['oil']
        water = df['water']
        plt.plot(date, gas, label='Gas Rate', c='red')
        plt.plot(date, oil, label='Oil Rate', c='green')
        plt.plot(date, water, label='Water Rate', c='blue')
        plt.xlabel("Date")
        plt.ylabel("Volume")
        plt.legend(loc='upper right')
        
    def log_plot_production(self, df):
        plt.clf()
        plt.yscale('log') #Plot log scale on y-axis
        self.plot_prod(df)
        plt.show()
    
    def plot_production(self, df):
        plt.clf()
        plt.yscale('linear') #plot linear scale on y-axis
        self.plot_prod(df)
        plt.show()
    
    def plot_economics(self, df, variable_name, num_bins = 10):
        plt.clf()
        data = df[variable_name]
        binwidth = (data.max() - data.min()) / num_bins
        if binwidth == 0: #Prevent error from having same output for all cases
            binwidth = 1
        plt.hist(data, bins=np.arange(min(data)-0.01, max(data) + binwidth, binwidth), label = variable_name)
        plt.xlabel("Value")
        plt.legend(loc='upper right')
    
    def generate_well_report(self, well, file_name='output.pdf'):
        plt.clf()
        self.plot_prod(well.df)
        plt.title("\n".join(wrap(str(well), 60)))
        plt.tight_layout()
        with PdfPages(file_name) as pp:
            pp.savefig()
    
    def generate_field_report_graphs(self, field, file_name='output.pdf'):
        with PdfPages(file_name) as pp:
            for well in field.wells:
                plt.clf()
                self.plot_prod(well.df)
                plt.title("\n".join(wrap(str(well), 60)))
                plt.tight_layout()
                pp.savefig()
    
    def generate_econ_report(self, df, file_name='output.pdf'):
        with PdfPages(file_name) as pp:
            for column in df.columns: 
                if type(df[column][0]) != type("string"): #exclude non-numeric columns
                    plt.clf()
                    self.plot_economics(df, column)
                    plt.title("\n".join(wrap(str(column), 60)))
                    plt.tight_layout()
                    pp.savefig()
           
    def generate_field_table(self, df, file_name='output.html'):
        df.to_html(file_name)
    
    def plot_montecarlo(self, df_list, var_name):
        #Accepts a list of dataframes and plots variable according to the quantized breaks
        plt.clf()
        quantize_number = len(df_list)
        
        for i in range(len(df_list)):
            df = df_list[i]
            date = df['date']
            var_values = df[var_name]
            plt.plot(date, var_values, label='{0:.0%}'.format(i/quantize_number))
        plt.legend(loc='upper right')
    
    def generate_montecarlo_report(self, df_list, file_name='output.pdf', var_name="gas"):
        self.plot_montecarlo(df_list, var_name = "gas")
        plt.tight_layout()
        with PdfPages(file_name) as pp:
            pp.savefig()
    
    def generate_montecarlo_csv(self, df_list, file_name='output.csv', var_name="gas"):
        #Create a csv of the quantized file
        self.montecarlo_csv_df=pd.DataFrame()
        quantize_number = len(df_list)
        
        for i in range(len(df_list)):
            df = df_list[i]
            label = '{0:.0%}'.format(i/quantize_number)
            if i == 0:
                self.montecarlo_csv_df['date'] = df['date']
            self.montecarlo_csv_df[label] = df[var_name]
        self.montecarlo_csv_df.to_csv(file_name, sep=',', encoding='utf-8')
    


# In[41]:


class Project(object):
    """
    This class is designed to fully contain the functionality of the project and minimize any human interaction
    Useful methods:
    run: creates a field and drills the wells with the information populated from the data loader
    gen_econ: Creates summary and field level economics
    gen_vis: Creates graphs and tables for field level and project level economics
    montecarlo: Implementation of montecarlo simulation.  It has been hard coded to vary utica and marcellus type curve gas rates
    run_montecarlo: Wrapper function for the montecarlo simulation.  Creates graphs and reports from the output of the montecarlo
    """
    def __init__(self):
        self.dl = DataLoader()
        self.dl.load_drill_schedule('input/drill_schedule.csv')
        self.costs = Costs.Costs()

        self.dl.load_type_curve('input/utica_type_curve.csv', 7500)
        self.dl.load_type_curve('input/marcellus_type_curve.csv', 7000)
        self.dl.load_opex('input/opex.csv')
        self.dl.load_prices('input/price.csv')
        self.dl.load_costs(self.costs)
        
        self.econ = Economics()
        self.vis = Visualizer()
        
    def run(self):
        self.src_field = Field()
        self.src_field.drill_wells(self.dl)
        self.gen_econ()
        self.gen_vis()
        
        #Show some nice outputs for the user when complete
        self.vis.plot_production(self.econ.df)
        print(self.econ.summary_df)
        
    def run_montecarlo(self, iterations = 25, quantize_breaks = 10):
        try:
            self.montecarlo(iterations)
            self.quantize(quantize_breaks)
            self.vis.generate_montecarlo_csv(self.montecarlo_quantize_df_list, file_name = 'output/montecarlo_output.csv')
            self.vis.generate_montecarlo_report(self.montecarlo_quantize_df_list, file_name = 'output/montecarlo_report.pdf')
            self.vis.generate_econ_report(self.econ.summary_df, file_name = 'output/montecarlo_economics.pdf')
        except MontecarloError:
            print("Error Encountered while performing montecarlo analysis")
            
    def gen_econ(self):
        #Calculate Economics
        self.econ.generate_summary_economics(self.src_field)
        self.econ.generate_field_economics(self.src_field)
    
    def gen_vis(self):
        #Visualizations to pdf
        self.vis.generate_econ_report(self.econ.single_well_df, file_name = 'output/summary_economics.pdf')
        self.vis.generate_field_report_graphs(self.src_field, 'output/single_well_graphs.pdf')
        self.vis.generate_field_table(self.econ.summary_df, file_name = 'output/summary_table.html')
        self.vis.generate_field_table(self.econ.single_well_df, file_name = 'output/single_well_table.html')
    
    def montecarlo(self, iterations = 10):
        #This iterates on the gas type curve for each well and scales the result from a log-normal distribution
        #based on historical performance information
        self.montecarlo_df_list = []
        
        #Parameters established by analysis of existing wells
        marcellus_mean = 1149
        marcellus_mu = 7.047
        marcellus_sigma = 0.178
        
        utica_mean = 1672
        utica_mu = 7.422
        utica_sigma = 0.180
        
        sample_length = self.src_field.well_count
        variable = 'gas'
        
        for iteration in range(iterations):
            if iteration % 10 == 0:
                print("Iteration " + str(iteration) + " Current Time " + str(dt.datetime.now()))
            scale_factors = []
            
            #Perform montecarlo on each well
            for i in range(sample_length):
                if self.src_field.wells[i].well_dict['Formation'] == "Marcellus":
                    scale_factor = np.random.lognormal(marcellus_mu, marcellus_sigma) / marcellus_mean #Distribution to sample with
                else:
                    scale_factor = np.random.lognormal(utica_mu, utica_sigma) / utica_mean #Distribution to sample with
                
                scale_factors.append(scale_factor)
                self.src_field.wells[i].df[variable] = self.src_field.wells[i].df[variable] * scale_factor #Change values based off montecarlo distribution
                self.src_field.wells[i].calc_all()
            
            self.econ.generate_summary_economics(self.src_field)
            self.montecarlo_df_list.append(self.econ.df)
            
            #Reset the wells to their original state
            for i in range(sample_length):
                self.src_field.wells[i].df[variable] = project.src_field.wells[i].df[variable] / scale_factors[i] #Revert back to original value
                
        #Sort the montecarlo results by production
        self.montecarlo_df_list = sorted(self.montecarlo_df_list, key=lambda x: np.sum(x[variable]))
    
    def quantize(self, quantize_breaks = 10):
        #Find the quantile data for the various scenarios
        #Generally speaking, we are looking for 10%, 20%, 30%, 40%, 50%, 60%, 70%, 80%, 90% likelihood scenarios
        iterations = len(self.montecarlo_df_list)
        self.montecarlo_quantize_df_list = []
        for i in range(1, iterations, iterations // quantize_breaks):
            self.montecarlo_quantize_df_list.append(self.montecarlo_df_list[i])


# In[42]:


if __name__ == "__main__":
    project = Project()
    project.run()
    project.run_montecarlo()

