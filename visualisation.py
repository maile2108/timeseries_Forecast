import matplotlib.pyplot as plt # visualization lib

import pandas as pd # data processing lib

class Visualisation():
    def __init__(self, dataFrame):
        self.data_frame = dataFrame
        self.datatimeIdx = dataFrame.index
        self.year_list = self.datatimeIdx.year
        self.columns = self.data_frame.columns
        
        self.month_list = self.datatimeIdx.month.unique()
        self.total_consum_df = self.data_frame['Consumption']
        self.wind_product_df = self.data_frame['Wind']
        self.solar_product_df = self.data_frame['Solar']
        self.ws_product_df = self.data_frame['Wind+Solar']
        
        self.color_list = ['r', 'g', 'b', 'y']
        
    # plot figure from the load data
    def original_plot(self):
        """
        Original data visualisation
        """
        # Plot original data as time series
        
        # Display all data in one figure
        fig, axs = plt.subplots(4, 1) # Sub-figure
        fig.tight_layout() # To make visible 
        for i in range(len(self.columns)):
            # Position of each sub plot with
            axs2 = axs[i]
            # Plot data with setting label and color attributes
            axs2.plot(self.data_frame[self.columns[i]], label = self.columns[i], color = self.color_list[i])
            # Set up Legend
            axs2.legend(loc = 1,  ncol = 1, prop={'size': 6})
          
        # Save PDF file
        plt.savefig('./Data_Image.pdf', bbox_inches='tight') # Save pdf file
        plt.show()
    
    def year_plot(self):
        """
        Display total consumption by years.
        It also contains the Wind, Solar production by years
        """
        
        self.year_uni_list = self.year_list.unique()
        total_year_list = []
        wind_year_list = []
        solar_year_list = []
        
        for i in range(len(self.year_uni_list)):
            year = self.year_uni_list[i]
            total_year_list.append(self.total_consum_df[self.year_list.map(lambda x: x == year)])
            wind_year_list.append(self.wind_product_df[self.year_list.map(lambda x: x == year)])
            solar_year_list.append(self.solar_product_df[self.year_list.map(lambda x: x == year)])

        # Draw total consumption only
        fig, axs = plt.subplots(6, 2) # Divide subplot for each year
        for j in range(len(self.year_uni_list)):
            position = divmod(j, 2) # Position of sub-figure
            axs2 = axs[position]
            
            axs2.plot(total_year_list[j], label = self.year_uni_list[j])
            # Set up Legen and update tick parameter
            axs2.legend(loc = 1,  ncol = 1, prop={'size': 6})
            # Adjust axes parameters to make visible 
            axs2.tick_params(length = 1, labelsize = 4, pad = 1)
        
        # Save PDF file    
        plt.savefig('./Consumption_year.pdf', bbox_inches='tight') # Save pdf file
        plt.show()

    
    def distribution_plot(self):
        """
        Draw distribution of Consumption and Production

        """        
        fig, axs = plt.subplots(2, 2) # Divide sub-plot
        fig.tight_layout() #to make visible 
        for i in range(len(self.columns)):
            position = divmod(i, 2) # Position of sub-figure
            axs2 = axs[position]
            
            data = self.data_frame[self.columns[i]]
            data = data[data.notnull()] # Remove NaN values
            axs2.hist(data, histtype='step', color = self.color_list[i])
            axs2.set_title(self.columns[i])
            
        # Save PDF file
        plt.savefig('./Distribution.pdf', bbox_inches='tight') # Save pdf file
        plt.show()
        
    def data_boxplot(self):
        """
        Display min, max value of Consumption and Production, total values by months
        """
        # Get unique month of data
        month_uni_list = self.month_list.unique()
        
        # Remove nan value
        consumption  = self.total_consum_df[self.total_consum_df.notnull()]
        wind_product = self.wind_product_df[self.wind_product_df.notnull()]
        solar_product = self.solar_product_df[self.solar_product_df.notnull()]
        ws_product = self.ws_product_df[self.ws_product_df.notnull()]

        cons_month = []
        wind_month = []
        solar_month = []
        ws_month = []
        
        # Get values of index has same month
        for i in range(len(month_uni_list)):
            cons_month.append(consumption[consumption.index.month.map(lambda x: x == month_uni_list[i])])
            wind_month.append(wind_product[wind_product.index.month.map(lambda x: x == month_uni_list[i])])
            solar_month.append(solar_product[solar_product.index.month.map(lambda x: x == month_uni_list[i])])
            ws_month.append(ws_product[ws_product.index.month.map(lambda x: x == month_uni_list[i])])

        month_box_data = [cons_month, wind_month, solar_month, ws_month] # Joint data
        fig, axs = plt.subplots(2, 2)
        fig.tight_layout()
        for i in range(len(month_box_data)):
            position = divmod(i, 2) # Position of sub-figure
            axs2 = axs[position]
            axs2.boxplot(month_box_data[i], showfliers=False)
            axs2.set_title(self.columns[i])
        
        # Save PDF file    
        plt.savefig('./Boxplot.pdf', bbox_inches='tight') # Save pdf file
        plt.show()


# Test
if __name__ == '__main__':
    data_frame = pd.read_csv('dataset.csv', index_col = 'Date', parse_dates = True)
    data_frame.info()
    data_visual = Visualisation(data_frame)
    data_visual.data_boxplot()
