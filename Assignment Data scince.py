import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
from lmfit import Model
import matplotlib.font_manager as fm

def read_clean_data(file,columns):
   
    # Read the data from the CSV file
    data = pd.read_csv(file)
    mean_imputer = SimpleImputer(strategy='mean')

    data[columns] = mean_imputer.fit_transform(data[columns])
    return data 


def fit_exponential_growth_model(time_points, actual_values):

    # Define the exponential growth model function
    def exponential_growth_model(x, amplitude, growth_rate):
        return amplitude * np.exp(growth_rate * np.array(x))

    # Create a model and set initial parameters
    model = Model(exponential_growth_model)
    initial_params = model.make_params(amplitude=1, growth_rate=0.001)

    # Fit the model to the data
    fitting_result = model.fit(actual_values, x=time_points, params=initial_params)

    return fitting_result


def plot_curve_fit_with_confidence_interval(time_points, actual_values, fitting_result):

    # Set custom font to a system font
    sns.set(style="whitegrid", font_scale=1.2)

# Create a figure and axis
    fig, ax = plt.subplots(figsize=(12, 8))

# Scatter plot for actual data with a distinctive marker and color
    sns.scatterplot(x=time_points, y=actual_values, label='Actual Data', s=100, marker='o', color='navy', ax=ax)

# Line plot for the exponential growth fit with a unique linestyle and color
    sns.lineplot(x=time_points, y=fitting_result.best_fit, label='Exponential Growth Fit', linestyle='--', color='darkorange', linewidth=2, ax=ax)

# Confidence interval plot with a different fill color
    ax.fill_between(time_points, fitting_result.best_fit - fitting_result.eval_uncertainty(),
                fitting_result.best_fit + fitting_result.eval_uncertainty(),
                alpha=0.2, label='Confidence Interval', color='lightgray')

# Set labels and title with a creative touch
    ax.set_xlabel('Time (Years)', fontsize=14, fontweight='bold', color='darkslategray')
    ax.set_ylabel('Investment Inflows (% of GDP)', fontsize=14, fontweight='bold', color='darkslategray')
    ax.set_ylim(0, 30)
    ax.set_title('Curve Fitting for Foreign Direct Investment Over Time', fontsize=16, fontweight='bold', color='darkorange')

# Add legend with a personalized style
    ax.legend(frameon=False, fontsize=12, loc='upper left')

# Remove spines for a cleaner look
    sns.despine()

# Add a personalized touch to grid lines
    ax.grid(True, linestyle='--', alpha=0.3, color='darkslategray')

# Show the plot
    plt.show()
def scatter_plot_countries_gdp_foreign_investment(data, cluster_centers):
    plt.figure(figsize=(12, 8))

    # Create a customized color palette
    custom_palette = sns.color_palette("coolwarm", as_cmap=True)

    # Manually set color for Cluster 1
    cluster_colors = {0: 'blue', 1: 'green', 2: 'purple'}  # Add more colors as needed

    # Enhance the scatter plot using sns.scatterplot
    sns.scatterplot(x="GDP per capita growth (annual %)", 
                    y="Foreign direct investment net inflows (% of GDP)",
                    hue="Cluster", palette=custom_palette, size=data["Cluster"], sizes=(40, 200),
                    data=data, linewidth=0.5, alpha=0.7)

    # Highlight cluster centers with a distinctive marker
    plt.scatter(cluster_centers[:, 1], cluster_centers[:, 3], marker='x', s=150,  label='Centroids')

    # Manually set color for Cluster 1
    plt.scatter(data[data['Cluster'] == 1]["GDP per capita growth (annual %)"],
                data[data['Cluster'] == 1]["Foreign direct investment net inflows (% of GDP)"],
                color=cluster_colors[1], label='Cluster 1', s=80, linewidth=0.5, alpha=0.7)

    # Add title and labels with a touch of creativity
    plt.title('Country Clustering with Distinctive Centroids', fontsize=16, fontweight='bold', color='darkgreen')
    plt.ylabel('Foreign direct Investment', fontsize=14, fontweight='bold')
    plt.xlabel("GDP per capita growth (annual %)", fontsize=14, fontweight='bold')
    plt.legend(title='Clusters', title_fontsize='12')

    # Add a touch of transparency to the background
    plt.gca().set_facecolor('lightgray')

    # Adjust the appearance of ticks and spines
    plt.tick_params(axis='both', colors='black')
    for spine in plt.gca().spines.values():
        spine.set_edgecolor('black')

    # Enhance readability by removing grid lines
    sns.despine()

    # Add a personalized touch to grid lines
    plt.grid(True, linestyle='--', alpha=0.3, color='black')

    # Display the custom plot
    plt.show()
def plot__emissions_over_time(time_points, green_house_data, predicted_for_years, predicted_values):
  
    # Set custom font to a system font
    custom_font = fm.FontProperties(fname=fm.findfont(fm.FontProperties(family='arial')))

    # Set a beautiful style
    sns.set(style="whitegrid")
    
    plt.figure(figsize=(12, 8))

    # Line plot for actual Foreign direct investment
    sns.lineplot(x=time_points, y=green_house_data, label='Green house Emissions (Actual)', color='#4C72B0', linewidth=2)

    # Line plot for predicted values
    plt.plot(predicted_for_years, predicted_values, label='Predicted Values', color='#55A868', linestyle='--', linewidth=2)

    plt.xlabel('Time', fontproperties=custom_font, fontsize=14)
    plt.ylabel('Foreign direct investment net inflows (% of GDP)', fontproperties=custom_font, fontsize=14)
    plt.title("Other greenhouse gas emissions (% change from 1990)", fontproperties=custom_font, fontsize=16)
    plt.legend(prop=custom_font)

    # Set background color
    plt.gca().set_facecolor('#F0F0F0')

    # Set tick color
    plt.tick_params(axis='both', colors='black')

    # Set spines color
    for spine in plt.gca().spines.values():
        spine.set_edgecolor('#000000')

    # Add grid lines
    plt.grid(True, linestyle='--', alpha=0.5)

    # Add a note about extrapolation
    plt.annotate('Note: Future predictions beyond observed data are extrapolations. Exercise caution in interpretation.',
                 xy=(0.5, -0.1), xycoords="axes fraction",
                 ha="center", va="center", fontsize=10, color='gray', style='italic')

    plt.show()   

def extract_country_data(cleaned_data, country_name):
    country_data = cleaned_data[cleaned_data['Country Name'] == country_name]
    time_points = country_data['Time']
    investment = country_data['Foreign direct investment net inflows (% of GDP)']
    return time_points, investment


def plot_investment_graphs(ax, time_points, investment, predicted_for_years, predicted_values, country_name):
    
    ax.scatter(time_points, investment, label=f'{country_name} Actual Data', marker='o')
    ax.plot(predicted_for_years, predicted_values, label=f'{country_name} Predicted Data', linestyle='--')
    ax.set_xlabel('Year')
    ax.set_ylabel('Foreign direct investment ')
    ax.set_title(f'Foreign direct investment Over Time - {country_name}')
    ax.legend()


def get_countries_future_predictions():
    countries = ['China', 'Canada', 'Pakistan', 'Australia']

    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    axs = axs.flatten()

    for i, country in enumerate(countries):
        ax = axs[i]
        time_points_country, investment_countries = extract_country_data(cleaned_data, country)
        predicted_values_country = fitting_result.eval(x=np.array(predicted_for_years))
        plot_investment_graphs(ax, time_points_country, investment_countries, predicted_for_years, predicted_values_country, country)

    plt.tight_layout()
    plt.show()

filepath = "countries_data_indicators.csv"
indicators = [
     "Foreign direct investment net inflows (% of GDP)",
     "Other greenhouse gas emissions (% change from 1990)",
     "Electric power consumption (kWh per capita)",
     "GDP per capita growth (annual %)"
]
cleaned_data = read_clean_data(filepath,indicators)

print(cleaned_data)

scaler = StandardScaler()

normalized_data = scaler.fit_transform(cleaned_data[indicators])

kmeans = KMeans(n_clusters=3, random_state=42)
cleaned_data['Cluster'] = kmeans.fit_predict(normalized_data)

cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)

silhouette_avg = silhouette_score(normalized_data, cleaned_data['Cluster'])
print(f"Silhouette Score: {silhouette_avg:.4f}")

scatter_plot_countries_gdp_foreign_investment(cleaned_data, cluster_centers)

time_points = cleaned_data['Time']
greenhouse_data = cleaned_data['Foreign direct investment net inflows (% of GDP)']

fitting_result = fit_exponential_growth_model(time_points, greenhouse_data)

# Plot the curve fit for the entire dataset
plot_curve_fit_with_confidence_interval(time_points, greenhouse_data, fitting_result)

# Generate time points for prediction for the entire dataset
predicted_for_years = [2030, 2035, 2040,2045,2050]

# Predict values for the future years using the fitted model for the entire dataset
predicted_values = fitting_result.eval(x=np.array(predicted_for_years))

# Display the predicted values for the entire dataset
for year, value in zip(predicted_for_years, predicted_values):
    print(f"Predictions {value:.2f}")

# Filter data for China

get_countries_future_predictions()
