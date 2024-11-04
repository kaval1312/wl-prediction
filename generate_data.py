import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Set random seed for reproducibility
np.random.seed(42)

# Generate dates for 5 years of daily data
start_date = datetime(2019, 1, 1)
dates = [start_date + timedelta(days=x) for x in range(365 * 5)]  # 5 years of data

# Well names with different characteristics
wells = {
    'Well_Superman': {'basin': 'Permian', 'type': 'Horizontal', 'depth': 12000},
    'Well_Joker': {'basin': 'Bakken', 'type': 'Vertical', 'depth': 8000},
    'Well_Batman': {'basin': 'Eagle Ford', 'type': 'Horizontal', 'depth': 10000},
    'Well_Ironman': {'basin': 'Permian', 'type': 'Directional', 'depth': 11000},
    'Well_Spiderman': {'basin': 'Bakken', 'type': 'Horizontal', 'depth': 9500}
}


def generate_series(base_value, trend, noise_level, length, seasonality=False):
    trend_series = np.linspace(0, trend, length)
    noise = np.random.normal(0, noise_level, length)
    if seasonality:
        seasonal_component = np.sin(np.linspace(0, 4 * np.pi, length)) * base_value * 0.1
        return np.maximum(0, base_value + trend_series + noise + seasonal_component)
    return np.maximum(0, base_value + trend_series + noise)


all_data = []

for well_name, well_info in wells.items():
    # Base characteristics affect well performance
    depth_factor = well_info['depth'] / 10000
    type_factor = 1.2 if well_info['type'] == 'Horizontal' else 0.8
    base = depth_factor * type_factor

    for date in dates:
        # Weather and seasonal factors
        season_factor = 1 + 0.1 * np.sin(2 * np.pi * date.timetuple().tm_yday / 365)
        is_winter = date.month in [12, 1, 2]

        # Environmental Data
        oil_production = generate_series(1000 * base, -100, 50, 1, seasonality=True)[0]
        water_cut = generate_series(20 * base, 15, 5, 1)[0]
        gas_production = generate_series(500 * base, -50, 25, 1, seasonality=True)[0]

        # Reservoir Characteristics
        reservoir_data = {
            'Reservoir_Pressure_PSI': round(generate_series(2000 * base, -200, 50, 1)[0], 2),
            'Well_Head_Temperature_F': round(generate_series(150 * base * season_factor, 0, 5, 1)[0], 2),
            'Porosity_Percentage': round(generate_series(12 * base, -0.1, 0.2, 1)[0], 2),
            'Permeability_mD': round(generate_series(50 * base, -1, 2, 1)[0], 2),
            'Formation_Volume_Factor': round(generate_series(1.2 * base, 0, 0.05, 1)[0], 3),
            'Gas_Oil_Ratio': round(generate_series(1000 * base, 50, 20, 1)[0], 2)
        }

        # Financial Data
        price_factors = {
            'Oil_Price_USD': round(generate_series(70, 5, 3, 1, seasonality=True)[0], 2),
            'Gas_Price_USD': round(generate_series(3.5, 0.2, 0.1, 1, seasonality=True)[0], 2)
        }

        financial_data = {
            'Well_Lease_Cost_USD': round(generate_series(5000 * base, 100, 200, 1)[0], 2),
            'Salary_Cost_USD': round(generate_series(8000 * base, 200, 300, 1)[0], 2),
            'Insurance_Cost_USD': round(generate_series(2000 * base, 50, 100, 1)[0], 2),
            'Environmental_Compliance_Cost_USD': round(generate_series(1500 * base, 100, 50, 1)[0], 2),
            'Water_Disposal_Cost_USD': round(generate_series(1000 * base, 50, 30, 1)[0], 2),
            'Chemical_Treatment_Cost_USD': round(generate_series(800 * base, 30, 20, 1)[0], 2)
        }

        # Equipment Data
        equipment_data = {
            'Pump_Efficiency_Percentage': round(generate_series(95, -5, 2, 1)[0], 2),
            'Pump_Power_Usage_KWh': round(generate_series(100 * base, 2, 5, 1)[0], 2),
            'Tubing_Pressure_PSI': round(generate_series(1000 * base, -10, 20, 1)[0], 2),
            'Casing_Pressure_PSI': round(generate_series(500 * base, -5, 10, 1)[0], 2),
            'Wellhead_Temperature_F': round(generate_series(120 * base * season_factor, 0, 5, 1)[0], 2),
            'Artificial_Lift_Power_KWh': round(generate_series(150 * base, 3, 7, 1)[0], 2)
        }

        # Maintenance Schedule
        maintenance_data = {
            'Last_Workover_Days': round(generate_series(365, 1, 0, 1)[0], 0),
            'Next_Maintenance_Days': round(generate_series(30, -1, 0, 1)[0], 0),
            'Equipment_Runtime_Hours': round(generate_series(24, 0, 0.5, 1)[0], 2),
            'Downtime_Hours': round(max(0, 24 - generate_series(24, 0, 0.5, 1)[0]), 2)
        }

        # Production Tests
        test_data = {
            'Oil_Cut_Percentage': round(100 - water_cut, 2),
            'Gas_Oil_Ratio_SCF_BBL': round(gas_production / oil_production if oil_production > 0 else 0, 2),
            'Water_Oil_Ratio': round(water_cut / (100 - water_cut) if water_cut < 100 else 999, 2),
            'Liquid_Rate_BBL_Day': round(oil_production * (1 + water_cut / 100), 2)
        }

        # Calculate revenues and costs
        revenue = (oil_production * price_factors['Oil_Price_USD'] +
                   gas_production * price_factors['Gas_Price_USD'])

        operating_costs = sum(financial_data.values())
        maintenance_cost = generate_series(2000 * base, 50, 100, 1)[0]

        # Random major equipment failures and replacements
        equipment_purchase = 0
        if np.random.random() < 0.005:  # 0.5% chance each day
            equipment_purchase = np.random.choice([50000, 75000, 100000, 150000, 200000])
            maintenance_data['Last_Workover_Days'] = 0

        # Combine all data
        record = {
            'Date': date,
            'Well_Name': well_name,
            'Basin': well_info['basin'],
            'Well_Type': well_info['type'],
            'Well_Depth_Ft': well_info['depth'],

            # Production
            'Oil_Production_BBL': round(oil_production, 2),
            'Water_Cut_Percentage': round(water_cut, 2),
            'Gas_Production_MCF': round(gas_production, 2),

            # Prices
            **price_factors,

            # Financial
            'Revenue_USD': round(revenue, 2),
            'Operating_Cost_USD': round(operating_costs, 2),
            'Maintenance_Cost_USD': round(maintenance_cost, 2),
            'Equipment_Purchase_USD': round(equipment_purchase, 2),
            'Net_Income_USD': round(revenue - operating_costs - maintenance_cost - equipment_purchase, 2),

            # Include all other data
            **reservoir_data,
            **financial_data,
            **equipment_data,
            **maintenance_data,
            **test_data
        }

        # Add some risk factors
        record.update({
            'Environmental_Risk_Score': round(np.random.uniform(1, 5), 2),
            'Equipment_Risk_Score': round(np.random.uniform(1, 5), 2),
            'Financial_Risk_Score': round(np.random.uniform(1, 5), 2),
            'Overall_Risk_Score': round(np.random.uniform(1, 5), 2)
        })

        all_data.append(record)

# Create DataFrame
df = pd.DataFrame(all_data)

# Calculate some rolling averages and trends
for well in wells.keys():
    well_data = df[df['Well_Name'] == well].copy()

    # 30-day rolling averages
    for col in ['Oil_Production_BBL', 'Gas_Production_MCF', 'Net_Income_USD']:
        df.loc[df['Well_Name'] == well, f'{col}_30D_Avg'] = well_data[col].rolling(30).mean()

    # Production decline rates
    df.loc[df['Well_Name'] == well, 'Production_Decline_Rate'] = well_data['Oil_Production_BBL'].pct_change(365)

# Sort the data
df = df.sort_values(['Well_Name', 'Date'])

# Save to CSV
df.to_csv('enhanced_well_data.csv', index=False)

# Print summary of available columns
print("\nAvailable Columns:")
for col in sorted(df.columns):
    print(f"- {col}")

print("\nData Shape:", df.shape)
print("\nDate Range:", df['Date'].min(), "to", df['Date'].max())
print("\nSample Statistics:")
print(df[['Oil_Production_BBL', 'Gas_Production_MCF', 'Net_Income_USD']].describe())