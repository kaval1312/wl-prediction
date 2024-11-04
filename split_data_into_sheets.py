import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

# Set random seed for reproducibility
np.random.seed(42)

# Generate dates for 5 years of daily data
start_date = datetime(2019, 1, 1)
dates = [start_date + timedelta(days=x) for x in range(365 * 5)]

# Well characteristics with enhanced environmental factors
wells = {
    'Well_Superman': {
        'basin': 'Permian',
        'type': 'Horizontal',
        'depth': 12000,
        'crew_size': 8,
        'contractor': 'ABC Services',
        'lease_terms': '5 years',
        'initial_cost': 5000000,
        'environmental_sensitivity': 'High',
        'water_source': 'Local Aquifer',
        'proximity_to_protected_areas': 'Near',
        'soil_type': 'Sandy Loam',
        'emission_control_system': 'Advanced',
        'spill_prevention_system': 'Tier 3',
        'groundwater_monitoring': 'Continuous'
    },
    'Well_Joker': {
        'basin': 'Bakken',
        'type': 'Vertical',
        'depth': 8000,
        'crew_size': 6,
        'contractor': 'XYZ Drilling',
        'lease_terms': '3 years',
        'initial_cost': 3000000,
        'environmental_sensitivity': 'Medium',
        'water_source': 'Municipal',
        'proximity_to_protected_areas': 'Far',
        'soil_type': 'Clay',
        'emission_control_system': 'Standard',
        'spill_prevention_system': 'Tier 2',
        'groundwater_monitoring': 'Weekly'
    },
    'Well_Batman': {
        'basin': 'Eagle Ford',
        'type': 'Horizontal',
        'depth': 10000,
        'crew_size': 7,
        'contractor': 'DEF Energy',
        'lease_terms': '4 years',
        'initial_cost': 4500000,
        'environmental_sensitivity': 'High',
        'water_source': 'River',
        'proximity_to_protected_areas': 'Medium',
        'soil_type': 'Silt',
        'emission_control_system': 'Advanced',
        'spill_prevention_system': 'Tier 3',
        'groundwater_monitoring': 'Daily'
    },
    'Well_Ironman': {
        'basin': 'Permian',
        'type': 'Directional',
        'depth': 11000,
        'crew_size': 7,
        'contractor': 'GHI Drilling',
        'lease_terms': '5 years',
        'initial_cost': 4800000,
        'environmental_sensitivity': 'Medium',
        'water_source': 'Local Aquifer',
        'proximity_to_protected_areas': 'Far',
        'soil_type': 'Sandy Clay',
        'emission_control_system': 'Advanced',
        'spill_prevention_system': 'Tier 2',
        'groundwater_monitoring': 'Daily'
    },
    'Well_Spiderman': {
        'basin': 'Bakken',
        'type': 'Horizontal',
        'depth': 9500,
        'crew_size': 6,
        'contractor': 'JKL Services',
        'lease_terms': '4 years',
        'initial_cost': 4200000,
        'environmental_sensitivity': 'Low',
        'water_source': 'Municipal',
        'proximity_to_protected_areas': 'Far',
        'soil_type': 'Loam',
        'emission_control_system': 'Standard',
        'spill_prevention_system': 'Tier 1',
        'groundwater_monitoring': 'Weekly'
    }
}

# Maintenance items and their base costs
maintenance_items = {
    'Replacement_Parts': {
        'ESP_System': {
            'Pump': 15000,
            'Motor': 12000,
            'Cable': 5000,
            'Sensor_Package': 3000,
            'Controller': 4000,
            'VFD': 8000
        },
        'Wellhead_Equipment': {
            'Christmas_Tree': 20000,
            'Valves': 5000,
            'Pressure_Gauges': 1000,
            'Safety_Devices': 3000,
            'Flanges': 2000,
            'Spools': 1500
        },
        'Downhole_Equipment': {
            'Tubing': 10000,
            'Packers': 8000,
            'Safety_Valves': 6000,
            'Gas_Lift_Valves': 4000,
            'Production_Liner': 15000,
            'Sand_Screens': 7000
        },
        'Surface_Equipment': {
            'Separators': 25000,
            'Heater_Treaters': 20000,
            'Storage_Tanks': 30000,
            'Compressors': 15000,
            'Pumping_Units': 18000
        }
    },
    'Services': {
        'Regular_Maintenance': {
            'Well_Logging': 5000,
            'Pressure_Testing': 3000,
            'Flow_Testing': 2500,
            'Corrosion_Monitoring': 2000,
            'Scale_Treatment': 1500
        },
        'Cleaning': {
            'Tank_Cleaning': 4000,
            'Pipeline_Pigging': 3500,
            'Well_Bore_Cleaning': 5000,
            'Surface_Equipment_Cleaning': 2000
        },
        'Inspection': {
            'NDT_Inspection': 3000,
            'Corrosion_Inspection': 2500,
            'Safety_System_Check': 2000,
            'Environmental_Compliance_Check': 2500
        }
    },
    'Repairs': {
        'Emergency_Repairs': {
            'Leak_Repair': 10000,
            'Equipment_Failure': 15000,
            'Power_System': 8000,
            'Control_System': 5000
        },
        'Scheduled_Repairs': {
            'Worn_Parts': 5000,
            'Calibration': 2000,
            'Preventive_Replacement': 4000
        }
    }
}

environmental_categories = {
    'Emissions_Monitoring': {
        'Methane_Emissions': {
            'Continuous_Monitoring': 1000,
            'Leak_Detection': 800,
            'Repair_Program': 1200,
            'Reporting': 500
        },
        'CO2_Emissions': {
            'Monitoring': 900,
            'Reporting': 400,
            'Offset_Programs': 2000
        },
        'VOC_Emissions': {
            'Monitoring': 700,
            'Control_Systems': 1500,
            'Reporting': 300
        }
    },
    'Water_Management': {
        'Produced_Water': {
            'Treatment': 2000,
            'Disposal': 1500,
            'Recycling': 2500,
            'Quality_Testing': 800
        },
        'Groundwater': {
            'Monitoring_Wells': 1200,
            'Sample_Analysis': 600,
            'Report_Generation': 400,
            'Remediation': 5000
        },
        'Surface_Water': {
            'Quality_Monitoring': 700,
            'Runoff_Control': 900,
            'Spill_Prevention': 1000
        }
    },
    'Soil_Management': {
        'Contamination_Monitoring': {
            'Sampling': 500,
            'Analysis': 800,
            'Reporting': 300
        },
        'Remediation': {
            'Soil_Treatment': 3000,
            'Disposal': 2000,
            'Site_Restoration': 5000
        }
    },
    'Waste_Management': {
        'Drilling_Waste': {
            'Treatment': 1500,
            'Disposal': 2000,
            'Recycling': 1000
        },
        'Chemical_Waste': {
            'Storage': 800,
            'Treatment': 1200,
            'Disposal': 1500
        },
        'General_Waste': {
            'Collection': 300,
            'Sorting': 200,
            'Disposal': 500
        }
    },
    'Wildlife_Protection': {
        'Habitat_Preservation': {
            'Monitoring': 1000,
            'Protection_Measures': 1500,
            'Restoration': 3000
        },
        'Species_Monitoring': {
            'Surveys': 800,
            'Reporting': 400,
            'Mitigation': 2000
        }
    }
}

financial_categories = {
    'Labor': {
        'Operations': {
            'Operators': 5000,
            'Technicians': 4500,
            'Engineers': 7000,
            'Supervisors': 8000,
            'Support_Staff': 3500
        },
        'Maintenance': {
            'Maintenance_Crew': 4000,
            'Specialists': 6000,
            'Contractors': 5500
        },
        'Management': {
            'Site_Manager': 10000,
            'HSE_Manager': 8000,
            'Technical_Manager': 9000
        }
    },
    'Insurance': {
        'Operational': {
            'Equipment_Insurance': 2000,
            'Business_Interruption': 3000,
            'General_Liability': 2500
        },
        'Environmental': {
            'Pollution_Liability': 4000,
            'Environmental_Damage': 3500,
            'Remediation_Cost': 3000
        },
        'Personnel': {
            'Workers_Comp': 1500,
            'Health_Insurance': 2000,
            'Life_Insurance': 1000
        }
    },
    'Lease': {
        'Land': {
            'Surface_Rights': 10000,
            'Mineral_Rights': 15000,
            'Access_Rights': 5000
        },
        'Equipment': {
            'Heavy_Machinery': 8000,
            'Vehicles': 3000,
            'Temporary_Equipment': 2000
        },
        'Facilities': {
            'Office_Space': 2000,
            'Storage': 1500,
            'Worker_Facilities': 1000
        }
    },
    'Regulatory_Compliance': {
        'Permits': {
            'Drilling_Permits': 5000,
            'Environmental_Permits': 4000,
            'Operating_Permits': 3000
        },
        'Reporting': {
            'Environmental_Reports': 2000,
            'Production_Reports': 1500,
            'Safety_Reports': 1000
        },
        'Audits': {
            'Environmental_Audits': 3000,
            'Safety_Audits': 2500,
            'Financial_Audits': 4000
        }
    }
}

monte_carlo_params = {
    'Production': {
        'Oil_Rate': {
            'min': 0.7,
            'most_likely': 1.0,
            'max': 1.3
        },
        'Gas_Rate': {
            'min': 0.6,
            'most_likely': 1.0,
            'max': 1.4
        },
        'Water_Cut': {
            'min': 0.8,
            'most_likely': 1.0,
            'max': 1.2
        }
    },
    'Costs': {
        'Operating_Cost': {
            'min': 0.8,
            'most_likely': 1.0,
            'max': 1.3
        },
        'Maintenance_Cost': {
            'min': 0.7,
            'most_likely': 1.0,
            'max': 1.5
        }
    },
    'Prices': {
        'Oil_Price': {
            'min': 40,
            'most_likely': 70,
            'max': 100
        },
        'Gas_Price': {
            'min': 2,
            'most_likely': 3.5,
            'max': 5
        }
    },
    'Environmental': {
        'Emission_Rates': {
            'min': 0.8,
            'most_likely': 1.0,
            'max': 1.2
        },
        'Water_Treatment_Cost': {
            'min': 0.9,
            'most_likely': 1.0,
            'max': 1.4
        }
    }
}

def run_monte_carlo_simulation(base_values, params, n_simulations=1000):
    """Run Monte Carlo simulation using numpy's random functions"""
    results = {}
    for category, items in params.items():
        category_results = {}
        for item, ranges in items.items():
            simulated_values = np.random.triangular(
                ranges['min'],
                ranges['most_likely'],
                ranges['max'],
                size=n_simulations
            )
            category_results[item] = simulated_values
        results[category] = category_results
    return results

def generate_series(base_value, trend, noise_level, length, seasonality=False):
    """Generate time series data with trend and optional seasonality"""
    trend_series = np.linspace(0, trend, length)
    noise = np.random.normal(0, noise_level, length)
    if seasonality:
        seasonal_component = np.sin(np.linspace(0, 4*np.pi, length)) * base_value * 0.1
        return np.maximum(0, base_value + trend_series + noise + seasonal_component)
    return np.maximum(0, base_value + trend_series + noise)

def generate_environmental_data(date, base_factor, well_info):
    """Generate environmental monitoring data"""
    env_data = {}
    for category, subcategories in environmental_categories.items():
        for subcat, items in subcategories.items():
            for item, base_cost in items.items():
                seasonal_factor = 1 + 0.1 * np.sin(2 * np.pi * date.timetuple().tm_yday / 365)
                sensitivity_factor = 1.2 if well_info['environmental_sensitivity'] == 'High' else 1.0
                cost = base_cost * base_factor * seasonal_factor * sensitivity_factor
                env_data[f'{category}_{subcat}_{item}'] = round(cost * (1 + np.random.normal(0, 0.1)), 2)
    return env_data

def generate_maintenance_events(date, base_factor):
    """Generate maintenance events and costs"""
    maintenance_records = []
    for category, subcategories in maintenance_items.items():
        for subcat, items in subcategories.items():
            for item, base_cost in items.items():
                if np.random.random() < 0.02:  # 2% chance each day
                    cost = base_cost * base_factor * (1 + np.random.normal(0, 0.2))
                    maintenance_records.append({
                        'Maintenance_Category': category,
                        'Subcategory': subcat,
                        'Maintenance_Item': item,
                        'Cost': round(cost, 2),
                        'Status': np.random.choice(['Completed', 'In Progress', 'Scheduled']),
                        'Priority': np.random.choice(['High', 'Medium', 'Low'])
                    })
    return maintenance_records

def generate_financial_details(date, base_factor, well_info):
    """Generate detailed financial records"""
    financial_records = {}
    for category, subcategories in financial_categories.items():
        category_total = 0
        for subcat, items in subcategories.items():
            subcat_total = 0
            for item, base_cost in items.items():
                if category == 'Labor':
                    cost = base_cost * well_info['crew_size'] * base_factor * (1 + np.random.normal(0, 0.1))
                else:
                    cost = base_cost * base_factor * (1 + np.random.normal(0, 0.15))
                financial_records[f'{category}_{subcat}_{item}'] = round(cost, 2)
                subcat_total += cost
            financial_records[f'{category}_{subcat}_Total'] = round(subcat_total, 2)
            category_total += subcat_total
        financial_records[f'{category}_Total'] = round(category_total, 2)
    return financial_records

# Main data generation
print("Starting data generation...")
all_data = []
maintenance_data = []
financial_data = []
environmental_data = []
monte_carlo_results = {}

for well_name, well_info in wells.items():
    print(f"\nGenerating data for {well_name}...")

    monte_carlo_results[well_name] = run_monte_carlo_simulation(None, monte_carlo_params)

    depth_factor = well_info['depth'] / 10000
    type_factor = 1.2 if well_info['type'] == 'Horizontal' else 0.8
    base = depth_factor * type_factor

    for date in dates:
        # Generate base record
        season_factor = 1 + 0.1 * np.sin(2 * np.pi * date.timetuple().tm_yday / 365)

        # Production data with Monte Carlo adjustments
        mc_factor = np.random.choice(monte_carlo_results[well_name]['Production']['Oil_Rate'])
        oil_production = generate_series(1000 * base * mc_factor, -100, 50, 1, seasonality=True)[0]

        mc_factor = np.random.choice(monte_carlo_results[well_name]['Production']['Water_Cut'])
        water_cut = generate_series(20 * base * mc_factor, 15, 5, 1)[0]

        mc_factor = np.random.choice(monte_carlo_results[well_name]['Production']['Gas_Rate'])
        gas_production = generate_series(500 * base * mc_factor, -50, 25, 1, seasonality=True)[0]

        base_record = {
            'Date': date,
            'Well_Name': well_name,
            'Basin': well_info['basin'],
            'Well_Type': well_info['type'],
            'Well_Depth_Ft': well_info['depth'],
            'Contractor': well_info['contractor'],
            'Crew_Size': well_info['crew_size'],
            'Lease_Terms': well_info['lease_terms'],
            'Initial_Cost': well_info['initial_cost'],
            'Environmental_Sensitivity': well_info['environmental_sensitivity'],
            'Water_Source': well_info['water_source'],
            'Proximity_To_Protected_Areas': well_info['proximity_to_protected_areas'],
            'Soil_Type': well_info['soil_type'],
            'Emission_Control_System': well_info['emission_control_system'],
            'Spill_Prevention_System': well_info['spill_prevention_system'],
            'Groundwater_Monitoring': well_info['groundwater_monitoring'],

            # Production metrics
            'Oil_Production_BBL': round(oil_production, 2),
            'Water_Cut_Percentage': round(water_cut, 2),
            'Gas_Production_MCF': round(gas_production, 2),
            'Liquid_Rate_BBL_Day': round(oil_production * (1 + water_cut/100), 2),
            'Gas_Oil_Ratio': round(gas_production / oil_production if oil_production > 0 else 0, 2),
            'Water_Oil_Ratio': round(water_cut / (100 - water_cut) if water_cut < 100 else 999, 2),

            # Reservoir conditions
            'Reservoir_Pressure_PSI': round(generate_series(2000 * base, -200, 50, 1)[0], 2),
            'Bottom_Hole_Temperature_F': round(generate_series(180 * base, 0, 5, 1)[0], 2),
            'Porosity_Percentage': round(generate_series(12 * base, -0.1, 0.2, 1)[0], 2),
            'Permeability_mD': round(generate_series(50 * base, -1, 2, 1)[0], 2),

            # Equipment performance
            'Pump_Efficiency_Percentage': round(generate_series(95, -5, 2, 1)[0], 2),
            'Runtime_Hours': round(generate_series(24, 0, 0.5, 1)[0], 2),
            'Downtime_Hours': round(max(0, 24 - generate_series(24, 0, 0.5, 1)[0]), 2),
            'Power_Usage_KWh': round(generate_series(100 * base, 2, 5, 1)[0], 2)
        }

        # Add maintenance records
        maint_records = generate_maintenance_events(date, base)
        for record in maint_records:
            maintenance_record = base_record.copy()
            maintenance_record.update(record)
            maintenance_data.append(maintenance_record)

        # Add environmental monitoring data
        env_record = base_record.copy()
        env_record.update(generate_environmental_data(date, base, well_info))
        environmental_data.append(env_record)

        # Add financial details
        fin_record = base_record.copy()
        fin_record.update(generate_financial_details(date, base, well_info))
        financial_data.append(fin_record)

        # Add to main records
        all_data.append(base_record)

# Create DataFrames
print("\nCreating DataFrames...")
df_main = pd.DataFrame(all_data)
df_maintenance = pd.DataFrame(maintenance_data)
df_financial = pd.DataFrame(financial_data)
df_environmental = pd.DataFrame(environmental_data)

# Calculate metrics
print("\nCalculating metrics...")
for well in wells.keys():
    well_mask = df_main['Well_Name'] == well
    well_data = df_main[well_mask].copy()

    if not well_data.empty:
        well_data = well_data.sort_values('Date')

        # Production metrics
        for col in ['Oil_Production_BBL', 'Gas_Production_MCF', 'Water_Cut_Percentage']:
            rolling_avg = well_data[col].rolling(window=30, min_periods=1).mean()
            df_main.loc[well_mask, f'{col}_30D_Avg'] = rolling_avg

        # Production decline rate
        if len(well_data) >= 365:
            decline_rate = well_data['Oil_Production_BBL'].pct_change(periods=365)
            df_main.loc[well_mask, 'Production_Decline_Rate'] = decline_rate
        else:
            df_main.loc[well_mask, 'Production_Decline_Rate'] = np.nan

        # Equipment performance metrics
        for col in ['Pump_Efficiency_Percentage', 'Runtime_Hours']:
            rolling_avg = well_data[col].rolling(window=7, min_periods=1).mean()
            df_main.loc[well_mask, f'{col}_7D_Avg'] = rolling_avg

# Save data with tabs for each well
print("\nSaving data files with tabs per well...")
try:
    writers = {
        'production': pd.ExcelWriter('production_data.xlsx', engine='openpyxl'),
        'maintenance': pd.ExcelWriter('maintenance_data.xlsx', engine='openpyxl'),
        'environmental': pd.ExcelWriter('environmental_data.xlsx', engine='openpyxl'),
        'financial': pd.ExcelWriter('financial_data.xlsx', engine='openpyxl'),
        'monte_carlo': pd.ExcelWriter('monte_carlo_data.xlsx', engine='openpyxl')
    }

    # Process each well
    for well_name in wells.keys():
        print(f"\nProcessing {well_name}...")
        well_prefix = well_name.replace('Well_', '')

        # Production Data
        well_production = df_main[df_main['Well_Name'] == well_name].copy()
        well_production.to_excel(writers['production'], sheet_name=well_prefix, index=False)

        # Maintenance Data
        well_maintenance = df_maintenance[df_maintenance['Well_Name'] == well_name].copy()
        well_maintenance.to_excel(writers['maintenance'], sheet_name=well_prefix, index=False)

        # Environmental Data
        well_environmental = df_environmental[df_environmental['Well_Name'] == well_name].copy()
        well_environmental.to_excel(writers['environmental'], sheet_name=well_prefix, index=False)

        # Financial Data
        well_financial = df_financial[df_financial['Well_Name'] == well_name].copy()
        well_financial.to_excel(writers['financial'], sheet_name=well_prefix, index=False)

        # Monte Carlo Data
        well_monte_carlo = pd.DataFrame(monte_carlo_results[well_name]).reset_index()
        well_monte_carlo.to_excel(writers['monte_carlo'], sheet_name=well_prefix, index=False)

    # Add summary sheets
    summary_df = pd.DataFrame({
        'Well_Name': list(wells.keys()),
        'Basin': [wells[w]['basin'] for w in wells.keys()],
        'Type': [wells[w]['type'] for w in wells.keys()],
        'Depth_Ft': [wells[w]['depth'] for w in wells.keys()],
        'Environmental_Sensitivity': [wells[w]['environmental_sensitivity'] for w in wells.keys()]
    })

    for writer in writers.values():
        summary_df.to_excel(writer, sheet_name='Summary', index=False)
        writer.close()

    print("\nAll files created successfully!")

except Exception as e:
    print(f"Error saving files: {str(e)}")
    print("\nDetailed error information:")
    import traceback
    traceback.print_exc()

    # Close any open writers
    for writer in writers.values():
        try:
            writer.close()
        except:
            pass

# Print final statistics
print("\nFinal Data Summary:")
print(f"Total number of days: {len(dates)}")
print(f"Number of wells: {len(wells)}")
print(f"Records in main data: {len(df_main):,}")
print(f"Maintenance events: {len(df_maintenance):,}")
print(f"Environmental records: {len(df_environmental):,}")
print(f"Financial records: {len(df_financial):,}")

# Check for missing values
print("\nMissing Values Check:")
for col in df_main.columns:
    missing = df_main[col].isna().sum()
    if missing > 0:
        print(f"- {col}: {missing} missing values")










# import pandas as pd
# import numpy as np
# from datetime import datetime, timedelta
# np.random.seed(42)
# start_date = datetime(2019, 1, 1)
# dates = [start_date + timedelta(days=x) for x in range(365 * 5)]
#
# # Well characteristics with enhanced environmental factors
# wells = {
#     'Well_Superman': {
#         'basin': 'Permian',
#         'type': 'Horizontal',
#         'depth': 12000,
#         'crew_size': 8,
#         'contractor': 'ABC Services',
#         'lease_terms': '5 years',
#         'initial_cost': 5000000,
#         'environmental_sensitivity': 'High',
#         'water_source': 'Local Aquifer',
#         'proximity_to_protected_areas': 'Near',
#         'soil_type': 'Sandy Loam',
#         'emission_control_system': 'Advanced',
#         'spill_prevention_system': 'Tier 3',
#         'groundwater_monitoring': 'Continuous'
#     },
#     'Well_Joker': {
#         'basin': 'Bakken',
#         'type': 'Vertical',
#         'depth': 8000,
#         'crew_size': 6,
#         'contractor': 'XYZ Drilling',
#         'lease_terms': '3 years',
#         'initial_cost': 3000000,
#         'environmental_sensitivity': 'Medium',
#         'water_source': 'Municipal',
#         'proximity_to_protected_areas': 'Far',
#         'soil_type': 'Clay',
#         'emission_control_system': 'Standard',
#         'spill_prevention_system': 'Tier 2',
#         'groundwater_monitoring': 'Weekly'
#     },
#     'Well_Batman': {
#         'basin': 'Eagle Ford',
#         'type': 'Horizontal',
#         'depth': 10000,
#         'crew_size': 7,
#         'contractor': 'DEF Energy',
#         'lease_terms': '4 years',
#         'initial_cost': 4500000,
#         'environmental_sensitivity': 'High',
#         'water_source': 'River',
#         'proximity_to_protected_areas': 'Medium',
#         'soil_type': 'Silt',
#         'emission_control_system': 'Advanced',
#         'spill_prevention_system': 'Tier 3',
#         'groundwater_monitoring': 'Daily'
#     },
#     'Well_Ironman': {
#         'basin': 'Permian',
#         'type': 'Directional',
#         'depth': 11000,
#         'crew_size': 7,
#         'contractor': 'GHI Drilling',
#         'lease_terms': '5 years',
#         'initial_cost': 4800000,
#         'environmental_sensitivity': 'Medium',
#         'water_source': 'Local Aquifer',
#         'proximity_to_protected_areas': 'Far',
#         'soil_type': 'Sandy Clay',
#         'emission_control_system': 'Advanced',
#         'spill_prevention_system': 'Tier 2',
#         'groundwater_monitoring': 'Daily'
#     },
#     'Well_Spiderman': {
#         'basin': 'Bakken',
#         'type': 'Horizontal',
#         'depth': 9500,
#         'crew_size': 6,
#         'contractor': 'JKL Services',
#         'lease_terms': '4 years',
#         'initial_cost': 4200000,
#         'environmental_sensitivity': 'Low',
#         'water_source': 'Municipal',
#         'proximity_to_protected_areas': 'Far',
#         'soil_type': 'Loam',
#         'emission_control_system': 'Standard',
#         'spill_prevention_system': 'Tier 1',
#         'groundwater_monitoring': 'Weekly'
#     }
# }
# # Continuing from the previous wells definition...
#
# # Maintenance items and their base costs
# maintenance_items = {
#     'Replacement_Parts': {
#         'ESP_System': {
#             'Pump': 15000,
#             'Motor': 12000,
#             'Cable': 5000,
#             'Sensor_Package': 3000,
#             'Controller': 4000,
#             'VFD': 8000
#         },
#         'Wellhead_Equipment': {
#             'Christmas_Tree': 20000,
#             'Valves': 5000,
#             'Pressure_Gauges': 1000,
#             'Safety_Devices': 3000,
#             'Flanges': 2000,
#             'Spools': 1500
#         },
#         'Downhole_Equipment': {
#             'Tubing': 10000,
#             'Packers': 8000,
#             'Safety_Valves': 6000,
#             'Gas_Lift_Valves': 4000,
#             'Production_Liner': 15000,
#             'Sand_Screens': 7000
#         },
#         'Surface_Equipment': {
#             'Separators': 25000,
#             'Heater_Treaters': 20000,
#             'Storage_Tanks': 30000,
#             'Compressors': 15000,
#             'Pumping_Units': 18000
#         }
#     },
#     'Services': {
#         'Regular_Maintenance': {
#             'Well_Logging': 5000,
#             'Pressure_Testing': 3000,
#             'Flow_Testing': 2500,
#             'Corrosion_Monitoring': 2000,
#             'Scale_Treatment': 1500
#         },
#         'Cleaning': {
#             'Tank_Cleaning': 4000,
#             'Pipeline_Pigging': 3500,
#             'Well_Bore_Cleaning': 5000,
#             'Surface_Equipment_Cleaning': 2000
#         },
#         'Inspection': {
#             'NDT_Inspection': 3000,
#             'Corrosion_Inspection': 2500,
#             'Safety_System_Check': 2000,
#             'Environmental_Compliance_Check': 2500
#         }
#     },
#     'Repairs': {
#         'Emergency_Repairs': {
#             'Leak_Repair': 10000,
#             'Equipment_Failure': 15000,
#             'Power_System': 8000,
#             'Control_System': 5000
#         },
#         'Scheduled_Repairs': {
#             'Worn_Parts': 5000,
#             'Calibration': 2000,
#             'Preventive_Replacement': 4000
#         }
#     }
# }
#
# environmental_categories = {
#     'Emissions_Monitoring': {
#         'Methane_Emissions': {
#             'Continuous_Monitoring': 1000,
#             'Leak_Detection': 800,
#             'Repair_Program': 1200,
#             'Reporting': 500
#         },
#         'CO2_Emissions': {
#             'Monitoring': 900,
#             'Reporting': 400,
#             'Offset_Programs': 2000
#         },
#         'VOC_Emissions': {
#             'Monitoring': 700,
#             'Control_Systems': 1500,
#             'Reporting': 300
#         }
#     },
#     'Water_Management': {
#         'Produced_Water': {
#             'Treatment': 2000,
#             'Disposal': 1500,
#             'Recycling': 2500,
#             'Quality_Testing': 800
#         },
#         'Groundwater': {
#             'Monitoring_Wells': 1200,
#             'Sample_Analysis': 600,
#             'Report_Generation': 400,
#             'Remediation': 5000
#         },
#         'Surface_Water': {
#             'Quality_Monitoring': 700,
#             'Runoff_Control': 900,
#             'Spill_Prevention': 1000
#         }
#     },
#     'Soil_Management': {
#         'Contamination_Monitoring': {
#             'Sampling': 500,
#             'Analysis': 800,
#             'Reporting': 300
#         },
#         'Remediation': {
#             'Soil_Treatment': 3000,
#             'Disposal': 2000,
#             'Site_Restoration': 5000
#         }
#     },
#     'Waste_Management': {
#         'Drilling_Waste': {
#             'Treatment': 1500,
#             'Disposal': 2000,
#             'Recycling': 1000
#         },
#         'Chemical_Waste': {
#             'Storage': 800,
#             'Treatment': 1200,
#             'Disposal': 1500
#         },
#         'General_Waste': {
#             'Collection': 300,
#             'Sorting': 200,
#             'Disposal': 500
#         }
#     },
#     'Wildlife_Protection': {
#         'Habitat_Preservation': {
#             'Monitoring': 1000,
#             'Protection_Measures': 1500,
#             'Restoration': 3000
#         },
#         'Species_Monitoring': {
#             'Surveys': 800,
#             'Reporting': 400,
#             'Mitigation': 2000
#         }
#     }
# }
#
# financial_categories = {
#     'Labor': {
#         'Operations': {
#             'Operators': 5000,
#             'Technicians': 4500,
#             'Engineers': 7000,
#             'Supervisors': 8000,
#             'Support_Staff': 3500
#         },
#         'Maintenance': {
#             'Maintenance_Crew': 4000,
#             'Specialists': 6000,
#             'Contractors': 5500
#         },
#         'Management': {
#             'Site_Manager': 10000,
#             'HSE_Manager': 8000,
#             'Technical_Manager': 9000
#         }
#     },
#     'Insurance': {
#         'Operational': {
#             'Equipment_Insurance': 2000,
#             'Business_Interruption': 3000,
#             'General_Liability': 2500
#         },
#         'Environmental': {
#             'Pollution_Liability': 4000,
#             'Environmental_Damage': 3500,
#             'Remediation_Cost': 3000
#         },
#         'Personnel': {
#             'Workers_Comp': 1500,
#             'Health_Insurance': 2000,
#             'Life_Insurance': 1000
#         }
#     },
#     'Lease': {
#         'Land': {
#             'Surface_Rights': 10000,
#             'Mineral_Rights': 15000,
#             'Access_Rights': 5000
#         },
#         'Equipment': {
#             'Heavy_Machinery': 8000,
#             'Vehicles': 3000,
#             'Temporary_Equipment': 2000
#         },
#         'Facilities': {
#             'Office_Space': 2000,
#             'Storage': 1500,
#             'Worker_Facilities': 1000
#         }
#     },
#     'Regulatory_Compliance': {
#         'Permits': {
#             'Drilling_Permits': 5000,
#             'Environmental_Permits': 4000,
#             'Operating_Permits': 3000
#         },
#         'Reporting': {
#             'Environmental_Reports': 2000,
#             'Production_Reports': 1500,
#             'Safety_Reports': 1000
#         },
#         'Audits': {
#             'Environmental_Audits': 3000,
#             'Safety_Audits': 2500,
#             'Financial_Audits': 4000
#         }
#     }
# }
#
# monte_carlo_params = {
#     'Production': {
#         'Oil_Rate': {
#             'min': 0.7,
#             'most_likely': 1.0,
#             'max': 1.3
#         },
#         'Gas_Rate': {
#             'min': 0.6,
#             'most_likely': 1.0,
#             'max': 1.4
#         },
#         'Water_Cut': {
#             'min': 0.8,
#             'most_likely': 1.0,
#             'max': 1.2
#         }
#     },
#     'Costs': {
#         'Operating_Cost': {
#             'min': 0.8,
#             'most_likely': 1.0,
#             'max': 1.3
#         },
#         'Maintenance_Cost': {
#             'min': 0.7,
#             'most_likely': 1.0,
#             'max': 1.5
#         }
#     },
#     'Prices': {
#         'Oil_Price': {
#             'min': 40,
#             'most_likely': 70,
#             'max': 100
#         },
#         'Gas_Price': {
#             'min': 2,
#             'most_likely': 3.5,
#             'max': 5
#         }
#     },
#     'Environmental': {
#         'Emission_Rates': {
#             'min': 0.8,
#             'most_likely': 1.0,
#             'max': 1.2
#         },
#         'Water_Treatment_Cost': {
#             'min': 0.9,
#             'most_likely': 1.0,
#             'max': 1.4
#         }
#     }
# }
#
# def run_monte_carlo_simulation(base_values, params, n_simulations=1000):
#     """Run Monte Carlo simulation using numpy's random functions instead of scipy"""
#     results = {}
#     for category, items in params.items():
#         category_results = {}
#         for item, ranges in items.items():
#             # Using numpy's triangular distribution
#             simulated_values = np.random.triangular(
#                 ranges['min'],
#                 ranges['most_likely'],
#                 ranges['max'],
#                 size=n_simulations
#             )
#             category_results[item] = simulated_values
#         results[category] = category_results
#     return results
#
# def generate_series(base_value, trend, noise_level, length, seasonality=False):
#     """Generate time series data with trend and optional seasonality"""
#     trend_series = np.linspace(0, trend, length)
#     noise = np.random.normal(0, noise_level, length)
#     if seasonality:
#         seasonal_component = np.sin(np.linspace(0, 4 * np.pi, length)) * base_value * 0.1
#         return np.maximum(0, base_value + trend_series + noise + seasonal_component)
#     return np.maximum(0, base_value + trend_series + noise)
#
# def generate_environmental_data(date, base_factor, well_info):
#     """Generate environmental monitoring data"""
#     env_data = {}
#     for category, subcategories in environmental_categories.items():
#         for subcat, items in subcategories.items():
#             for item, base_cost in items.items():
#                 # Add some randomness and seasonal variation
#                 seasonal_factor = 1 + 0.1 * np.sin(2 * np.pi * date.timetuple().tm_yday / 365)
#                 sensitivity_factor = 1.2 if well_info['environmental_sensitivity'] == 'High' else 1.0
#
#                 cost = base_cost * base_factor * seasonal_factor * sensitivity_factor
#                 env_data[f'{category}_{subcat}_{item}'] = round(cost * (1 + np.random.normal(0, 0.1)), 2)
#
#     return env_data
#
# def generate_maintenance_events(date, base_factor):
#     """Generate maintenance events and costs"""
#     maintenance_records = []
#
#     for category, subcategories in maintenance_items.items():
#         for subcat, items in subcategories.items():
#             for item, base_cost in items.items():
#                 # Randomize if maintenance occurred on this day
#                 if np.random.random() < 0.02:  # 2% chance each day
#                     cost = base_cost * base_factor * (1 + np.random.normal(0, 0.2))
#                     maintenance_records.append({
#                         'Maintenance_Category': category,
#                         'Subcategory': subcat,
#                         'Maintenance_Item': item,
#                         'Cost': round(cost, 2),
#                         'Status': np.random.choice(['Completed', 'In Progress', 'Scheduled']),
#                         'Priority': np.random.choice(['High', 'Medium', 'Low'])
#                     })
#
#     return maintenance_records
#
# def generate_financial_details(date, base_factor, well_info):
#     """Generate detailed financial records"""
#     financial_records = {}
#
#     for category, subcategories in financial_categories.items():
#         category_total = 0
#         for subcat, items in subcategories.items():
#             subcat_total = 0
#             for item, base_cost in items.items():
#                 # Adjust costs based on crew size and add some randomness
#                 if category == 'Labor':
#                     cost = base_cost * well_info['crew_size'] * base_factor * (1 + np.random.normal(0, 0.1))
#                 else:
#                     cost = base_cost * base_factor * (1 + np.random.normal(0, 0.15))
#
#                 financial_records[f'{category}_{subcat}_{item}'] = round(cost, 2)
#                 subcat_total += cost
#
#             financial_records[f'{category}_{subcat}_Total'] = round(subcat_total, 2)
#             category_total += subcat_total
#
#         financial_records[f'{category}_Total'] = round(category_total, 2)
#
#     return financial_records
#
# # Main data generation
# all_data = []
# maintenance_data = []
# financial_data = []
# environmental_data = []
# monte_carlo_results = {}
#
# print("Starting data generation...")
#
# for well_name, well_info in wells.items():
#     print(f"\nGenerating data for {well_name}...")
#
#     # Run Monte Carlo simulation for this well
#     monte_carlo_results[well_name] = run_monte_carlo_simulation(None, monte_carlo_params)
#
#     depth_factor = well_info['depth'] / 10000
#     type_factor = 1.2 if well_info['type'] == 'Horizontal' else 0.8
#     base = depth_factor * type_factor
#
#     for date in dates:
#         # Generate base record
#         season_factor = 1 + 0.1 * np.sin(2 * np.pi * date.timetuple().tm_yday / 365)
#
#         # Production data with Monte Carlo adjustments
#         mc_factor = np.random.choice(monte_carlo_results[well_name]['Production']['Oil_Rate'])
#         oil_production = generate_series(1000 * base * mc_factor, -100, 50, 1, seasonality=True)[0]
#
#         mc_factor = np.random.choice(monte_carlo_results[well_name]['Production']['Water_Cut'])
#         water_cut = generate_series(20 * base * mc_factor, 15, 5, 1)[0]
#
#         mc_factor = np.random.choice(monte_carlo_results[well_name]['Production']['Gas_Rate'])
#         gas_production = generate_series(500 * base * mc_factor, -50, 25, 1, seasonality=True)[0]
#
#         # Base record
#         base_record = {
#             'Date': date,
#             'Well_Name': well_name,
#             'Basin': well_info['basin'],
#             'Well_Type': well_info['type'],
#             'Well_Depth_Ft': well_info['depth'],
#             'Contractor': well_info['contractor'],
#             'Crew_Size': well_info['crew_size'],
#             'Lease_Terms': well_info['lease_terms'],
#             'Initial_Cost': well_info['initial_cost'],
#             'Environmental_Sensitivity': well_info['environmental_sensitivity'],
#
#             # Production
#             'Oil_Production_BBL': round(oil_production, 2),
#             'Water_Cut_Percentage': round(water_cut, 2),
#             'Gas_Production_MCF': round(gas_production, 2),
#
#             # Environmental base metrics
#             'Water_Source': well_info['water_source'],
#             'Proximity_To_Protected_Areas': well_info['proximity_to_protected_areas'],
#             'Soil_Type': well_info['soil_type'],
#             'Emission_Control_System': well_info['emission_control_system'],
#             'Spill_Prevention_System': well_info['spill_prevention_system'],
#             'Groundwater_Monitoring': well_info['groundwater_monitoring'],
#
#             # Production metrics
#             'Liquid_Rate_BBL_Day': round(oil_production * (1 + water_cut / 100), 2),
#             'Gas_Oil_Ratio': round(gas_production / oil_production if oil_production > 0 else 0, 2),
#             'Water_Oil_Ratio': round(water_cut / (100 - water_cut) if water_cut < 100 else 999, 2),
#
#             # Reservoir conditions
#             'Reservoir_Pressure_PSI': round(generate_series(2000 * base, -200, 50, 1)[0], 2),
#             'Bottom_Hole_Temperature_F': round(generate_series(180 * base, 0, 5, 1)[0], 2),
#             'Porosity_Percentage': round(generate_series(12 * base, -0.1, 0.2, 1)[0], 2),
#             'Permeability_mD': round(generate_series(50 * base, -1, 2, 1)[0], 2),
#
#             # Equipment performance
#             'Pump_Efficiency_Percentage': round(generate_series(95, -5, 2, 1)[0], 2),
#             'Runtime_Hours': round(generate_series(24, 0, 0.5, 1)[0], 2),
#             'Downtime_Hours': round(max(0, 24 - generate_series(24, 0, 0.5, 1)[0]), 2),
#             'Power_Usage_KWh': round(generate_series(100 * base, 2, 5, 1)[0], 2)
#         }
#
#         # Add maintenance records
#         maint_records = generate_maintenance_events(date, base)
#         for record in maint_records:
#             maintenance_record = base_record.copy()
#             maintenance_record.update(record)
#             maintenance_data.append(maintenance_record)
#
#         # Add environmental monitoring data
#         env_record = base_record.copy()
#         env_record.update(generate_environmental_data(date, base, well_info))
#         environmental_data.append(env_record)
#
#         # Add financial details
#         fin_record = base_record.copy()
#         fin_record.update(generate_financial_details(date, base, well_info))
#         financial_data.append(fin_record)
#
#         # Add to main records
#         all_data.append(base_record)
# # After creating DataFrames, modify the calculation part like this:
# print("Creating DataFrames...")
# df_main = pd.DataFrame(all_data)
# df_maintenance = pd.DataFrame(maintenance_data)
# df_financial = pd.DataFrame(financial_data)
# df_environmental = pd.DataFrame(environmental_data)
#
# # Create Monte Carlo DataFrame
# print("Creating Monte Carlo simulation DataFrame...")
# monte_carlo_df = pd.DataFrame()
# for well_name, results in monte_carlo_results.items():
#     well_mc_data = []
#     # Create rows for each simulation
#     for sim_idx in range(len(next(iter(results['Production'].values())))):
#         row = {'Well_Name': well_name, 'Simulation_ID': sim_idx}
#         # Add results for each category and metric
#         for category, items in results.items():
#             for item, values in items.items():
#                 row[f'{category}_{item}'] = values[sim_idx]
#         well_mc_data.append(row)
#     # Convert to DataFrame and append
#     well_mc_df = pd.DataFrame(well_mc_data)
#     monte_carlo_df = pd.concat([monte_carlo_df, well_mc_df], ignore_index=True)
#
# # Add Monte Carlo sheets to Excel file
# print("Adding Monte Carlo data to Excel...")
# try:
#     with pd.ExcelWriter('Well_Analysis_Data.xlsx', mode='a') as writer:
#         # Monte Carlo Summary sheet
#         monte_carlo_summary = monte_carlo_df.groupby('Well_Name').agg({
#             'Production_Oil_Rate': ['mean', 'std', 'min', 'max'],
#             'Production_Gas_Rate': ['mean', 'std', 'min', 'max'],
#             'Production_Water_Cut': ['mean', 'std', 'min', 'max'],
#             'Costs_Operating_Cost': ['mean', 'std', 'min', 'max'],
#             'Costs_Maintenance_Cost': ['mean', 'std', 'min', 'max']
#         }).round(2)
#         monte_carlo_summary.to_excel(writer, sheet_name='Monte_Carlo_Summary')
#
#         # Individual Monte Carlo sheets for each well
#         for well_name in wells.keys():
#             well_mc_data = monte_carlo_df[monte_carlo_df['Well_Name'] == well_name]
#             sheet_name = f"{well_name.replace('Well_', '')}_07_MonteCarlo"
#             well_mc_data.to_excel(writer, sheet_name=sheet_name, index=False)
# except Exception as e:
#     print(f"Error adding Monte Carlo data: {str(e)}")
#
#
# # Calculate rolling averages and trends with error handling
# print("Calculating metrics...")
# for well in wells.keys():
#     well_mask = df_main['Well_Name'] == well
#     well_data = df_main[well_mask].copy()  # Get a copy of the data for this well
#
#     if not well_data.empty:  # Check if we have data for this well
#         # Sort the data by date first
#         well_data = well_data.sort_values('Date')
#
#         # Production metrics
#         for col in ['Oil_Production_BBL', 'Gas_Production_MCF', 'Water_Cut_Percentage']:
#             # Calculate 30-day rolling average
#             rolling_avg = well_data[col].rolling(window=30, min_periods=1).mean()
#             df_main.loc[well_mask, f'{col}_30D_Avg'] = rolling_avg
#
#         # Production decline rate - calculate only if we have enough data
#         if len(well_data) >= 365:
#             decline_rate = well_data['Oil_Production_BBL'].pct_change(periods=365)
#             df_main.loc[well_mask, 'Production_Decline_Rate'] = decline_rate
#         else:
#             # If we don't have enough data, fill with NaN or 0
#             df_main.loc[well_mask, 'Production_Decline_Rate'] = np.nan
#
#         # Equipment performance metrics
#         for col in ['Pump_Efficiency_Percentage', 'Runtime_Hours']:
#             # Calculate 7-day rolling average
#             rolling_avg = well_data[col].rolling(window=7, min_periods=1).mean()
#             df_main.loc[well_mask, f'{col}_7D_Avg'] = rolling_avg
#
# # Verify data
# print("\nData Verification:")
# print(f"Total records in main data: {len(df_main)}")
# for well in wells.keys():
#     well_data = df_main[df_main['Well_Name'] == well]
#     print(f"{well}: {len(well_data)} records")
#
# # Save files with error handling
# print("\nSaving files...")
# try:
#     df_main.to_csv('well_production_data.csv', index=False)
#     print("- Saved well_production_data.csv")
#
#     df_maintenance.to_csv('well_maintenance_data.csv', index=False)
#     print("- Saved well_maintenance_data.csv")
#
#     df_financial.to_csv('well_financial_data.csv', index=False)
#     print("- Saved well_financial_data.csv")
#
#     df_environmental.to_csv('well_environmental_data.csv', index=False)
#     print("- Saved well_environmental_data.csv")
#
#     monte_carlo_df.to_csv('monte_carlo_simulations.csv', index=False)
#     print("- Saved monte_carlo_simulations.csv")
# except Exception as e:
#     print(f"Error saving files: {str(e)}")
#
# # After saving the CSVs, add this code to create the Excel workbook with sheets per well
# print("\nCreating Excel workbook with sheets per well...")
#
# try:
#     with pd.ExcelWriter('Well_Analysis_Data.xlsx', engine='openpyxl') as writer:
#         # Create Main Dashboard
#         wells_summary = df_main[['Well_Name', 'Basin', 'Well_Type', 'Well_Depth_Ft']].drop_duplicates()
#         wells_summary.to_excel(writer, sheet_name='Main_Dashboard', index=False)
#
#         # Create Date Range sheet
#         date_summary = df_main[['Date']].drop_duplicates().sort_values('Date')
#         date_summary['Year'] = pd.to_datetime(date_summary['Date']).dt.year
#         date_summary['Month'] = pd.to_datetime(date_summary['Date']).dt.month
#         date_summary['Quarter'] = pd.to_datetime(date_summary['Date']).dt.quarter
#         date_summary.to_excel(writer, sheet_name='00_Date_Range', index=False)
#
#         # Process each well
#         for well_name in wells.keys():
#             print(f"\nProcessing {well_name}...")
#             well_data = df_main[df_main['Well_Name'] == well_name].copy()
#             prefix = well_name.replace('Well_', '')
#
#             # Production Data
#             production_cols = [
#                 'Date', 'Oil_Production_BBL', 'Oil_Production_BBL_30D_Avg',
#                 'Gas_Production_MCF', 'Gas_Production_MCF_30D_Avg',
#                 'Water_Cut_Percentage', 'Production_Decline_Rate',
#                 'Liquid_Rate_BBL_Day', 'Gas_Oil_Ratio', 'Water_Oil_Ratio'
#             ]
#             well_data[production_cols].to_excel(
#                 writer,
#                 sheet_name=f'{prefix}_01_Production',
#                 index=False
#             )
#
#             # Reservoir Data
#             reservoir_cols = [
#                 'Date', 'Reservoir_Pressure_PSI', 'Bottom_Hole_Temperature_F',
#                 'Porosity_Percentage', 'Permeability_mD'
#             ]
#             well_data[reservoir_cols].to_excel(
#                 writer,
#                 sheet_name=f'{prefix}_02_Reservoir',
#                 index=False
#             )
#
#             # Equipment Data
#             equipment_cols = [
#                 'Date', 'Pump_Efficiency_Percentage', 'Runtime_Hours',
#                 'Downtime_Hours', 'Power_Usage_KWh'
#             ]
#             well_data[reservoir_cols].to_excel(
#                 writer,
#                 sheet_name=f'{prefix}_03_Equipment',
#                 index=False
#             )
#
#             # Get maintenance data for this well
#             well_maintenance = df_maintenance[df_maintenance['Well_Name'] == well_name].copy()
#             well_maintenance.to_excel(
#                 writer,
#                 sheet_name=f'{prefix}_04_Maintenance',
#                 index=False
#             )
#
#             # Get environmental data for this well
#             well_environmental = df_environmental[df_environmental['Well_Name'] == well_name].copy()
#             # Select relevant columns for environmental data
#             env_cols = [col for col in well_environmental.columns if any(
#                 keyword in col for keyword in ['Emissions', 'Water', 'Soil', 'Waste', 'Wildlife']
#             )]
#             well_environmental[['Date'] + env_cols].to_excel(
#                 writer,
#                 sheet_name=f'{prefix}_05_Environmental',
#                 index=False
#             )
#
#             # Get financial data for this well
#             well_financial = df_financial[df_financial['Well_Name'] == well_name].copy()
#             # Select relevant columns for financial data
#             fin_cols = [col for col in well_financial.columns if any(
#                 keyword in col for keyword in ['Cost', 'Revenue', 'Price', 'Income', 'Lease', 'Insurance']
#             )]
#             well_financial[['Date'] + fin_cols].to_excel(
#                 writer,
#                 sheet_name=f'{prefix}_06_Financial',
#                 index=False
#             )
#
#             # Create summary statistics
#             summary_stats = pd.DataFrame([{
#                 'Metric': 'Total Oil Production (BBL)',
#                 'Value': well_data['Oil_Production_BBL'].sum(),
#                 'Unit': 'BBL'
#             }, {
#                 'Metric': 'Average Daily Oil Production',
#                 'Value': well_data['Oil_Production_BBL'].mean(),
#                 'Unit': 'BBL/day'
#             }, {
#                 'Metric': 'Total Gas Production (MCF)',
#                 'Value': well_data['Gas_Production_MCF'].sum(),
#                 'Unit': 'MCF'
#             }, {
#                 'Metric': 'Average Water Cut',
#                 'Value': well_data['Water_Cut_Percentage'].mean(),
#                 'Unit': '%'
#             }, {
#                 'Metric': 'Average Pump Efficiency',
#                 'Value': well_data['Pump_Efficiency_Percentage'].mean(),
#                 'Unit': '%'
#             }, {
#                 'Metric': 'Total Downtime Hours',
#                 'Value': well_data['Downtime_Hours'].sum(),
#                 'Unit': 'Hours'
#             }])
#
#             summary_stats.to_excel(
#                 writer,
#                 sheet_name=f'{prefix}_00_Summary',
#                 index=False
#             )
#
#         print("\nExcel file created successfully!")
#
#         # Print summary of created sheets
#         print("\nWorkbook Structure:")
#         print("1. Main_Dashboard (Overview of all wells)")
#         print("2. 00_Date_Range (Time period covered)")
#         for well_name in wells.keys():
#             prefix = well_name.replace('Well_', '')
#             print(f"\n3. {well_name} Sheets:")
#             print(f"   - {prefix}_00_Summary (Key metrics)")
#             print(f"   - {prefix}_01_Production (Production data)")
#             print(f"   - {prefix}_02_Reservoir (Reservoir data)")
#             print(f"   - {prefix}_03_Equipment (Equipment status)")
#             print(f"   - {prefix}_04_Maintenance (Maintenance records)")
#             print(f"   - {prefix}_05_Environmental (Environmental data)")
#             print(f"   - {prefix}_06_Financial (Financial data)")
#
# except Exception as e:
#     print(f"Error creating Excel file: {str(e)}")
#     print("\nDetailed error information:")
#     import traceback
#
#     traceback.print_exc()
# # Print final statistics
# print("\nFinal Data Summary:")
# print(f"Total number of days: {len(dates)}")
# print(f"Number of wells: {len(wells)}")
# print(f"Records in main data: {len(df_main):,}")
# print(f"Maintenance events: {len(df_maintenance):,}")
# print(f"Environmental records: {len(df_environmental):,}")
# print(f"Financial records: {len(df_financial):,}")
#
# # Check for missing values
# print("\nMissing Values Check:")
# for col in df_main.columns:
#     missing = df_main[col].isna().sum()
#     if missing > 0:
#         print(f"- {col}: {missing} missing values")