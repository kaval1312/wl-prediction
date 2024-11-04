import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from .constants import (WELL_CONFIGS, MONTE_CARLO_PARAMS, MAINTENANCE_ITEMS,
                        ENVIRONMENTAL_CATEGORIES, FINANCIAL_CATEGORIES)
from .data_structures import WellData, SimulationParameters


class DataGenerator:
    def __init__(self, start_date: datetime = datetime(2019, 1, 1),
                 num_years: int = 5):
        self.start_date = start_date
        self.num_years = num_years
        self.dates = [start_date + timedelta(days=x)
                      for x in range(365 * num_years)]
        self.simulation_params = SimulationParameters()

    def generate_series(self, base_value: float, trend: float,
                        noise_level: float, length: int,
                        seasonality: bool = False) -> np.ndarray:
        """Generate time series data with trend and optional seasonality"""
        trend_series = np.linspace(0, trend, length)
        noise = np.random.normal(0, noise_level, length)
        if seasonality:
            seasonal_component = np.sin(np.linspace(0, 4 * np.pi, length)) * base_value * 0.1
            return np.maximum(0, base_value + trend_series + noise + seasonal_component)
        return np.maximum(0, base_value + trend_series + noise)

    def generate_well_data(self, well_name: str) -> WellData:
        """Generate complete well data"""
        if well_name not in WELL_CONFIGS:
            raise ValueError(f"Well {well_name} not found in configurations")

        well_config = WELL_CONFIGS[well_name]
        well_data = WellData(well_name=well_name, **well_config)

        # Generate all data
        well_data.production_data = self.generate_production_data(well_config)
        well_data.maintenance_data = self.generate_maintenance_data(well_config)
        well_data.environmental_data = self.generate_environmental_data(well_config)
        well_data.financial_data = self.generate_financial_data(well_config)
        well_data.monte_carlo_data = self.generate_monte_carlo_data(well_config)

        return well_data

    def generate_production_data(self, well_config: Dict) -> pd.DataFrame:
        """Generate production data"""
        all_data = []
        depth_factor = well_config['depth'] / 10000
        type_factor = 1.2 if well_config['type'] == 'Horizontal' else 0.8
        base = depth_factor * type_factor

        for date in self.dates:
            season_factor = 1 + 0.1 * np.sin(2 * np.pi * date.timetuple().tm_yday / 365)

            # Production metrics
            oil_production = self.generate_series(1000 * base, -100, 50, 1, True)[0]
            water_cut = self.generate_series(20 * base, 15, 5, 1)[0]
            gas_production = self.generate_series(500 * base, -50, 25, 1, True)[0]

            record = {
                'Date': date,
                'Oil_Production_BBL': round(oil_production, 2),
                'Water_Cut_Percentage': round(water_cut, 2),
                'Gas_Production_MCF': round(gas_production, 2),
                'Liquid_Rate_BBL_Day': round(oil_production * (1 + water_cut / 100), 2),
                'Gas_Oil_Ratio': round(gas_production / oil_production if oil_production > 0 else 0, 2),
                'Water_Oil_Ratio': round(water_cut / (100 - water_cut) if water_cut < 100 else 999, 2),
                'Reservoir_Pressure_PSI': round(self.generate_series(2000 * base, -200, 50, 1)[0], 2),
                'Bottom_Hole_Temperature_F': round(self.generate_series(180 * base, 0, 5, 1)[0], 2),
                'Porosity_Percentage': round(self.generate_series(12 * base, -0.1, 0.2, 1)[0], 2),
                'Permeability_mD': round(self.generate_series(50 * base, -1, 2, 1)[0], 2),
                'Pump_Efficiency_Percentage': round(self.generate_series(95, -5, 2, 1)[0], 2),
                'Runtime_Hours': round(self.generate_series(24, 0, 0.5, 1)[0], 2),
                'Downtime_Hours': round(max(0, 24 - self.generate_series(24, 0, 0.5, 1)[0]), 2),
                'Power_Usage_KWh': round(self.generate_series(100 * base, 2, 5, 1)[0], 2)
            }
            all_data.append(record)

        return pd.DataFrame(all_data)

    def generate_maintenance_data(self, well_config: Dict) -> pd.DataFrame:
        """Generate maintenance data"""
        maintenance_records = []
        depth_factor = well_config['depth'] / 10000
        type_factor = 1.2 if well_config['type'] == 'Horizontal' else 0.8
        base = depth_factor * type_factor

        for date in self.dates:
            # Check each maintenance item
            for category, subcategories in MAINTENANCE_ITEMS.items():
                for subcat, items in subcategories.items():
                    for item, base_cost in items.items():
                        # 2% chance of maintenance event each day
                        if np.random.random() < 0.02:
                            cost = base_cost * base * (1 + np.random.normal(0, 0.2))
                            record = {
                                'Date': date,
                                'Maintenance_Category': category,
                                'Subcategory': subcat,
                                'Maintenance_Item': item,
                                'Cost': round(cost, 2),
                                'Status': np.random.choice(['Completed', 'In Progress', 'Scheduled']),
                                'Priority': np.random.choice(['High', 'Medium', 'Low'])
                            }
                            maintenance_records.append(record)

        return pd.DataFrame(maintenance_records)

    def generate_environmental_data(self, well_config: Dict) -> pd.DataFrame:
        """Generate environmental data"""
        environmental_records = []
        base_factor = 1.2 if well_config['environmental_sensitivity'] == 'High' else 1.0

        for date in self.dates:
            record = {'Date': date}

            # Generate data for each environmental category
            for category, subcategories in ENVIRONMENTAL_CATEGORIES.items():
                for subcat, items in subcategories.items():
                    for item, base_cost in items.items():
                        seasonal_factor = 1 + 0.1 * np.sin(2 * np.pi * date.timetuple().tm_yday / 365)
                        cost = base_cost * base_factor * seasonal_factor
                        record[f"{category}_{subcat}_{item}"] = round(cost * (1 + np.random.normal(0, 0.1)), 2)

            environmental_records.append(record)

        return pd.DataFrame(environmental_records)

    def generate_financial_data(self, well_config: Dict) -> pd.DataFrame:
        """Generate financial data"""
        financial_records = []
        depth_factor = well_config['depth'] / 10000
        type_factor = 1.2 if well_config['type'] == 'Horizontal' else 0.8
        base = depth_factor * type_factor

        for date in self.dates:
            record = {'Date': date}

            # Generate data for each financial category
            for category, subcategories in FINANCIAL_CATEGORIES.items():
                category_total = 0
                for subcat, items in subcategories.items():
                    subcat_total = 0
                    for item, base_cost in items.items():
                        if category == 'Labor':
                            cost = base_cost * well_config['crew_size'] * base * (1 + np.random.normal(0, 0.1))
                        else:
                            cost = base_cost * base * (1 + np.random.normal(0, 0.15))

                        record[f"{category}_{subcat}_{item}"] = round(cost, 2)
                        subcat_total += cost

                    record[f"{category}_{subcat}_Total"] = round(subcat_total, 2)
                    category_total += subcat_total

                record[f"{category}_Total"] = round(category_total, 2)

            financial_records.append(record)

        return pd.DataFrame(financial_records)

    def generate_monte_carlo_data(self, well_config: Dict) -> pd.DataFrame:
        """Generate Monte Carlo simulation data"""
        results = {}
        n_simulations = 1000

        for category, items in MONTE_CARLO_PARAMS.items():
            category_results = {}
            for item, ranges in items.items():
                simulated_values = np.random.triangular(
                    ranges['min'],
                    ranges['most_likely'],
                    ranges['max'],
                    size=n_simulations
                )
                category_results[f"{category}_{item}"] = simulated_values

            results.update(category_results)

        return pd.DataFrame(results)

    def calculate_decline_curve(self, initial_rate: float, decline_rate: float,
                                b_factor: float, time_periods: int) -> np.ndarray:
        """Calculate hyperbolic decline curve"""
        time = np.arange(time_periods)
        return initial_rate / (1 + b_factor * decline_rate * time) ** (1 / b_factor)

    def generate_decline_curves(self, well_config: Dict) -> pd.DataFrame:
        """Generate decline curves for oil and gas production"""
        initial_oil = 1000 * well_config['depth'] / 10000
        initial_gas = 500 * well_config['depth'] / 10000

        decline_data = []
        for date in self.dates:
            days_from_start = (date - self.start_date).days

            # Oil decline curve with hyperbolic decline
            oil_rate = self.calculate_decline_curve(
                initial_rate=initial_oil,
                decline_rate=0.1,
                b_factor=0.8,
                time_periods=days_from_start + 1
            )[-1]

            # Gas decline curve with different parameters
            gas_rate = self.calculate_decline_curve(
                initial_rate=initial_gas,
                decline_rate=0.15,
                b_factor=0.9,
                time_periods=days_from_start + 1
            )[-1]

            record = {
                'Date': date,
                'Oil_Decline_Rate': round(oil_rate, 2),
                'Gas_Decline_Rate': round(gas_rate, 2)
            }
            decline_data.append(record)

        return pd.DataFrame(decline_data)

    def generate_all_well_data(self) -> Dict[str, WellData]:
        """Generate data for all wells"""
        all_well_data = {}
        for well_name in WELL_CONFIGS.keys():
            try:
                well_data = self.generate_well_data(well_name)
                all_well_data[well_name] = well_data
            except Exception as e:
                print(f"Error generating data for {well_name}: {str(e)}")
        return all_well_data

    def save_all_data(self, data_dir: str, all_well_data: Dict[str, WellData]) -> None:
        """Save all generated data to Excel files"""
        file_writers = {}
        try:
            # Create Excel writers for each data type
            for data_type in ['production', 'maintenance', 'environmental', 'financial', 'monte_carlo']:
                file_writers[data_type] = pd.ExcelWriter(
                    f"{data_dir}/{data_type}_data.xlsx",
                    engine='openpyxl'
                )

            # Write data for each well
            for well_name, well_data in all_well_data.items():
                # Production data
                if well_data.production_data is not None:
                    well_data.production_data.to_excel(
                        file_writers['production'],
                        sheet_name=well_name,
                        index=False
                    )

                # Maintenance data
                if well_data.maintenance_data is not None:
                    well_data.maintenance_data.to_excel(
                        file_writers['maintenance'],
                        sheet_name=well_name,
                        index=False
                    )

                # Environmental data
                if well_data.environmental_data is not None:
                    well_data.environmental_data.to_excel(
                        file_writers['environmental'],
                        sheet_name=well_name,
                        index=False
                    )

                # Financial data
                if well_data.financial_data is not None:
                    well_data.financial_data.to_excel(
                        file_writers['financial'],
                        sheet_name=well_name,
                        index=False
                    )

                # Monte Carlo data
                if well_data.monte_carlo_data is not None:
                    well_data.monte_carlo_data.to_excel(
                        file_writers['monte_carlo'],
                        sheet_name=well_name,
                        index=False
                    )

            # Save all files
            for writer in file_writers.values():
                writer.close()

        except Exception as e:
            print(f"Error saving data: {str(e)}")
            # Ensure all writers are closed even if an error occurs
            for writer in file_writers.values():
                try:
                    writer.close()
                except:
                    pass

    def add_summary_sheets(self, data_dir: str) -> None:
        """Add summary sheets to all data files"""
        summary_df = pd.DataFrame({
            'Well_Name': list(WELL_CONFIGS.keys()),
            'Basin': [WELL_CONFIGS[w]['basin'] for w in WELL_CONFIGS.keys()],
            'Type': [WELL_CONFIGS[w]['type'] for w in WELL_CONFIGS.keys()],
            'Depth_Ft': [WELL_CONFIGS[w]['depth'] for w in WELL_CONFIGS.keys()],
            'Environmental_Sensitivity': [WELL_CONFIGS[w]['environmental_sensitivity']
                                          for w in WELL_CONFIGS.keys()]
        })

        for data_type in ['production', 'maintenance', 'environmental', 'financial', 'monte_carlo']:
            try:
                with pd.ExcelWriter(
                        f"{data_dir}/{data_type}_data.xlsx",
                        engine='openpyxl',
                        mode='a'
                ) as writer:
                    summary_df.to_excel(writer, sheet_name='Summary', index=False)
            except Exception as e:
                print(f"Error adding summary sheet to {data_type}_data.xlsx: {str(e)}")