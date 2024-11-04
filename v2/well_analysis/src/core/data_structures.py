from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union, Any
from datetime import datetime
import pandas as pd
import numpy as np
from pathlib import Path
from .constants import (WELL_CONFIGS, MONTE_CARLO_PARAMS, MAINTENANCE_ITEMS,
                        ENVIRONMENTAL_CATEGORIES, FINANCIAL_CATEGORIES, DATA_FILES)


@dataclass
class WellData:
    well_name: str
    basin: str
    type: str
    depth: float
    crew_size: int
    contractor: str
    lease_terms: str
    initial_cost: float
    environmental_sensitivity: str
    water_source: str
    proximity_to_protected_areas: str
    soil_type: str
    emission_control_system: str
    spill_prevention_system: str
    groundwater_monitoring: str
    production_data: Optional[pd.DataFrame] = None
    maintenance_data: Optional[pd.DataFrame] = None
    environmental_data: Optional[pd.DataFrame] = None
    financial_data: Optional[pd.DataFrame] = None
    monte_carlo_data: Optional[pd.DataFrame] = None


@dataclass
class SimulationParameters:
    monte_carlo_params: Dict = field(default_factory=lambda: MONTE_CARLO_PARAMS)
    maintenance_items: Dict = field(default_factory=lambda: MAINTENANCE_ITEMS)
    environmental_categories: Dict = field(default_factory=lambda: ENVIRONMENTAL_CATEGORIES)
    financial_categories: Dict = field(default_factory=lambda: FINANCIAL_CATEGORIES)


class DataLoader:
    def __init__(self, data_dir: Union[str, Path]):
        self.data_dir = Path(data_dir)
        self.wells_data: Dict[str, WellData] = {}
        self.simulation_params = SimulationParameters()

    def load_all_data(self) -> None:
        """Load all data for all wells"""
        for well_name, well_config in WELL_CONFIGS.items():
            well_data = WellData(
                well_name=well_name,
                **well_config
            )

            # Load production data
            try:
                prod_data = pd.read_excel(
                    self.data_dir / DATA_FILES['production'],
                    sheet_name=well_name
                )
                well_data.production_data = prod_data
            except Exception as e:
                print(f"Error loading production data for {well_name}: {str(e)}")

            # Load maintenance data
            try:
                maint_data = pd.read_excel(
                    self.data_dir / DATA_FILES['maintenance'],
                    sheet_name=well_name
                )
                well_data.maintenance_data = maint_data
            except Exception as e:
                print(f"Error loading maintenance data for {well_name}: {str(e)}")

            # Load environmental data
            try:
                env_data = pd.read_excel(
                    self.data_dir / DATA_FILES['environmental'],
                    sheet_name=well_name
                )
                well_data.environmental_data = env_data
            except Exception as e:
                print(f"Error loading environmental data for {well_name}: {str(e)}")

            # Load financial data
            try:
                fin_data = pd.read_excel(
                    self.data_dir / DATA_FILES['financial'],
                    sheet_name=well_name
                )
                well_data.financial_data = fin_data
            except Exception as e:
                print(f"Error loading financial data for {well_name}: {str(e)}")

            # Load Monte Carlo data
            try:
                mc_data = pd.read_excel(
                    self.data_dir / DATA_FILES['monte_carlo'],
                    sheet_name=well_name
                )
                well_data.monte_carlo_data = mc_data
            except Exception as e:
                print(f"Error loading Monte Carlo data for {well_name}: {str(e)}")

            self.wells_data[well_name] = well_data

    def load_well_data(self, well_name: str) -> Optional[WellData]:
        """Load data for a specific well"""
        if well_name not in WELL_CONFIGS:
            print(f"Well {well_name} not found in configurations")
            return None

        well_config = WELL_CONFIGS[well_name]
        well_data = WellData(
            well_name=well_name,
            **well_config
        )

        # Load all data types for the specific well
        data_types = {
            'production': 'production_data',
            'maintenance': 'maintenance_data',
            'environmental': 'environmental_data',
            'financial': 'financial_data',
            'monte_carlo': 'monte_carlo_data'
        }

        for data_type, attr_name in data_types.items():
            try:
                data = pd.read_excel(
                    self.data_dir / DATA_FILES[data_type],
                    sheet_name=well_name
                )
                setattr(well_data, attr_name, data)
            except Exception as e:
                print(f"Error loading {data_type} data for {well_name}: {str(e)}")

        return well_data

    def save_well_data(self, well_data: WellData) -> None:
        """Save data for a specific well"""
        data_types = {
            'production': well_data.production_data,
            'maintenance': well_data.maintenance_data,
            'environmental': well_data.environmental_data,
            'financial': well_data.financial_data,
            'monte_carlo': well_data.monte_carlo_data
        }

        for data_type, data in data_types.items():
            if data is not None:
                try:
                    with pd.ExcelWriter(
                            self.data_dir / DATA_FILES[data_type],
                            engine='openpyxl',
                            mode='a' if (self.data_dir / DATA_FILES[data_type]).exists() else 'w'
                    ) as writer:
                        data.to_excel(writer, sheet_name=well_data.well_name, index=False)
                except Exception as e:
                    print(f"Error saving {data_type} data for {well_data.well_name}: {str(e)}")

    def get_simulation_parameters(self) -> SimulationParameters:
        """Get simulation parameters"""
        return self.simulation_params

    def update_simulation_parameters(self, new_params: SimulationParameters) -> None:
        """Update simulation parameters"""
        self.simulation_params = new_params

    def validate_data(self, well_data: WellData) -> bool:
        """Validate well data"""
        required_columns = {
            'production_data': ['Date', 'Oil_Production_BBL', 'Gas_Production_MCF', 'Water_Cut_Percentage'],
            'maintenance_data': ['Date', 'Maintenance_Category', 'Maintenance_Item', 'Cost'],
            'environmental_data': ['Date'],  # Add specific environmental columns
            'financial_data': ['Date']  # Add specific financial columns
        }

        for data_type, columns in required_columns.items():
            df = getattr(well_data, data_type)
            if df is not None:
                missing_cols = [col for col in columns if col not in df.columns]
                if missing_cols:
                    print(f"Missing columns in {data_type}: {missing_cols}")
                    return False

        return True