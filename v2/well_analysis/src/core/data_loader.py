import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from datetime import datetime
import logging
from .constants import (WELL_CONFIGS, DATA_FILES)
from .data_structures import WellData, SimulationParameters

logger = logging.getLogger(__name__)


class DataLoader:
    """Data loader class for handling all data I/O operations"""

    def __init__(self, data_dir: Union[str, Path]):
        self.data_dir = Path(data_dir)
        self._setup_logging()

    def _setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

    def load_excel_data(self, file_path: Path, sheet_name: str) -> Optional[pd.DataFrame]:
        """Load data from Excel file with error handling"""
        try:
            return pd.read_excel(file_path, sheet_name=sheet_name)
        except Exception as e:
            logger.error(f"Error loading {file_path} - {sheet_name}: {str(e)}")
            return None

    def save_excel_data(self, df: pd.DataFrame, file_path: Path,
                        sheet_name: str, mode: str = 'w') -> bool:
        """Save data to Excel file with error handling"""
        try:
            with pd.ExcelWriter(file_path, engine='openpyxl', mode=mode) as writer:
                df.to_excel(writer, sheet_name=sheet_name, index=False)
            return True
        except Exception as e:
            logger.error(f"Error saving to {file_path} - {sheet_name}: {str(e)}")
            return False

    def load_well_data(self, well_name: str) -> Optional[WellData]:
        """Load all data for a specific well"""
        if well_name not in WELL_CONFIGS:
            logger.error(f"Well {well_name} not found in configurations")
            return None

        well_config = WELL_CONFIGS[well_name]
        well_data = WellData(well_name=well_name, **well_config)

        # Load each data type
        data_types = {
            'production': 'production_data',
            'maintenance': 'maintenance_data',
            'environmental': 'environmental_data',
            'financial': 'financial_data',
            'monte_carlo': 'monte_carlo_data'
        }

        for data_type, attr_name in data_types.items():
            file_path = self.data_dir / DATA_FILES[data_type]
            df = self.load_excel_data(file_path, well_name)
            if df is not None:
                setattr(well_data, attr_name, df)

        return well_data

    def load_all_wells(self) -> Dict[str, WellData]:
        """Load data for all wells"""
        wells_data = {}
        for well_name in WELL_CONFIGS.keys():
            well_data = self.load_well_data(well_name)
            if well_data is not None:
                wells_data[well_name] = well_data
        return wells_data

    def save_well_data(self, well_data: WellData, backup: bool = True) -> bool:
        """Save all data for a specific well"""
        success = True

        # Create backup if requested
        if backup:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_dir = self.data_dir / "backup" / timestamp
            backup_dir.mkdir(parents=True, exist_ok=True)

        data_types = {
            'production': well_data.production_data,
            'maintenance': well_data.maintenance_data,
            'environmental': well_data.environmental_data,
            'financial': well_data.financial_data,
            'monte_carlo': well_data.monte_carlo_data
        }

        for data_type, df in data_types.items():
            if df is not None:
                file_path = self.data_dir / DATA_FILES[data_type]

                # Create backup
                if backup:
                    backup_path = backup_dir / DATA_FILES[data_type]
                    if file_path.exists():
                        import shutil
                        shutil.copy2(file_path, backup_path)

                # Save current data
                mode = 'a' if file_path.exists() else 'w'
                if not self.save_excel_data(df, file_path, well_data.well_name, mode):
                    success = False

        return success

    def validate_data(self, well_data: WellData) -> bool:
        """Validate well data structure and contents"""
        required_columns = {
            'production_data': [
                'Date', 'Oil_Production_BBL', 'Gas_Production_MCF',
                'Water_Cut_Percentage', 'Liquid_Rate_BBL_Day'
            ],
            'maintenance_data': [
                'Date', 'Maintenance_Category', 'Maintenance_Item',
                'Cost', 'Status', 'Priority'
            ],
            'environmental_data': [
                'Date'  # Add specific environmental columns
            ],
            'financial_data': [
                'Date'  # Add specific financial columns
            ]
        }

        is_valid = True

        # Check data types existence
        for data_type, columns in required_columns.items():
            df = getattr(well_data, data_type)
            if df is None:
                logger.warning(f"Missing {data_type} for well {well_data.well_name}")
                is_valid = False
                continue

            # Check required columns
            missing_cols = [col for col in columns if col not in df.columns]
            if missing_cols:
                logger.warning(f"Missing columns in {data_type}: {missing_cols}")
                is_valid = False

            # Check data types
            if 'Date' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['Date']):
                logger.warning(f"Date column in {data_type} is not datetime type")
                is_valid = False

            # Check for null values
            null_counts = df.isnull().sum()
            if null_counts.any():
                logger.warning(f"Null values found in {data_type}:\n{null_counts[null_counts > 0]}")
                is_valid = False

        return is_valid

    def get_data_summary(self, well_data: WellData) -> Dict[str, Any]:
        """Generate summary statistics for well data"""
        summary = {
            'well_name': well_data.well_name,
            'basin': well_data.basin,
            'type': well_data.type,
            'depth': well_data.depth
        }

        # Production summary
        if well_data.production_data is not None:
            prod_data = well_data.production_data
            summary.update({
                'total_oil_production': prod_data['Oil_Production_BBL'].sum(),
                'avg_daily_oil': prod_data['Oil_Production_BBL'].mean(),
                'total_gas_production': prod_data['Gas_Production_MCF'].sum(),
                'avg_water_cut': prod_data['Water_Cut_Percentage'].mean()
            })

        # Maintenance summary
        if well_data.maintenance_data is not None:
            maint_data = well_data.maintenance_data
            summary.update({
                'total_maintenance_cost': maint_data['Cost'].sum(),
                'maintenance_events': len(maint_data),
                'avg_maintenance_cost': maint_data['Cost'].mean()
            })

        # Financial summary
        if well_data.financial_data is not None:
            fin_cols = [col for col in well_data.financial_data.columns if 'Cost' in col or 'Total' in col]
            for col in fin_cols:
                summary[f'total_{col.lower()}'] = well_data.financial_data[col].sum()

        return summary

    def export_summary(self, wells_data: Dict[str, WellData],
                       output_file: Path) -> None:
        """Export summary data for all wells"""
        summaries = []
        for well_data in wells_data.values():
            summary = self.get_data_summary(well_data)
            summaries.append(summary)

        summary_df = pd.DataFrame(summaries)
        self.save_excel_data(summary_df, output_file, 'Summary')