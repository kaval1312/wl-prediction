from typing import Dict, Any
from pathlib import Path

# File paths
ROOT_DIR = Path(__file__).parent.parent.parent
CONFIG_DIR = ROOT_DIR / "config"
DATA_DIR = ROOT_DIR / "tests" / "test_data"

# Well configurations
WELL_CONFIGS: Dict[str, Dict[str, Any]] = {
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

# Monte Carlo simulation parameters
MONTE_CARLO_PARAMS = {
    'Production': {
        'Oil_Rate': {'min': 0.7, 'most_likely': 1.0, 'max': 1.3},
        'Gas_Rate': {'min': 0.6, 'most_likely': 1.0, 'max': 1.4},
        'Water_Cut': {'min': 0.8, 'most_likely': 1.0, 'max': 1.2}
    },
    'Costs': {
        'Operating_Cost': {'min': 0.8, 'most_likely': 1.0, 'max': 1.3},
        'Maintenance_Cost': {'min': 0.7, 'most_likely': 1.0, 'max': 1.5}
    },
    'Prices': {
        'Oil_Price': {'min': 40, 'most_likely': 70, 'max': 100},
        'Gas_Price': {'min': 2, 'most_likely': 3.5, 'max': 5}
    },
    'Environmental': {
        'Emission_Rates': {'min': 0.8, 'most_likely': 1.0, 'max': 1.2},
        'Water_Treatment_Cost': {'min': 0.9, 'most_likely': 1.0, 'max': 1.4}
    }
}

# Maintenance items and their base costs
MAINTENANCE_ITEMS = {
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

# Environmental categories and costs
ENVIRONMENTAL_CATEGORIES = {
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

# Financial categories
FINANCIAL_CATEGORIES = {
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

# File names
DATA_FILES = {
    'production': 'production_data.xlsx',
    'maintenance': 'maintenance_data.xlsx',
    'environmental': 'environmental_data.xlsx',
    'financial': 'financial_data.xlsx',
    'monte_carlo': 'monte_carlo_data.xlsx'
}