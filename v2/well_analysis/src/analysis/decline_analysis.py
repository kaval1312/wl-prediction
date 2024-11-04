import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from scipy.optimize import curve_fit
import logging
from ..core.data_structures import WellData

logger = logging.getLogger(__name__)


class DeclineAnalyzer:
    def __init__(self):
        """Initialize decline curve analysis"""
        self.results = {}

    @staticmethod
    def exponential_decline(t: np.ndarray, q_i: float, d: float) -> np.ndarray:
        """Exponential decline curve"""
        return q_i * np.exp(-d * t)

    @staticmethod
    def hyperbolic_decline(t: np.ndarray, q_i: float, d_i: float, b: float) -> np.ndarray:
        """Hyperbolic decline curve"""
        return q_i / (1 + b * d_i * t) ** (1 / b)

    @staticmethod
    def harmonic_decline(t: np.ndarray, q_i: float, d_i: float) -> np.ndarray:
        """Harmonic decline curve (special case of hyperbolic where b=1)"""
        return q_i / (1 + d_i * t)

    def fit_decline_curve(self, time: np.ndarray, rate: np.ndarray,
                          decline_type: str = 'hyperbolic') -> Tuple[np.ndarray, Dict]:
        """Fit decline curve to production data"""
        try:
            # Remove any zero or negative rates
            mask = rate > 0
            time_clean = time[mask]
            rate_clean = rate[mask]

            if decline_type == 'exponential':
                popt, pcov = curve_fit(
                    self.exponential_decline,
                    time_clean,
                    rate_clean,
                    p0=[rate_clean[0], 0.1],
                    bounds=([0, 0], [np.inf, 1])
                )
                fitted_curve = self.exponential_decline(time, *popt)
                params = {'q_i': popt[0], 'd': popt[1]}

            elif decline_type == 'hyperbolic':
                popt, pcov = curve_fit(
                    self.hyperbolic_decline,
                    time_clean,
                    rate_clean,
                    p0=[rate_clean[0], 0.1, 0.5],
                    bounds=([0, 0, 0], [np.inf, 1, 2])
                )
                fitted_curve = self.hyperbolic_decline(time, *popt)
                params = {'q_i': popt[0], 'd_i': popt[1], 'b': popt[2]}

            elif decline_type == 'harmonic':
                popt, pcov = curve_fit(
                    self.harmonic_decline,
                    time_clean,
                    rate_clean,
                    p0=[rate_clean[0], 0.1],
                    bounds=([0, 0], [np.inf, 1])
                )
                fitted_curve = self.harmonic_decline(time, *popt)
                params = {'q_i': popt[0], 'd_i': popt[1]}

            else:
                raise ValueError(f"Unknown decline type: {decline_type}")

            return fitted_curve, params

        except Exception as e:
            logger.error(f"Error fitting decline curve: {str(e)}")
            return np.zeros_like(time), {}

    def analyze_well_decline(self, well_data: WellData,
                             fluid_type: str = 'oil') -> Dict:
        """Analyze production decline for a well"""
        try:
            # Get production data
            prod_data = well_data.production_data
            if fluid_type == 'oil':
                rate_col = 'Oil_Production_BBL'
            elif fluid_type == 'gas':
                rate_col = 'Gas_Production_MCF'
            else:
                raise ValueError(f"Unknown fluid type: {fluid_type}")

            # Prepare time and rate data
            time = np.arange(len(prod_data))
            rate = prod_data[rate_col].values

            # Fit different decline curves
            results = {}
            for decline_type in ['exponential', 'hyperbolic', 'harmonic']:
                fitted_curve, params = self.fit_decline_curve(time, rate, decline_type)

                # Calculate error metrics
                mse = np.mean((rate - fitted_curve) ** 2)
                rmse = np.sqrt(mse)
                mae = np.mean(np.abs(rate - fitted_curve))

                results[decline_type] = {
                    'parameters': params,
                    'fitted_curve': fitted_curve,
                    'mse': mse,
                    'rmse': rmse,
                    'mae': mae
                }

            # Determine best fit
            best_fit = min(results.keys(), key=lambda k: results[k]['rmse'])
            results['best_fit'] = best_fit

            return results

        except Exception as e:
            logger.error(f"Error analyzing well decline: {str(e)}")
            return {}

    def forecast_production(self, well_data: WellData,
                            forecast_days: int = 365,
                            fluid_type: str = 'oil') -> pd.DataFrame:
        """Forecast future production"""
        try:
            # Analyze decline
            decline_results = self.analyze_well_decline(well_data, fluid_type)
            if not decline_results:
                return pd.DataFrame()

            # Get best fit parameters
            best_fit = decline_results['best_fit']
            params = decline_results[best_fit]['parameters']

            # Generate forecast
            current_days = len(well_data.production_data)
            forecast_time = np.arange(current_days, current_days + forecast_days)

            if best_fit == 'exponential':
                forecast_rate = self.exponential_decline(
                    forecast_time,
                    params['q_i'],
                    params['d']
                )
            elif best_fit == 'hyperbolic':
                forecast_rate = self.hyperbolic_decline(
                    forecast_time,
                    params['q_i'],
                    params['d_i'],
                    params['b']
                )
            else:  # harmonic
                forecast_rate = self.harmonic_decline(
                    forecast_time,
                    params['q_i'],
                    params['d_i']
                )

            # Create forecast DataFrame
            last_date = well_data.production_data['Date'].iloc[-1]
            forecast_dates = pd.date_range(
                start=last_date + pd.Timedelta(days=1),
                periods=forecast_days,
                freq='D'
            )

            forecast_df = pd.DataFrame({
                'Date': forecast_dates,
                f'{fluid_type.capitalize()}_Rate': forecast_rate,
                'Forecast_Type': best_fit
            })

            return forecast_df

        except Exception as e:
            logger.error(f"Error forecasting production: {str(e)}")
            return pd.DataFrame()

    def calculate_reserves(self, well_data: WellData,
                           economic_limit: float,
                           fluid_type: str = 'oil') -> Dict:
        """Calculate remaining reserves"""
        try:
            # Analyze decline
            decline_results = self.analyze_well_decline(well_data, fluid_type)
            if not decline_results:
                return {}

            # Get best fit parameters
            best_fit = decline_results['best_fit']
            params = decline_results[best_fit]['parameters']

            # Calculate time to economic limit
            if best_fit == 'exponential':
                t_limit = -np.log(economic_limit / params['q_i']) / params['d']
            elif best_fit == 'hyperbolic':
                t_limit = (
                                  (params['q_i'] / economic_limit) ** params['b'] - 1
                          ) / (params['b'] * params['d_i'])
            else:  # harmonic
                t_limit = (params['q_i'] / economic_limit - 1) / params['d_i']

            # Calculate cumulative production
            current_days = len(well_data.production_data)
            if fluid_type == 'oil':
                current_cum = well_data.production_data['Oil_Production_BBL'].sum()
            else:
                current_cum = well_data.production_data['Gas_Production_MCF'].sum()

            # Calculate remaining reserves
            time = np.linspace(current_days, t_limit, 1000)
            if best_fit == 'exponential':
                rate = self.exponential_decline(time, params['q_i'], params['d'])
            elif best_fit == 'hyperbolic':
                rate = self.hyperbolic_decline(
                    time, params['q_i'], params['d_i'], params['b'])
            else:
                rate = self.harmonic_decline(time, params['q_i'], params['d_i'])

            remaining_reserves = np.trapz(rate, time)

            return {
                'current_cumulative': current_cum,
                'remaining_reserves': remaining_reserves,
                'total_reserves': current_cum + remaining_reserves,
                'days_to_limit': t_limit - current_days,
                'economic_limit': economic_limit,
                'best_fit_type': best_fit
            }

        except Exception as e:
            logger.error(f"Error calculating reserves: {str(e)}")
            return {}

    def calculate_decline_rate(self, well_data: WellData,
                               period_days: int = 365) -> Dict:
        """Calculate production decline rate"""
        try:
            prod_data = well_data.production_data

            # Calculate decline rates for oil and gas
            decline_rates = {}
            for fluid_type in ['oil', 'gas']:
                if fluid_type == 'oil':
                    rate_col = 'Oil_Production_BBL'
                else:
                    rate_col = 'Gas_Production_MCF'

                # Calculate rates for different time periods
                for days in [30, 90, 180, 365]:
                    if len(prod_data) >= days:
                        initial_rate = prod_data[rate_col].iloc[-days:].head(30).mean()
                        final_rate = prod_data[rate_col].iloc[-30:].mean()

                        nominal_decline = (initial_rate - final_rate) / initial_rate
                        effective_decline = -np.log(final_rate / initial_rate) / (days / 365)

                        decline_rates[f'{fluid_type}_{days}d_nominal'] = nominal_decline
                        decline_rates[f'{fluid_type}_{days}d_effective'] = effective_decline

            return decline_rates

        except Exception as e:
            logger.error(f"Error calculating decline rate: {str(e)}")
            return {}

    def analyze_production_metrics(self, well_data: WellData) -> Dict:
        """Analyze various production metrics"""
        try:
            prod_data = well_data.production_data

            # Calculate production metrics
            metrics = {
                'days_on_production': len(prod_data),
                'peak_oil_rate': prod_data['Oil_Production_BBL'].max(),
                'peak_gas_rate': prod_data['Gas_Production_MCF'].max(),
                'current_oil_rate': prod_data['Oil_Production_BBL'].iloc[-30:].mean(),
                'current_gas_rate': prod_data['Gas_Production_MCF'].iloc[-30:].mean(),
                'cumulative_oil': prod_data['Oil_Production_BBL'].sum(),
                'cumulative_gas': prod_data['Gas_Production_MCF'].sum(),
                'average_water_cut': prod_data['Water_Cut_Percentage'].mean(),
                'current_water_cut': prod_data['Water_Cut_Percentage'].iloc[-30:].mean()
            }

            # Calculate rolling averages
            for window in [30, 90, 180]:
                metrics.update({
                    f'oil_rate_{window}d_avg': prod_data['Oil_Production_BBL']
                    .rolling(window=window).mean().iloc[-1],
                    f'gas_rate_{window}d_avg': prod_data['Gas_Production_MCF']
                    .rolling(window=window).mean().iloc[-1],
                    f'water_cut_{window}d_avg': prod_data['Water_Cut_Percentage']
                    .rolling(window=window).mean().iloc[-1]
                })

            return metrics

        except Exception as e:
            logger.error(f"Error analyzing production metrics: {str(e)}")
            return {}