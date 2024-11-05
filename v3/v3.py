import os
import subprocess
import tempfile
import warnings

import matplotlib.pyplot as plt
import numpy as np
import numpy_financial as npf
import pandas as pd
import seaborn as sns
import streamlit as st
from matplotlib.gridspec import GridSpec

# Set up styles and configurations
warnings.filterwarnings("ignore")
sns.set_style("whitegrid")
plt.style.use('tableau-colorblind10')
colors = sns.color_palette("husl", 8)

# Configure Streamlit page
st.set_page_config(page_title="Well Abandonment Monte Carlo Simulation with ROI", layout="wide")


# Initialize session state
def init_session_state():
    if 'simulation_results' not in st.session_state:
        st.session_state.simulation_results = None
    if 'last_params' not in st.session_state:
        st.session_state.last_params = None


init_session_state()


def store_simulation_results(initial_productions, decline_rates, oil_prices, operating_costs,
                             abandonment_times, npvs, irr_values, payback_periods,
                             profitability_indexes, simulation_years):
    st.session_state.simulation_results = {
        'initial_productions': initial_productions,
        'decline_rates': decline_rates,
        'oil_prices': oil_prices,
        'operating_costs': operating_costs,
        'abandonment_times': abandonment_times,
        'npvs': npvs,
        'irr_values': irr_values,
        'payback_periods': payback_periods,
        'profitability_indexes': profitability_indexes,
        'simulation_years': simulation_years
    }


def create_sensitivity_plot(df, target_var, variables_to_analyze):
    correlations = []
    for var in variables_to_analyze:
        correlation = df[var].corr(df[target_var])
        correlations.append((var, correlation))

    correlations.sort(key=lambda x: abs(x[1]))

    fig, ax = plt.subplots(figsize=(10, 6))
    y_pos = np.arange(len(correlations))
    correlations_val = [x[1] for x in correlations]

    bars = ax.barh(y_pos, correlations_val)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([x[0] for x in correlations])
    ax.set_xlabel('Correlation Coefficient')
    ax.set_title(f'Sensitivity Analysis - Correlation with {target_var}')

    for i, bar in enumerate(bars):
        bar.set_color(colors[0] if correlations_val[i] >= 0 else colors[3])

    return fig


def create_decline_curve(initial_prod, decline_rate, years, decline_model="Exponential", b_factor=0.5):
    time = np.arange(years)
    if decline_model == "Exponential":
        production = initial_prod * np.exp(-decline_rate * time)
    else:  # Hyperbolic
        production = initial_prod / ((1 + b_factor * decline_rate * time) ** (1 / b_factor))
    return time, production


def create_dashboards():
    if st.session_state.simulation_results is None:
        st.warning("Please run the simulation first.")
        return

    results = st.session_state.simulation_results

    tabs = st.tabs(["Sensitivity Dashboard", "Economic Analysis",
                    "Comparative Analysis", "Monte Carlo Distribution"])

    with tabs[0]:
        st.subheader("Interactive Sensitivity Analysis Dashboard")

        col1, col2 = st.columns(2)
        with col1:
            param_x = st.selectbox(
                "Select Parameter for X-axis",
                ["Initial Production", "Decline Rate", "Oil Price", "Operating Cost", "NPV", "IRR"],
                key="sensitivity_param_x"
            )

        with col2:
            param_y = st.selectbox(
                "Select Parameter for Y-axis",
                ["NPV", "IRR", "Payback Period", "Profitability Index"],
                key="sensitivity_param_y"
            )

        x_data = {
            "Initial Production": results['initial_productions'],
            "Decline Rate": results['decline_rates'] * 100,
            "Oil Price": results['oil_prices'],
            "Operating Cost": results['operating_costs'],
            "NPV": results['npvs'],
            "IRR": results['irr_values'] * 100
        }

        y_data = {
            "NPV": results['npvs'],
            "IRR": results['irr_values'] * 100,
            "Payback Period": results['payback_periods'],
            "Profitability Index": results['profitability_indexes']
        }

        fig_interactive = plt.figure(figsize=(10, 6))
        scatter = plt.scatter(x_data[param_x], y_data[param_y],
                              c=results['abandonment_times'], cmap='viridis', alpha=0.6)
        plt.colorbar(scatter, label='Abandonment Time (years)')
        plt.xlabel(param_x)
        plt.ylabel(param_y)
        plt.title(f'{param_y} vs {param_x} Relationship')
        plt.grid(True, alpha=0.3)
        st.pyplot(fig_interactive)

        if st.checkbox("Show Regression Analysis", key="show_regression"):
            x = x_data[param_x]
            y = y_data[param_y]
            mask = ~(np.isnan(x) | np.isnan(y))
            x = x[mask]
            y = y[mask]
            z = np.polyfit(x, y, 1)
            p = np.poly1d(z)
            correlation = np.corrcoef(x, y)[0, 1]

            st.write(f"Correlation coefficient: {correlation:.3f}")
            st.write(f"Linear regression equation: y = {z[0]:.3f}x + {z[1]:.3f}")

    with tabs[1]:
        st.subheader("Economic Analysis Dashboard")
        years = range(1, int(max(results['abandonment_times'])) + 1)

        yearly_metrics = pd.DataFrame({
            'Year': years,
            'Cumulative NPV': [np.mean([npv if abandon >= year else 0
                                        for npv, abandon in zip(results['npvs'],
                                                                results['abandonment_times'])])
                               for year in years],
            'Active Wells': [np.sum(results['abandonment_times'] >= year)
                             for year in years],
            'Average Production': [np.mean([ip * np.exp(-dr * (year - 1))
                                            if abandon >= year else 0
                                            for ip, dr, abandon in
                                            zip(results['initial_productions'],
                                                results['decline_rates'],
                                                results['abandonment_times'])])
                                   for year in years]
        })

        metric_choice = st.selectbox(
            "Select Economic Metric",
            ["Cumulative NPV", "Active Wells", "Average Production"],
            key="econ_metric_choice"
        )

        fig_econ = plt.figure(figsize=(12, 6))
        plt.plot(yearly_metrics['Year'], yearly_metrics[metric_choice],
                 marker='o', linewidth=2, markersize=8)
        plt.xlabel('Year')
        plt.ylabel(metric_choice)
        plt.title(f'{metric_choice} Over Time')
        plt.grid(True)
        st.pyplot(fig_econ)

        col1, col2 = st.columns(2)
        with col1:
            npv_threshold = st.slider(
                "NPV Threshold (USD)",
                float(min(results['npvs'])),
                float(max(results['npvs'])),
                (float(min(results['npvs'])), float(max(results['npvs']))),
                key="npv_threshold"
            )

        with col2:
            irr_threshold = st.slider(
                "IRR Threshold (%)",
                float(min(results['irr_values'] * 100)),
                float(max(results['irr_values'] * 100)),
                (float(min(results['irr_values'] * 100)),
                 float(max(results['irr_values'] * 100))),
                key="irr_threshold"
            )

        success_npv = np.sum((results['npvs'] >= npv_threshold[0]) &
                             (results['npvs'] <= npv_threshold[1])) / len(results['npvs']) * 100
        success_irr = np.sum((results['irr_values'] * 100 >= irr_threshold[0]) &
                             (results['irr_values'] * 100 <= irr_threshold[1])) / len(results['irr_values']) * 100

        st.metric("Projects Meeting NPV Criteria", f"{success_npv:.1f}%")
        st.metric("Projects Meeting IRR Criteria", f"{success_irr:.1f}%")

    with tabs[2]:
        st.subheader("Comparative Analysis Dashboard")

        col1, col2 = st.columns(2)
        with col1:
            percentile_lower = st.number_input(
                "Lower Percentile",
                min_value=0,
                max_value=100,
                value=25,
                key="comp_percentile_lower"
            )
        with col2:
            percentile_upper = st.number_input(
                "Upper Percentile",
                min_value=0,
                max_value=100,
                value=75,
                key="comp_percentile_upper"
            )

        metrics = {
            'NPV': results['npvs'],
            'IRR (%)': results['irr_values'] * 100,
            'Payback Period': results['payback_periods'],
            'Profitability Index': results['profitability_indexes']
        }

        comparison_df = pd.DataFrame({
            'Metric': list(metrics.keys()),
            f'{percentile_lower}th Percentile': [np.percentile(v, percentile_lower)
                                                 for v in metrics.values()],
            'Mean': [np.mean(v) for v in metrics.values()],
            'Median': [np.median(v) for v in metrics.values()],
            f'{percentile_upper}th Percentile': [np.percentile(v, percentile_upper)
                                                 for v in metrics.values()]
        })

        st.table(comparison_df)

        fig_violin = plt.figure(figsize=(12, 6))
        data_to_plot = [
            results['npvs'] / 1e6,
            results['irr_values'] * 100,
            results['payback_periods'],
            results['profitability_indexes']
        ]

        plt.violinplot(data_to_plot, showmeans=True)
        plt.xticks([1, 2, 3, 4], ['NPV (M$)', 'IRR (%)', 'Payback (years)', 'PI'])
        plt.title('Distribution Comparison of Key Metrics')
        st.pyplot(fig_violin)

    with tabs[3]:
        st.subheader("Monte Carlo Distribution Analysis")

        param_choice = st.selectbox(
            "Select Parameter to Analyze",
            ["Initial Production", "Decline Rate", "Oil Price", "Operating Cost",
             "NPV", "IRR", "Payback Period", "Profitability Index"],
            key="mc_param_choice"
        )

        param_data = {
            "Initial Production": results['initial_productions'],
            "Decline Rate": results['decline_rates'] * 100,
            "Oil Price": results['oil_prices'],
            "Operating Cost": results['operating_costs'],
            "NPV": results['npvs'],
            "IRR": results['irr_values'] * 100,
            "Payback Period": results['payback_periods'],
            "Profitability Index": results['profitability_indexes']
        }

        col1, col2 = st.columns(2)
        with col1:
            fig_hist = plt.figure(figsize=(10, 6))
            sns.histplot(data=param_data[param_choice], kde=True)
            plt.title(f'{param_choice} Distribution')
            plt.xlabel(param_choice)
            plt.ylabel('Frequency')
            st.pyplot(fig_hist)

        with col2:
            fig_box = plt.figure(figsize=(10, 6))
            plt.boxplot(param_data[param_choice], vert=False)
            plt.scatter(param_data[param_choice],
                        np.ones_like(param_data[param_choice]) * 1,
                        alpha=0.2)
            plt.title(f'{param_choice} Box Plot')
            plt.ylabel('')
            st.pyplot(fig_box)

        moments_df = pd.DataFrame({
            'Statistic': ['Mean', 'Median', 'Std Dev', 'Skewness', 'Kurtosis',
                          '10th Percentile', '90th Percentile'],
            'Value': [
                np.mean(param_data[param_choice]),
                np.median(param_data[param_choice]),
                np.std(param_data[param_choice]),
                pd.Series(param_data[param_choice]).skew(),
                pd.Series(param_data[param_choice]).kurtosis(),
                np.percentile(param_data[param_choice], 10),
                np.percentile(param_data[param_choice], 90)
            ]
        })

        st.table(moments_df)

        threshold = st.slider(
            f"Select {param_choice} Threshold",
            float(min(param_data[param_choice])),
            float(max(param_data[param_choice])),
            float(np.median(param_data[param_choice])),
            key="mc_threshold"
        )

        prob_exceed = np.mean(param_data[param_choice] > threshold) * 100
        st.metric(f"Probability of Exceeding {threshold:.2f}", f"{prob_exceed:.1f}%")


def create_visualization_tabs(initial_productions, decline_rates, oil_prices, operating_costs,
                              abandonment_times, npvs, irr_values, payback_periods,
                              profitability_indexes, simulation_years, decline_model="Exponential"):
    tabs = st.tabs(["Basic Metrics", "Advanced Analysis", "Production Profile", "Risk Analysis",
                    "Sensitivity Dashboard", "Economic Analysis", "Comparative Analysis", "Monte Carlo Distribution"])

    with tabs[0]:
        fig2 = plt.figure(figsize=(15, 10))
        gs = GridSpec(2, 2, figure=fig2)

        ax1 = fig2.add_subplot(gs[0, 0])
        sns.histplot(data=npvs, kde=True, color=colors[0], ax=ax1)
        ax1.set_xlabel('NPV (USD)')
        ax1.set_title('NPV Distribution')

        ax2 = fig2.add_subplot(gs[0, 1])
        sns.histplot(data=irr_values * 100, kde=True, color=colors[1], ax=ax2)
        ax2.set_xlabel('IRR (%)')
        ax2.set_title('IRR Distribution')

        ax3 = fig2.add_subplot(gs[1, 0])
        sns.histplot(data=payback_periods, kde=True, color=colors[2], ax=ax3)
        ax3.set_xlabel('Payback Period (years)')
        ax3.set_title('Payback Period Distribution')

        ax4 = fig2.add_subplot(gs[1, 1])
        sns.histplot(data=profitability_indexes, kde=True, color=colors[3], ax=ax4)
        ax4.set_xlabel('Profitability Index')
        ax4.set_title('Profitability Index Distribution')

        plt.tight_layout()
        st.pyplot(fig2)

    with tabs[1]:
        col1, col2 = st.columns(2)

        with col1:
            fig_cdf = plt.figure(figsize=(10, 6))
            plt.hist(npvs, bins=50, density=True, cumulative=True,
                     label='CDF', alpha=0.8, color=colors[4])
            plt.xlabel('NPV (USD)')
            plt.ylabel('Cumulative Probability')
            plt.title('Cumulative Distribution of NPV')
            plt.grid(True, alpha=0.3)
            st.pyplot(fig_cdf)

        with col2:
            fig_scatter = plt.figure(figsize=(10, 6))
            plt.scatter(npvs, irr_values * 100, alpha=0.5, c=colors[5])
            plt.xlabel('NPV (USD)')
            plt.ylabel('IRR (%)')
            plt.title('NPV vs IRR Relationship')
            plt.grid(True, alpha=0.3)
            st.pyplot(fig_scatter)

    with tabs[2]:
        col1, col2 = st.columns(2)

        with col1:
            fig_decline = plt.figure(figsize=(10, 6))
            time, prod_exp = create_decline_curve(initial_productions.mean(),
                                                  decline_rates.mean(),
                                                  simulation_years,
                                                  decline_model)
            time, prod_hyp = create_decline_curve(initial_productions.mean(),
                                                  decline_rates.mean(),
                                                  simulation_years,
                                                  "Hyperbolic")
            plt.plot(time, prod_exp, label='Exponential Decline', color=colors[6])
            plt.plot(time, prod_hyp, label='Hyperbolic Decline', color=colors[7])
            plt.xlabel('Year')
            plt.ylabel('Production Rate (bbl/day)')
            plt.title('Production Decline Curves')
            plt.legend()
            plt.grid(True, alpha=0.3)
            st.pyplot(fig_decline)

        with col2:
            fig_abandon = plt.figure(figsize=(10, 6))
            sns.histplot(data=abandonment_times, kde=True, color=colors[0])
            plt.xlabel('Abandonment Time (years)')
            plt.ylabel('Frequency')
            plt.title('Abandonment Time Distribution')
            plt.grid(True, alpha=0.3)
            st.pyplot(fig_abandon)

    with tabs[3]:
        col1, col2 = st.columns(2)

        with col1:
            variables_to_analyze = ['Initial Production Rate', 'Decline Rate',
                                    'Oil Price', 'Operating Cost']
            analysis_df = pd.DataFrame({
                'Initial Production Rate': initial_productions,
                'Decline Rate': decline_rates,
                'Oil Price': oil_prices,
                'Operating Cost': operating_costs,
                'NPV': npvs
            })
            fig_sensitivity = create_sensitivity_plot(analysis_df, 'NPV',
                                                      variables_to_analyze)
            st.pyplot(fig_sensitivity)

        with col2:
            fig_risk = plt.figure(figsize=(10, 6))
            plt.scatter(abandonment_times, npvs, c=irr_values * 100,
                        cmap='viridis', alpha=0.6)
            plt.colorbar(label='IRR (%)')
            plt.xlabel('Abandonment Time (years)')
            plt.ylabel('NPV (USD)')
            plt.title('Risk Matrix: NPV vs Abandonment Time')
            plt.grid(True, alpha=0.3)
            st.pyplot(fig_risk)

    with tabs[4]:
        st.subheader("Interactive Sensitivity Analysis Dashboard")

        col1, col2 = st.columns(2)
        with col1:
            param_x = st.selectbox(
                "Select Parameter for X-axis",
                ["Initial Production", "Decline Rate", "Oil Price", "Operating Cost", "NPV", "IRR"],
                key="param_x"
            )

        with col2:
            param_y = st.selectbox(
                "Select Parameter for Y-axis",
                ["NPV", "IRR", "Payback Period", "Profitability Index"],
                key="param_y"
            )

        x_data = {
            "Initial Production": initial_productions,
            "Decline Rate": decline_rates * 100,
            "Oil Price": oil_prices,
            "Operating Cost": operating_costs,
            "NPV": npvs,
            "IRR": irr_values * 100
        }

        y_data = {
            "NPV": npvs,
            "IRR": irr_values * 100,
            "Payback Period": payback_periods,
            "Profitability Index": profitability_indexes
        }

        fig_interactive = plt.figure(figsize=(10, 6))
        scatter = plt.scatter(x_data[param_x], y_data[param_y],
                              c=abandonment_times, cmap='viridis', alpha=0.6)
        plt.colorbar(scatter, label='Abandonment Time (years)')
        plt.xlabel(param_x)
        plt.ylabel(param_y)
        plt.title(f'{param_y} vs {param_x} Relationship')
        plt.grid(True, alpha=0.3)
        st.pyplot(fig_interactive)

        if st.checkbox("Show Regression Analysis", key="show_regression"):
            x = x_data[param_x]
            y = y_data[param_y]
            z = np.polyfit(x, y, 1)
            p = np.poly1d(z)
            correlation = np.corrcoef(x, y)[0, 1]

            st.write(f"Correlation coefficient: {correlation:.3f}")
            st.write(f"Linear regression equation: y = {z[0]:.3f}x + {z[1]:.3f}")

    with tabs[5]:
        st.subheader("Economic Analysis Dashboard")

        years = range(1, int(max(abandonment_times)) + 1)
        yearly_metrics = pd.DataFrame({
            'Year': years,
            'Cumulative NPV': [np.mean([npv if abandon >= year else 0
                                        for npv, abandon in zip(npvs, abandonment_times)])
                               for year in years],
            'Active Wells': [np.sum(abandonment_times >= year) for year in years],
            'Average Production': [np.mean([ip * np.exp(-dr * (year - 1)) if abandon >= year else 0
                                            for ip, dr, abandon in
                                            zip(initial_productions, decline_rates, abandonment_times)])
                                   for year in years]
        })

        metric_choice = st.selectbox(
            "Select Economic Metric",
            ["Cumulative NPV", "Active Wells", "Average Production"],
            key="metric_choice"
        )

        fig_econ = plt.figure(figsize=(12, 6))
        plt.plot(yearly_metrics['Year'], yearly_metrics[metric_choice],
                 marker='o', linewidth=2, markersize=8)
        plt.xlabel('Year')
        plt.ylabel(metric_choice)
        plt.title(f'{metric_choice} Over Time')
        plt.grid(True)
        st.pyplot(fig_econ)

        col1, col2 = st.columns(2)
        with col1:
            npv_threshold = st.slider(
                "NPV Threshold (USD)",
                float(min(npvs)),
                float(max(npvs)),
                (float(min(npvs)), float(max(npvs))),
                key="npv_threshold"
            )

        with col2:
            irr_threshold = st.slider(
                "IRR Threshold (%)",
                float(min(irr_values * 100)),
                float(max(irr_values * 100)),
                (float(min(irr_values * 100)), float(max(irr_values * 100))),
                key="irr_threshold"
            )

        success_npv = np.sum((npvs >= npv_threshold[0]) & (npvs <= npv_threshold[1])) / len(npvs) * 100
        success_irr = np.sum((irr_values * 100 >= irr_threshold[0]) &
                             (irr_values * 100 <= irr_threshold[1])) / len(irr_values) * 100

        st.metric("Projects Meeting NPV Criteria", f"{success_npv:.1f}%")
        st.metric("Projects Meeting IRR Criteria", f"{success_irr:.1f}%")

    with tabs[6]:
        st.subheader("Comparative Analysis Dashboard")

        col1, col2 = st.columns(2)

        with col1:
            percentile_lower = st.number_input(
                "Lower Percentile",
                min_value=0, max_value=100, value=25,
                key="percentile_lower"
            )
        with col2:
            percentile_upper = st.number_input(
                "Upper Percentile",
                min_value=0, max_value=100, value=75,
                key="percentile_upper"
            )

        metrics = {
            'NPV': npvs,
            'IRR (%)': irr_values * 100,
            'Payback Period': payback_periods,
            'Profitability Index': profitability_indexes
        }

        comparison_df = pd.DataFrame({
            'Metric': list(metrics.keys()),
            f'{percentile_lower}th Percentile': [np.percentile(v, percentile_lower) for v in metrics.values()],
            'Mean': [np.mean(v) for v in metrics.values()],
            'Median': [np.median(v) for v in metrics.values()],
            f'{percentile_upper}th Percentile': [np.percentile(v, percentile_upper) for v in metrics.values()]
        })

        st.table(comparison_df)

        fig_violin = plt.figure(figsize=(12, 6))
        data_to_plot = [
            npvs / 1e6,  # Convert to millions
            irr_values * 100,
            payback_periods,
            profitability_indexes
        ]

        plt.violinplot(data_to_plot, showmeans=True)
        plt.xticks([1, 2, 3, 4], ['NPV (M$)', 'IRR (%)', 'Payback (years)', 'PI'])
        plt.title('Distribution Comparison of Key Metrics')
        st.pyplot(fig_violin)

    with tabs[7]:
        st.subheader("Monte Carlo Distribution Analysis")

        param_choice = st.selectbox(
            "Select Parameter to Analyze",
            ["Initial Production", "Decline Rate", "Oil Price", "Operating Cost",
             "NPV", "IRR", "Payback Period", "Profitability Index"],
            key="param_distribution"
        )

        param_data = {
            "Initial Production": initial_productions,
            "Decline Rate": decline_rates * 100,
            "Oil Price": oil_prices,
            "Operating Cost": operating_costs,
            "NPV": npvs,
            "IRR": irr_values * 100,
            "Payback Period": payback_periods,
            "Profitability Index": profitability_indexes
        }

        col1, col2 = st.columns(2)

        with col1:
            fig_hist = plt.figure(figsize=(10, 6))
            sns.histplot(data=param_data[param_choice], kde=True)
            plt.title(f'{param_choice} Distribution')
            plt.xlabel(param_choice)
            plt.ylabel('Frequency')
            st.pyplot(fig_hist)

        with col2:
            fig_box = plt.figure(figsize=(10, 6))
            plt.boxplot(param_data[param_choice], vert=False)
            plt.scatter(param_data[param_choice],
                        np.ones_like(param_data[param_choice]) * 1,
                        alpha=0.2)
            plt.title(f'{param_choice} Box Plot')
            plt.ylabel('')
            st.pyplot(fig_box)

        moments_df = pd.DataFrame({
            'Statistic': ['Mean', 'Median', 'Std Dev', 'Skewness', 'Kurtosis',
                          '10th Percentile', '90th Percentile'],
            'Value': [
                np.mean(param_data[param_choice]),
                np.median(param_data[param_choice]),
                np.std(param_data[param_choice]),
                pd.Series(param_data[param_choice]).skew(),
                pd.Series(param_data[param_choice]).kurtosis(),
                np.percentile(param_data[param_choice], 10),
                np.percentile(param_data[param_choice], 90)
            ]
        })

        st.table(moments_df)

        threshold = st.slider(
            f"Select {param_choice} Threshold",
            float(min(param_data[param_choice])),
            float(max(param_data[param_choice])),
            float(np.median(param_data[param_choice])),
            key="threshold_slider"
        )

        prob_exceed = np.mean(param_data[param_choice] > threshold) * 100
        st.metric(f"Probability of Exceeding {threshold:.2f}", f"{prob_exceed:.1f}%")


def create_summary_statistics(abandonment_times, npvs, irr_values):
    summary_df = pd.DataFrame({
        'Statistic': ['Mean', 'Median', 'Std Dev', 'Min', 'Max',
                      'Skewness', 'Kurtosis', '10th Percentile',
                      '90th Percentile'],
        'Abandonment Time (years)': [
            np.mean(abandonment_times),
            np.median(abandonment_times),
            np.std(abandonment_times),
            np.min(abandonment_times),
            np.max(abandonment_times),
            pd.Series(abandonment_times).skew(),
            pd.Series(abandonment_times).kurtosis(),
            np.percentile(abandonment_times, 10),
            np.percentile(abandonment_times, 90)
        ],
        'NPV (USD)': [
            np.mean(npvs),
            np.median(npvs),
            np.std(npvs),
            np.min(npvs),
            np.max(npvs),
            pd.Series(npvs).skew(),
            pd.Series(npvs).kurtosis(),
            np.percentile(npvs, 10),
            np.percentile(npvs, 90)
        ],
        'IRR (%)': [
            np.mean(irr_values) * 100,
            np.median(irr_values) * 100,
            np.std(irr_values) * 100,
            np.min(irr_values) * 100,
            np.max(irr_values) * 100,
            pd.Series(irr_values).skew(),
            pd.Series(irr_values).kurtosis(),
            np.percentile(irr_values, 10) * 100,
            np.percentile(irr_values, 90) * 100
        ]
    })
    return summary_df


# Sidebar inputs
st.sidebar.header("Simulation Options")

# Option to choose input method
input_method = st.sidebar.radio("Select Input Method", ["Use Sidebar Parameters", "Upload Parameters CSV"])

if input_method == "Upload Parameters CSV":
    st.sidebar.subheader("Upload Parameters CSV")
    uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type=["csv"])

    # Option to choose execution mode
    st.sidebar.subheader("Execution Mode")
    execution_mode = st.sidebar.selectbox("Select Execution Mode", ["Run Locally", "Submit to SLURM"])

    if execution_mode == "Submit to SLURM":
        st.sidebar.subheader("SLURM Job Settings")
        job_name = st.sidebar.text_input("Job Name", value="monte_carlo_simulation")
        partition = st.sidebar.text_input("Partition", value="compute")
        time_limit = st.sidebar.text_input("Time Limit (HH:MM:SS)", value="01:00:00")
else:
    st.sidebar.subheader("Input Parameters")

    # Number of simulation runs
    num_simulations = st.sidebar.number_input(
        "Number of Simulations", min_value=100, max_value=50000, value=1000, step=100
    )

    # Simulation years
    simulation_years = st.sidebar.number_input(
        "Simulation Duration (years)", min_value=1, max_value=50, value=20, step=1
    )

    st.sidebar.subheader("Parameter Distributions")
    distribution_options = ["Normal", "Log-Normal", "Triangular"]


    # Function to sample distributions
    def sample_distribution(dist_name, params, size):
        if dist_name == "Normal":
            samples = np.random.normal(params['mean'], params['std'], size)
        elif dist_name == "Log-Normal":
            sigma = np.sqrt(np.log(1 + (params['std'] / params['mean']) ** 2))
            mu = np.log(params['mean']) - 0.5 * sigma ** 2
            samples = np.random.lognormal(mu, sigma, size)
        elif dist_name == "Triangular":
            samples = np.random.triangular(params['left'], params['mode'], params['right'], size)
        else:
            samples = np.random.normal(params['mean'], params['std'], size)
        return samples


    # Initial Production Rate
    st.sidebar.subheader("Initial Production Rate (barrels/day)")
    initial_prod_dist = st.sidebar.selectbox("Distribution", distribution_options, key="ip_dist")
    initial_prod_params = {}
    if initial_prod_dist == "Triangular":
        initial_prod_params['left'] = st.sidebar.number_input("Minimum", value=800.0, key="ip_min")
        initial_prod_params['mode'] = st.sidebar.number_input("Most Likely", value=1000.0, key="ip_mode")
        initial_prod_params['right'] = st.sidebar.number_input("Maximum", value=1200.0, key="ip_max")
    else:
        initial_prod_params['mean'] = st.sidebar.number_input("Mean", value=1000.0, key="ip_mean")
        initial_prod_params['std'] = st.sidebar.number_input("Standard Deviation", value=100.0, key="ip_std")

    # Decline Rate
    st.sidebar.subheader("Decline Rate (% per year)")
    decline_rate_dist = st.sidebar.selectbox("Distribution", distribution_options, key="dr_dist")
    decline_rate_params = {}
    if decline_rate_dist == "Triangular":
        decline_rate_params['left'] = st.sidebar.number_input("Minimum", value=5.0, key="dr_min")
        decline_rate_params['mode'] = st.sidebar.number_input("Most Likely", value=10.0, key="dr_mode")
        decline_rate_params['right'] = st.sidebar.number_input("Maximum", value=15.0, key="dr_max")
    else:
        decline_rate_params['mean'] = st.sidebar.number_input("Mean (%)", value=10.0, key="dr_mean")
        decline_rate_params['std'] = st.sidebar.number_input("Standard Deviation (%)", value=2.0, key="dr_std")

    # Decline Model
    st.sidebar.subheader("Production Decline Model")
    decline_model = st.sidebar.selectbox("Select Decline Model", ["Exponential", "Hyperbolic"])

    # Oil Price
    st.sidebar.subheader("Oil Price (USD/barrel)")
    oil_price_dist = st.sidebar.selectbox("Distribution", distribution_options, key="op_dist")
    oil_price_params = {}
    if oil_price_dist == "Triangular":
        oil_price_params['left'] = st.sidebar.number_input("Minimum", value=50.0, key="op_min")
        oil_price_params['mode'] = st.sidebar.number_input("Most Likely", value=70.0, key="op_mode")
        oil_price_params['right'] = st.sidebar.number_input("Maximum", value=90.0, key="op_max")
    else:
        oil_price_params['mean'] = st.sidebar.number_input("Mean", value=70.0, key="op_mean")
        oil_price_params['std'] = st.sidebar.number_input("Standard Deviation", value=10.0, key="op_std")

    # Operating Costs
    st.sidebar.subheader("Operating Costs (USD/barrel)")
    operating_cost_dist = st.sidebar.selectbox("Distribution", distribution_options, key="oc_dist")
    operating_cost_params = {}
    if operating_cost_dist == "Triangular":
        operating_cost_params['left'] = st.sidebar.number_input("Minimum", value=20.0, key="oc_min")
        operating_cost_params['mode'] = st.sidebar.number_input("Most Likely", value=30.0, key="oc_mode")
        operating_cost_params['right'] = st.sidebar.number_input("Maximum", value=40.0, key="oc_max")
    else:
        operating_cost_params['mean'] = st.sidebar.number_input("Mean", value=30.0, key="oc_mean")
        operating_cost_params['std'] = st.sidebar.number_input("Standard Deviation", value=5.0, key="oc_std")

    # Operating Cost Escalation Rate
    st.sidebar.subheader("Operating Cost Escalation Rate (% per year)")
    op_cost_esc_rate = st.sidebar.number_input("Escalation Rate (%)", value=2.0, key="oc_esc") / 100

    # Oil Price Escalation Rate
    st.sidebar.subheader("Oil Price Escalation Rate (% per year)")
    oil_price_esc_rate = st.sidebar.number_input("Escalation Rate (%)", value=1.0, key="op_esc") / 100

    # Maintenance Costs
    st.sidebar.subheader("Annual Maintenance Costs (USD/year)")
    maintenance_cost = st.sidebar.number_input("Maintenance Cost", value=50000.0, key="maint_cost")

    # Capital Expenditures (CapEx)
    st.sidebar.subheader("Capital Expenditures (USD)")
    capex = st.sidebar.number_input("Initial CapEx", value=1000000.0, key="capex")
    capex_schedule = st.sidebar.text_input(
        "Additional CapEx Schedule (year:amount, comma-separated)",
        value="5:200000,10:200000",
        key="capex_schedule"
    )

    # Abandonment Cost
    st.sidebar.subheader("Abandonment Cost (USD)")
    abandonment_cost_dist = st.sidebar.selectbox("Distribution", distribution_options, key="ac_dist")
    abandonment_cost_params = {}
    if abandonment_cost_dist == "Triangular":
        abandonment_cost_params['left'] = st.sidebar.number_input("Minimum", value=400000.0, key="ac_min")
        abandonment_cost_params['mode'] = st.sidebar.number_input("Most Likely", value=500000.0, key="ac_mode")
        abandonment_cost_params['right'] = st.sidebar.number_input("Maximum", value=600000.0, key="ac_max")
    else:
        abandonment_cost_params['mean'] = st.sidebar.number_input("Mean", value=500000.0, key="ac_mean")
        abandonment_cost_params['std'] = st.sidebar.number_input("Standard Deviation", value=50000.0, key="ac_std")

    # Discount Rate
    st.sidebar.subheader("Discount Rate (% per year)")
    discount_rate_dist = st.sidebar.selectbox("Distribution", distribution_options, key="discount_rate_dist")
    discount_rate_params = {}
    if discount_rate_dist == "Triangular":
        discount_rate_params['left'] = st.sidebar.number_input("Minimum", value=6.0, key="discount_rate_min")
        discount_rate_params['mode'] = st.sidebar.number_input("Most Likely", value=8.0, key="discount_rate_mode")
        discount_rate_params['right'] = st.sidebar.number_input("Maximum", value=10.0, key="discount_rate_max")
    else:
        discount_rate_params['mean'] = st.sidebar.number_input("Mean (%)", value=8.0, key="discount_rate_mean")
        discount_rate_params['std'] = st.sidebar.number_input("Standard Deviation (%)", value=1.0,
                                                              key="discount_rate_std")

    st.sidebar.subheader("Corporate Tax Rate (%)")
    tax_rate = st.sidebar.number_input("Tax Rate (%)", value=30.0, key="tax_rate") / 100

    # Royalties
    st.sidebar.subheader("Royalties (% of revenue)")
    royalty_rate = st.sidebar.number_input("Royalty Rate (%)", value=12.5, key="royalty_rate") / 100

    # Inflation Rate
    st.sidebar.subheader("Inflation Rate (% per year)")
    inflation_rate = st.sidebar.number_input("Inflation Rate (%)", value=2.0, key="inflation_rate") / 100

    # Economic Limit
    st.sidebar.subheader("Economic Limit (USD/year)")
    economic_limit = st.sidebar.number_input("Economic Limit (minimum net profit per year)", value=0.0,
                                             key="econ_limit")

    # Correlations
    st.sidebar.subheader("Parameter Correlations")
    apply_correlation = st.sidebar.checkbox("Apply Correlation Between Oil Price and Operating Costs")

    # Financing Options
    st.sidebar.header("Financing Options")
    use_debt = st.sidebar.checkbox("Use Debt Financing?")
    if use_debt:
        debt_ratio = st.sidebar.number_input("Debt Ratio (% of CapEx)", value=50.0, key="debt_ratio") / 100
        interest_rate = st.sidebar.number_input("Interest Rate on Debt (%)", value=5.0, key="interest_rate") / 100
        loan_term = st.sidebar.number_input("Loan Term (years)", value=10, min_value=1, max_value=30, key="loan_term")
    else:
        debt_ratio = 0
        interest_rate = 0
        loan_term = 0

    # Depreciation
    st.sidebar.header("Depreciation")
    use_depreciation = st.sidebar.checkbox("Apply Depreciation?")
    if use_depreciation:
        depreciation_years = st.sidebar.number_input("Depreciation Period (years)",
                                                     value=10, min_value=1, max_value=30, key="dep_years")
    else:
        depreciation_years = 0

    # Run simulation button
    run_simulation = st.sidebar.button("Run Simulation")

    # Main simulation logic
    if run_simulation:
        st.header("Simulation Results")
        try:
            initial_productions = sample_distribution(initial_prod_dist, initial_prod_params, num_simulations)
            decline_rates = sample_distribution(decline_rate_dist, decline_rate_params, num_simulations) / 100
            oil_prices = sample_distribution(oil_price_dist, oil_price_params, num_simulations)
            operating_costs = sample_distribution(operating_cost_dist, operating_cost_params, num_simulations)
            abandonment_costs = sample_distribution(abandonment_cost_dist, abandonment_cost_params, num_simulations)
            discount_rates = sample_distribution(discount_rate_dist, discount_rate_params, num_simulations) / 100

            if apply_correlation:
                oil_price_ranks = oil_prices.argsort().argsort()
                operating_costs = operating_costs[oil_price_ranks]

            abandonment_times = np.zeros(num_simulations)
            npvs = np.zeros(num_simulations)
            irr_values = np.zeros(num_simulations)
            payback_periods = np.zeros(num_simulations)
            profitability_indexes = np.zeros(num_simulations)

            progress_bar = st.progress(0)
            status_text = st.empty()

            capex_schedule_dict = {}
            if capex_schedule:
                capex_items = capex_schedule.split(',')
                for item in capex_items:
                    year_amount = item.strip().split(':')
                    if len(year_amount) == 2:
                        year, amount = year_amount
                        capex_schedule_dict[int(year)] = float(amount)

            for i in range(num_simulations):
                progress_bar.progress((i + 1) / num_simulations)
                status_text.text(f'Running simulation {i + 1} of {num_simulations}')

                initial_prod = initial_productions[i]
                decline_rate = decline_rates[i]
                oil_price = oil_prices[i]
                operating_cost = operating_costs[i]
                abandonment_cost = abandonment_costs[i]
                discount_rate = discount_rates[i]

                equity_ratio = 1 - debt_ratio
                total_capex = capex + sum(capex_schedule_dict.values())
                debt_amount = total_capex * debt_ratio
                equity_amount = total_capex * equity_ratio
                annual_debt_payment = npf.pmt(interest_rate, loan_term, -debt_amount) if use_debt else 0

                annual_depreciation = total_capex / depreciation_years if use_depreciation else 0

                cash_flows = []
                cumulative_cash_flow = -equity_amount
                production = initial_prod
                cumulative_production = 0

                for year in range(1, simulation_years + 1):
                    if decline_model == "Exponential":
                        production = initial_prod * np.exp(-decline_rate * (year - 1))
                    elif decline_model == "Hyperbolic":
                        b_factor = 0.5
                        production = initial_prod / ((1 + b_factor * decline_rate * (year - 1)) ** (1 / b_factor))

                    production = max(production, 0)
                    cumulative_production += production * 365

                    oil_price *= (1 + oil_price_esc_rate)
                    operating_cost *= (1 + op_cost_esc_rate)

                    revenue = production * 365 * oil_price
                    royalty = revenue * royalty_rate
                    cost = production * 365 * operating_cost + maintenance_cost
                    capex_year = capex_schedule_dict.get(year, 0)

                    depreciation = annual_depreciation if use_depreciation and year <= depreciation_years else 0

                    taxable_income = revenue - royalty - cost - depreciation - capex_year
                    tax = taxable_income * tax_rate if taxable_income > 0 else 0

                    net_cash_flow = revenue - royalty - cost - tax - capex_year

                    if use_debt and year <= loan_term:
                        net_cash_flow -= annual_debt_payment

                    cash_flows.append(net_cash_flow)
                    cumulative_cash_flow += net_cash_flow / ((1 + discount_rate) ** year)

                    if cumulative_cash_flow >= 0 and payback_periods[i] == 0:
                        payback_periods[i] = year

                    if net_cash_flow < economic_limit:
                        cumulative_cash_flow -= abandonment_cost / ((1 + discount_rate) ** year)
                        cash_flows[-1] -= abandonment_cost
                        abandonment_times[i] = year
                        npvs[i] = cumulative_cash_flow
                        break
                else:
                    abandonment_times[i] = simulation_years
                    cumulative_cash_flow -= abandonment_cost / ((1 + discount_rate) ** simulation_years)
                    cash_flows[-1] -= abandonment_cost
                    npvs[i] = cumulative_cash_flow

                cash_flow_series = [-equity_amount] + cash_flows
                try:
                    irr_values[i] = npf.irr(cash_flow_series)
                except:
                    irr_values[i] = np.nan

                total_pv_of_cash_flows = npf.npv(discount_rate, cash_flow_series[1:])
                profitability_indexes[i] = total_pv_of_cash_flows / equity_amount if equity_amount != 0 else np.nan

                if payback_periods[i] == 0:
                    payback_periods[i] = simulation_years

            progress_bar.empty()
            status_text.empty()

            store_simulation_results(
                initial_productions, decline_rates, oil_prices, operating_costs,
                abandonment_times, npvs, irr_values, payback_periods,
                profitability_indexes, simulation_years
            )

            main_tabs = st.tabs(["Basic Results", "Interactive Dashboards"])

            with main_tabs[0]:
                create_visualization_tabs(initial_productions, decline_rates, oil_prices, operating_costs,
                                          abandonment_times, npvs, irr_values, payback_periods,
                                          profitability_indexes, simulation_years, decline_model)

            with main_tabs[1]:
                create_dashboards()

            st.subheader("Statistical Summary")
            summary_df = create_summary_statistics(abandonment_times, npvs, irr_values)
            st.table(summary_df)

            st.subheader("Detailed Simulation Data")
            detailed_df = pd.DataFrame({
                'Simulation Run': np.arange(1, num_simulations + 1),
                'Abandonment Time (years)': abandonment_times,
                'NPV (USD)': npvs,
                'IRR (%)': irr_values * 100,
                'Payback Period (years)': payback_periods,
                'Profitability Index': profitability_indexes,
                'Initial Production (bbl/day)': initial_productions,
                'Decline Rate (%)': decline_rates * 100,
                'Oil Price (USD/bbl)': oil_prices,
                'Operating Cost (USD/bbl)': operating_costs
            })
            st.dataframe(detailed_df)

            csv = detailed_df.to_csv(index=False)
            st.download_button(
                label="Download Detailed Results as CSV",
                data=csv,
                file_name='monte_carlo_simulation_results.csv',
                mime='text/csv'
            )

        except Exception as e:
            st.error(f"An error occurred during the simulation: {e}")
elif input_method == "Upload Parameters CSV" and uploaded_file is not None:
    st.header("Simulation Results")
    try:
        # Read parameters from CSV
        param_df = pd.read_csv(uploaded_file)

        # Validate required columns
        required_columns = [
            'initial_prod', 'decline_rate', 'oil_price', 'operating_cost', 'abandonment_cost',
            'discount_rate', 'maintenance_cost', 'capex', 'tax_rate', 'royalty_rate',
            'economic_limit', 'simulation_years', 'oil_price_esc_rate', 'op_cost_esc_rate',
            'capex_schedule', 'decline_model', 'b_factor', 'depreciation_years',
            'debt_ratio', 'interest_rate', 'loan_term'
        ]
        missing_columns = set(required_columns) - set(param_df.columns)
        if missing_columns:
            st.error(f"The following required columns are missing in the CSV file: {', '.join(missing_columns)}")
        else:
            num_simulations = len(param_df)
            st.write(f"Number of simulations to run: {num_simulations}")

            if execution_mode == "Run Locally":
                # Initialize arrays to store results
                abandonment_times = np.zeros(num_simulations)
                npvs = np.zeros(num_simulations)
                irr_values = np.zeros(num_simulations)
                payback_periods = np.zeros(num_simulations)
                profitability_indexes = np.zeros(num_simulations)
                initial_productions = np.zeros(num_simulations)
                decline_rates = np.zeros(num_simulations)
                oil_prices = np.zeros(num_simulations)
                operating_costs = np.zeros(num_simulations)

                # Progress bar and status text
                progress_bar = st.progress(0)
                status_text = st.empty()

                # Simulation loop
                for i in range(num_simulations):
                    # Update progress
                    progress_bar.progress((i + 1) / num_simulations)
                    status_text.text(f'Processing simulation {i + 1} of {num_simulations}')

                    # Extract parameters for this simulation run
                    row = param_df.iloc[i]
                    initial_prod = row['initial_prod']
                    decline_rate = row['decline_rate'] / 100
                    oil_price = row['oil_price']
                    operating_cost = row['operating_cost']
                    abandonment_cost = row['abandonment_cost']
                    discount_rate = row['discount_rate'] / 100
                    maintenance_cost = row['maintenance_cost']
                    capex = row['capex']
                    tax_rate = row['tax_rate'] / 100
                    royalty_rate = row['royalty_rate'] / 100
                    economic_limit = row['economic_limit']
                    simulation_years = int(row['simulation_years'])
                    oil_price_esc_rate = row['oil_price_esc_rate'] / 100
                    op_cost_esc_rate = row['op_cost_esc_rate'] / 100
                    capex_schedule_str = row['capex_schedule']
                    decline_model = row['decline_model']
                    b_factor = row.get('b_factor', 0.5)
                    depreciation_years = int(row.get('depreciation_years', 0))
                    debt_ratio = row.get('debt_ratio', 0) / 100
                    interest_rate = row.get('interest_rate', 0) / 100
                    loan_term = int(row.get('loan_term', 0))

                    # Store input parameters
                    initial_productions[i] = initial_prod
                    decline_rates[i] = decline_rate
                    oil_prices[i] = oil_price
                    operating_costs[i] = operating_cost

                    # Parse CapEx schedule
                    capex_schedule_dict = {}
                    if pd.notna(capex_schedule_str):
                        capex_items = capex_schedule_str.split(',')
                        for item in capex_items:
                            year_amount = item.strip().split(':')
                            if len(year_amount) == 2:
                                year, amount = year_amount
                                capex_schedule_dict[int(year)] = float(amount)

                    # Financing calculations
                    equity_ratio = 1 - debt_ratio
                    total_capex = capex + sum(capex_schedule_dict.values())
                    debt_amount = total_capex * debt_ratio
                    equity_amount = total_capex * equity_ratio
                    annual_debt_payment = npf.pmt(interest_rate, loan_term, -debt_amount) if debt_ratio > 0 else 0

                    # Depreciation calculations
                    use_depreciation = depreciation_years > 0
                    annual_depreciation = total_capex / depreciation_years if use_depreciation else 0

                    cash_flows = []
                    cumulative_cash_flow = -equity_amount
                    production = initial_prod

                    # Yearly simulation
                    for year in range(1, simulation_years + 1):
                        # Calculate production decline
                        if decline_model == "Exponential":
                            production = initial_prod * np.exp(-decline_rate * (year - 1))
                        else:  # Hyperbolic
                            production = initial_prod / ((1 + b_factor * decline_rate * (year - 1))
                                                         ** (1 / b_factor))

                        production = max(production, 0)

                        # Calculate prices and costs with escalation
                        current_oil_price = oil_price * (1 + oil_price_esc_rate) ** (year - 1)
                        current_operating_cost = operating_cost * (1 + op_cost_esc_rate) ** (year - 1)

                        # Calculate revenues and costs
                        revenue = production * 365 * current_oil_price
                        royalty = revenue * royalty_rate
                        operating_expenses = production * 365 * current_operating_cost + maintenance_cost
                        capex_year = capex_schedule_dict.get(year, 0)

                        # Calculate depreciation for this year
                        depreciation = annual_depreciation if use_depreciation and year <= depreciation_years else 0

                        # Calculate taxable income and tax
                        taxable_income = revenue - royalty - operating_expenses - depreciation - capex_year
                        tax = max(0, taxable_income * tax_rate)

                        # Calculate net cash flow
                        net_cash_flow = revenue - royalty - operating_expenses - tax - capex_year

                        if debt_ratio > 0 and year <= loan_term:
                            net_cash_flow -= annual_debt_payment

                        cash_flows.append(net_cash_flow)
                        cumulative_cash_flow += net_cash_flow / ((1 + discount_rate) ** year)

                        # Check for payback period
                        if cumulative_cash_flow >= 0 and payback_periods[i] == 0:
                            payback_periods[i] = year

                        # Check economic limit
                        if net_cash_flow < economic_limit:
                            cumulative_cash_flow -= abandonment_cost / ((1 + discount_rate) ** year)
                            cash_flows[-1] -= abandonment_cost
                            abandonment_times[i] = year
                            npvs[i] = cumulative_cash_flow
                            break
                    else:
                        abandonment_times[i] = simulation_years
                        cumulative_cash_flow -= abandonment_cost / ((1 + discount_rate) ** simulation_years)
                        cash_flows[-1] -= abandonment_cost
                        npvs[i] = cumulative_cash_flow

                    # Calculate IRR and profitability index
                    cash_flow_series = [-equity_amount] + cash_flows
                    try:
                        irr_values[i] = npf.irr(cash_flow_series)
                    except:
                        irr_values[i] = np.nan

                    total_pv_of_cash_flows = npf.npv(discount_rate, cash_flow_series[1:])
                    profitability_indexes[
                        i] = total_pv_of_cash_flows / equity_amount if equity_amount != 0 else np.nan

                    if payback_periods[i] == 0:
                        payback_periods[i] = simulation_years

                # Clear progress indicators
                progress_bar.empty()
                status_text.empty()

                # Store and display results
                store_simulation_results(
                    initial_productions, decline_rates, oil_prices, operating_costs,
                    abandonment_times, npvs, irr_values, payback_periods,
                    profitability_indexes, simulation_years
                )

                main_tabs = st.tabs(["Basic Results", "Interactive Dashboards"])

                with main_tabs[0]:
                    create_visualization_tabs(
                        initial_productions, decline_rates, oil_prices, operating_costs,
                        abandonment_times, npvs, irr_values, payback_periods,
                        profitability_indexes, simulation_years, decline_model
                    )

                with main_tabs[1]:
                    create_dashboards()

                # Display statistical summary
                st.subheader("Statistical Summary")
                summary_df = create_summary_statistics(abandonment_times, npvs, irr_values)
                st.table(summary_df)

                # Create and display detailed results dataframe
                st.subheader("Detailed Simulation Data")
                detailed_df = pd.concat([
                    param_df,
                    pd.DataFrame({
                        'Abandonment Time (years)': abandonment_times,
                        'NPV (USD)': npvs,
                        'IRR (%)': irr_values * 100,
                        'Payback Period (years)': payback_periods,
                        'Profitability Index': profitability_indexes
                    })
                ], axis=1)
                st.dataframe(detailed_df)

                # Download button for detailed results
                csv = detailed_df.to_csv(index=False)
                st.download_button(
                    label="Download Detailed Results as CSV",
                    data=csv,
                    file_name='monte_carlo_simulation_results.csv',
                    mime='text/csv'
                )

            elif execution_mode == "Submit to SLURM":
                # SLURM job submission logic
                st.write("Submitting simulation to SLURM scheduler...")

                # Save parameters to temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as temp_csv:
                    param_df.to_csv(temp_csv.name, index=False)
                    csv_file_path = temp_csv.name

                # Create SLURM job script
                job_script = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --partition={partition}
#SBATCH --time={time_limit}
#SBATCH --output=slurm-%j.out

module load python/3.8

echo "Running Monte Carlo Simulation"
python simulate_monte_carlo.py "{csv_file_path}" "{job_name}_results.csv"
"""

                # Save job script
                with tempfile.NamedTemporaryFile(delete=False, suffix='.sh') as temp_script:
                    temp_script.write(job_script.encode('utf-8'))
                    script_file_path = temp_script.name

                # Submit job
                try:
                    result = subprocess.run(['sbatch', script_file_path], capture_output=True, text=True)
                    if result.returncode == 0:
                        st.success("Job submitted successfully!")
                        st.write(result.stdout)
                    else:
                        st.error("Failed to submit job.")
                        st.write(result.stderr)
                except Exception as e:
                    st.error(f"An error occurred while submitting the job: {e}")

                # Cleanup
                try:
                    os.remove(csv_file_path)
                    os.remove(script_file_path)
                except Exception as e:
                    st.warning(f"Warning: Could not clean up temporary files: {e}")

    except Exception as e:
        st.error(f"An error occurred: {e}")

elif input_method == "Upload Parameters CSV" and uploaded_file is None:
    st.write("Please upload a CSV file with simulation parameters to proceed.")

if __name__ == "__main__":
    main()
