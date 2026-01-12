# app.py
"""
Industrial Hybrid Energy System Designer
Production-grade Streamlit application for optimizing hybrid energy systems.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import io
import base64
from typing import Tuple, Dict, List, Optional
import json
from dataclasses import dataclass

# Optimization and mathematical libraries
from scipy.optimize import minimize, LinearConstraint, Bounds
from scipy.interpolate import interp1d
import warnings
warnings.filterwarnings('ignore')

# PDF generation
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
import tempfile

# Set page configuration
st.set_page_config(
    page_title="Industrial Hybrid Energy System Designer",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        font-weight: 700;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #374151;
        font-weight: 600;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #F9FAFB;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #3B82F6;
        margin-bottom: 1rem;
    }
    .warning-box {
        background-color: #FEF3C7;
        border: 1px solid #F59E0B;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #D1FAE5;
        border: 1px solid #10B981;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

@dataclass
class SystemParameters:
    """Data class for system parameters"""
    pv_size_kw: float
    battery_capacity_kwh: float
    battery_power_kw: float
    inverter_size_kw: float
    
@dataclass
class FinancialParameters:
    """Data class for financial parameters"""
    pv_capex_per_kw: float
    battery_capex_per_kwh: float
    inverter_capex_per_kw: float
    installation_cost_percent: float
    om_percent_per_year: float
    discount_rate_percent: float
    project_lifetime_years: int
    grid_tariff_peak: float
    grid_tariff_offpeak: float
    tariff_escalation_percent: float

class PVSystem:
    """Photovoltaic system model"""
    
    def __init__(self, size_kw: float, efficiency: float = 0.18):
        self.size_kw = size_kw
        self.efficiency = efficiency
        self.tilt_angle = 30  # degrees
        self.azimuth = 180  # South-facing
    
    def calculate_power_output(self, irradiance_w_m2: np.ndarray, 
                              temperature_c: np.ndarray,
                              timestamp: pd.DatetimeIndex) -> np.ndarray:
        """
        Calculate PV power output using PVWatts-like model
        """
        # Temperature correction
        temp_coeff = -0.0045  # % per °C
        temp_correction = 1 + temp_coeff * (temperature_c - 25)
        
        # Simple irradiance to power conversion
        pv_output = (irradiance_w_m2 / 1000) * self.size_kw * self.efficiency * temp_correction
        
        # Ensure non-negative values
        pv_output = np.maximum(pv_output, 0)
        
        return pv_output

class BatterySystem:
    """Battery energy storage system model with degradation"""
    
    def __init__(self, capacity_kwh: float, power_kw: float,
                 efficiency_charge: float = 0.95,
                 efficiency_discharge: float = 0.95,
                 dod_max: float = 0.9,
                 cycle_life: int = 6000):
        self.capacity_kwh = capacity_kwh
        self.power_kw = power_kw
        self.efficiency_charge = efficiency_charge
        self.efficiency_discharge = efficiency_discharge
        self.dod_max = dod_max  # Maximum depth of discharge
        self.cycle_life = cycle_life
        self.soc = 0.5 * capacity_kwh  # Initial SOC at 50%
        self.cycle_count = 0
        self.capacity_degradation = 1.0  # Initial capacity factor
        
    def dispatch(self, power_kw: float, timestep_hours: float = 1.0) -> Tuple[float, float]:
        """
        Dispatch battery with degradation tracking
        
        Returns:
            Tuple of (actual_power_kw, degradation_cost)
        """
        if abs(power_kw) > self.power_kw:
            power_kw = np.sign(power_kw) * self.power_kw
        
        energy_kwh = power_kw * timestep_hours
        
        if energy_kwh > 0:  # Charging
            actual_energy = min(
                energy_kwh * self.efficiency_charge,
                (self.capacity_kwh * self.dod_max - self.soc)
            )
            self.soc += actual_energy
        else:  # Discharging
            actual_energy = max(
                energy_kwh / self.efficiency_discharge,
                -self.soc
            )
            self.soc += actual_energy
        
        # Update degradation based on cycle depth
        if actual_energy != 0:
            cycle_depth = abs(actual_energy) / (self.capacity_kwh * self.dod_max)
            self.cycle_count += cycle_depth / 2  # Half cycle for each charge/discharge
            
            # Simple linear degradation model
            capacity_loss = self.cycle_count / self.cycle_life
            self.capacity_degradation = max(0.8, 1.0 - capacity_loss)
        
        actual_power = actual_energy / timestep_hours
        return actual_power, self.capacity_degradation

class GridConnection:
    """Grid connection model with time-of-use tariffs"""
    
    def __init__(self, tariff_peak: float, tariff_offpeak: float,
                 emission_factor_kgco2_per_kwh: float = 0.5):
        self.tariff_peak = tariff_peak  # $/kWh
        self.tariff_offpeak = tariff_offpeak  # $/kWh
        self.emission_factor = emission_factor_kgco2_per_kwh
        
    def get_tariff(self, hour: int) -> float:
        """Get tariff based on hour of day"""
        # Peak hours: 8 AM to 10 PM on weekdays
        is_weekday = hour // 24 < 5  # Monday-Friday
        hour_of_day = hour % 24
        
        if is_weekday and 8 <= hour_of_day < 22:
            return self.tariff_peak
        else:
            return self.tariff_offpeak

class EnergySystemSimulator:
    """Main simulator for the hybrid energy system"""
    
    def __init__(self, load_profile_kw: np.ndarray,
                 pv_system: PVSystem,
                 battery: BatterySystem,
                 grid: GridConnection,
                 outage_hours: List[int] = None):
        self.load_profile = load_profile_kw
        self.pv_system = pv_system
        self.battery = battery
        self.grid = grid
        self.outage_hours = outage_hours if outage_hours else []
        self.hours = len(load_profile_kw)
        
        # Simulation results storage
        self.results = {}
        
    def simulate_hourly(self, irradiance: np.ndarray, 
                       temperature: np.ndarray,
                       timestamp: pd.DatetimeIndex) -> Dict:
        """
        Run hourly simulation for one year
        """
        pv_generation = self.pv_system.calculate_power_output(
            irradiance, temperature, timestamp
        )
        
        # Initialize arrays for results
        grid_power = np.zeros(self.hours)
        battery_power = np.zeros(self.hours)
        battery_soc = np.zeros(self.hours)
        unmet_load = np.zeros(self.hours)
        battery_degradation = np.ones(self.hours)
        
        # Reset battery to initial state
        self.battery.soc = 0.5 * self.battery.capacity_kwh
        self.battery.cycle_count = 0
        self.battery.capacity_degradation = 1.0
        
        # Apply outage hours (grid unavailable)
        grid_available = np.ones(self.hours, dtype=bool)
        if self.outage_hours:
            for hour in self.outage_hours:
                if hour < self.hours:
                    grid_available[hour] = False
        
        for hour in range(self.hours):
            net_load = self.load_profile[hour] - pv_generation[hour]
            
            if grid_available[hour]:
                # Grid available - optimize battery dispatch
                tariff = self.grid.get_tariff(hour)
                
                if net_load > 0:  # Load > Generation
                    # Try to discharge battery first if tariff is high
                    if tariff > self.grid.tariff_offpeak * 1.2:  # Peak tariff
                        discharge_power = min(net_load, self.battery.power_kw)
                        actual_discharge, deg = self.battery.dispatch(-discharge_power)
                        battery_power[hour] = actual_discharge
                        net_load += actual_discharge  # Subtract from net load
                        battery_degradation[hour] = deg
                    
                    grid_power[hour] = max(0, net_load)
                    
                    # Charge battery if excess PV and off-peak
                    if pv_generation[hour] > self.load_profile[hour] and tariff < self.grid.tariff_peak * 0.8:
                        excess = pv_generation[hour] - self.load_profile[hour]
                        charge_power = min(excess, self.battery.power_kw)
                        actual_charge, deg = self.battery.dispatch(charge_power)
                        battery_power[hour] += actual_charge
                        battery_degradation[hour] = deg
                
                else:  # Generation > Load
                    excess = -net_load
                    
                    # Charge battery with excess
                    charge_power = min(excess, self.battery.power_kw)
                    actual_charge, deg = self.battery.dispatch(charge_power)
                    battery_power[hour] = actual_charge
                    battery_degradation[hour] = deg
                    
                    # Export to grid if still excess
                    if excess > charge_power:
                        grid_power[hour] = -(excess - charge_power)  # Negative for export
            else:
                # Grid outage - rely on battery and PV only
                if net_load > 0:  # Need additional power
                    discharge_power = min(net_load, self.battery.power_kw)
                    actual_discharge, deg = self.battery.dispatch(-discharge_power)
                    battery_power[hour] = actual_discharge
                    unmet_load[hour] = max(0, net_load + actual_discharge)
                    battery_degradation[hour] = deg
                else:
                    # Excess PV, charge battery
                    excess = -net_load
                    charge_power = min(excess, self.battery.power_kw)
                    actual_charge, deg = self.battery.dispatch(charge_power)
                    battery_power[hour] = actual_charge
                    battery_degradation[hour] = deg
            
            battery_soc[hour] = self.battery.soc
        
        # Calculate reliability
        total_load = np.sum(self.load_profile)
        unmet_total = np.sum(unmet_load)
        reliability = 1 - (unmet_total / total_load) if total_load > 0 else 1
        
        # Calculate costs and emissions
        grid_cost = 0
        grid_energy = 0
        exported_energy = 0
        
        for hour in range(self.hours):
            if grid_power[hour] > 0:
                tariff = self.grid.get_tariff(hour)
                grid_cost += grid_power[hour] * tariff
                grid_energy += grid_power[hour]
            elif grid_power[hour] < 0:
                # Export - assume lower feed-in tariff
                feed_in_tariff = self.grid.tariff_offpeak * 0.6
                grid_cost += grid_power[hour] * feed_in_tariff
                exported_energy += abs(grid_power[hour])
        
        # Calculate emissions
        grid_emissions = grid_energy * self.grid.emission_factor
        
        self.results = {
            'pv_generation': pv_generation,
            'grid_power': grid_power,
            'battery_power': battery_power,
            'battery_soc': battery_soc,
            'unmet_load': unmet_load,
            'reliability': reliability,
            'grid_cost': grid_cost,
            'grid_energy': grid_energy,
            'exported_energy': exported_energy,
            'grid_emissions': grid_emissions,
            'battery_degradation': battery_degradation
        }
        
        return self.results

class FinancialAnalyzer:
    """Financial analysis for the hybrid system"""
    
    def __init__(self, params: FinancialParameters):
        self.params = params
    
    def calculate_npv(self, pv_size_kw: float, battery_capacity_kwh: float,
                     inverter_size_kw: float, annual_grid_cost: float,
                     annual_om_cost: float) -> Tuple[float, float, float]:
        """
        Calculate NPV, IRR, and Payback period
        """
        # CAPEX calculation
        pv_capex = pv_size_kw * self.params.pv_capex_per_kw
        battery_capex = battery_capacity_kwh * self.params.battery_capex_per_kwh
        inverter_capex = inverter_size_kw * self.params.inverter_capex_per_kw
        
        total_capex = pv_capex + battery_capex + inverter_capex
        installation_cost = total_capex * self.params.installation_cost_percent
        total_capex += installation_cost
        
        # Annual cash flows
        annual_cash_flows = []
        
        for year in range(self.params.project_lifetime_years):
            # Grid cost savings (escalating)
            grid_savings = annual_grid_cost * (1 + self.params.tariff_escalation_percent/100) ** year
            
            # O&M costs
            om_cost = annual_om_cost * (1 + self.params.tariff_escalation_percent/100) ** year
            
            # Annual net cash flow
            net_cash_flow = grid_savings - om_cost
            annual_cash_flows.append(net_cash_flow)
        
        # Calculate NPV
        npv = -total_capex
        for year, cash_flow in enumerate(annual_cash_flows, 1):
            discount_factor = (1 + self.params.discount_rate_percent/100) ** year
            npv += cash_flow / discount_factor
        
        # Calculate IRR (simplified)
        try:
            # Simple IRR approximation
            total_return = sum(annual_cash_flows)
            avg_investment = total_capex / 2
            irr = (total_return / avg_investment) ** (1/self.params.project_lifetime_years) - 1
            irr = min(max(irr * 100, 0), 50)  # Bound between 0% and 50%
        except:
            irr = 0
        
        # Calculate Payback period
        cumulative_cash_flow = 0
        payback_period = None
        
        for year, cash_flow in enumerate(annual_cash_flows, 1):
            cumulative_cash_flow += cash_flow
            if cumulative_cash_flow >= total_capex and payback_period is None:
                payback_period = year
        
        if payback_period is None:
            payback_period = self.params.project_lifetime_years
        
        # Calculate LCOE
        total_energy_produced = sum(annual_cash_flows)  # Simplified
        lcoe = total_capex / total_energy_produced if total_energy_produced > 0 else 0
        
        return npv, irr, payback_period, lcoe, total_capex

class SystemOptimizer:
    """Optimization engine for system sizing"""
    
    def __init__(self, load_profile: np.ndarray, irradiance: np.ndarray,
                 temperature: np.ndarray, timestamp: pd.DatetimeIndex,
                 financial_params: FinancialParameters,
                 grid_params: Tuple[float, float],
                 emission_factor: float,
                 outage_hours: List[int] = None):
        self.load_profile = load_profile
        self.irradiance = irradiance
        self.temperature = temperature
        self.timestamp = timestamp
        self.financial_params = financial_params
        self.grid_peak, self.grid_offpeak = grid_params
        self.emission_factor = emission_factor
        self.outage_hours = outage_hours
        
    def objective_function(self, x: np.ndarray) -> float:
        """
        Objective function for optimization: Minimize LCOE while maintaining reliability
        """
        pv_size = x[0]
        battery_capacity = x[1]
        battery_power = battery_capacity / 4  # 4-hour battery
        
        # Create system components
        pv_system = PVSystem(pv_size)
        battery = BatterySystem(battery_capacity, battery_power)
        grid = GridConnection(self.grid_peak, self.grid_offpeak, self.emission_factor)
        
        # Simulate
        simulator = EnergySystemSimulator(
            self.load_profile, pv_system, battery, grid, self.outage_hours
        )
        results = simulator.simulate_hourly(self.irradiance, self.temperature, self.timestamp)
        
        # Check reliability constraint
        reliability = results['reliability']
        if reliability < 0.95:  # 95% reliability constraint
            # Penalize low reliability
            return 1e9 * (0.95 - reliability) ** 2
        
        # Calculate financial metrics
        analyzer = FinancialAnalyzer(self.financial_params)
        
        # Estimate annual grid cost without system
        grid_only = GridConnection(self.grid_peak, self.grid_offpeak)
        annual_grid_cost = 0
        for hour in range(len(self.load_profile)):
            tariff = grid_only.get_tariff(hour)
            annual_grid_cost += self.load_profile[hour] * tariff
        
        # Annual O&M cost
        annual_om = (pv_size * self.financial_params.pv_capex_per_kw * 
                    self.financial_params.om_percent_per_year/100)
        
        _, _, _, lcoe, _ = analyzer.calculate_npv(
            pv_size, battery_capacity, max(pv_size, battery_power),
            annual_grid_cost - results['grid_cost'], annual_om
        )
        
        return lcoe
    
    def optimize(self) -> Dict:
        """
        Run optimization to find optimal system size
        """
        # Define bounds
        bounds = Bounds(
            [10, 10],  # Minimum PV and battery
            [10000, 10000]  # Maximum PV and battery
        )
        
        # Initial guess
        x0 = [min(1000, max(self.load_profile)), min(2000, max(self.load_profile) * 4)]
        
        # Run optimization
        result = minimize(
            self.objective_function,
            x0,
            method='SLSQP',
            bounds=bounds,
            options={'maxiter': 50, 'ftol': 1e-6}
        )
        
        optimal_pv = max(10, result.x[0])
        optimal_battery = max(10, result.x[1])
        
        return {
            'pv_size_kw': optimal_pv,
            'battery_capacity_kwh': optimal_battery,
            'battery_power_kw': optimal_battery / 4,
            'success': result.success,
            'message': result.message,
            'lcoe': result.fun if result.success else None
        }

def create_sample_load_profile() -> pd.DataFrame:
    """Create a sample load profile if none provided"""
    dates = pd.date_range(start='2024-01-01', periods=8760, freq='H')
    
    # Create synthetic load profile
    base_load = 1000  # kW
    seasonal_variation = 200 * np.sin(2 * np.pi * (dates.dayofyear / 365))
    daily_pattern = 300 * np.sin(2 * np.pi * (dates.hour / 24))
    weekly_pattern = 150 * (dates.dayofweek < 5)  # Higher on weekdays
    noise = 50 * np.random.randn(len(dates))
    
    load = base_load + seasonal_variation + daily_pattern + weekly_pattern + noise
    load = np.maximum(load, 100)  # Minimum load
    
    return pd.DataFrame({
        'timestamp': dates,
        'load_kw': load
    })

def create_sample_irradiance_data() -> pd.DataFrame:
    """Create synthetic irradiance data"""
    dates = pd.date_range(start='2024-01-01', periods=8760, freq='H')
    
    # Clear-sky model
    day_of_year = dates.dayofyear
    hour_of_day = dates.hour
    
    # Solar declination
    declination = 23.45 * np.sin(2 * np.pi * (284 + day_of_year) / 365)
    
    # Hour angle
    hour_angle = 15 * (hour_of_day - 12)
    
    # Solar zenith angle
    lat_rad = np.radians(40)  # Latitude 40°N
    dec_rad = np.radians(declination)
    ha_rad = np.radians(hour_angle)
    
    cos_theta = (np.sin(lat_rad) * np.sin(dec_rad) + 
                 np.cos(lat_rad) * np.cos(dec_rad) * np.cos(ha_rad))
    cos_theta = np.maximum(cos_theta, 0)
    
    # Extraterrestrial radiation
    i0 = 1367  # W/m²
    irradiance = i0 * cos_theta
    
    # Atmospheric attenuation
    air_mass = 1 / (cos_theta + 0.50572 * (96.07995 - np.degrees(np.arccos(cos_theta))) ** -1.6364)
    transmittance = 0.7 ** air_mass
    
    irradiance = irradiance * transmittance
    irradiance = np.maximum(irradiance, 0)
    
    # Temperature data
    temperature = 15 + 10 * np.sin(2 * np.pi * (day_of_year - 105) / 365) + 5 * np.random.randn(len(dates))
    
    return pd.DataFrame({
        'timestamp': dates,
        'irradiance_w_m2': irradiance,
        'temperature_c': temperature
    })

def create_pdf_report(results: Dict, system_params: Dict,
                      financial_params: Dict, filename: str):
    """Generate PDF feasibility report"""
    doc = SimpleDocTemplate(filename, pagesize=letter)
    story = []
    styles = getSampleStyleSheet()
    
    # Title
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        spaceAfter=30,
        textColor=colors.HexColor('#1E3A8A')
    )
    story.append(Paragraph("Industrial Hybrid Energy System Feasibility Study", title_style))
    story.append(Spacer(1, 12))
    
    # Executive Summary
    story.append(Paragraph("Executive Summary", styles['Heading2']))
    story.append(Spacer(1, 6))
    
    summary_text = f"""
    This report presents the feasibility analysis for an industrial hybrid energy system 
    comprising {system_params['pv_size_kw']:.0f} kW of solar PV and {system_params['battery_capacity_kwh']:.0f} kWh 
    of battery storage. The optimized system achieves {results['reliability']*100:.1f}% supply reliability 
    with a Levelized Cost of Energy (LCOE) of ${results['lcoe']:.3f}/kWh.
    """
    story.append(Paragraph(summary_text, styles['Normal']))
    story.append(Spacer(1, 12))
    
    # System Configuration
    story.append(Paragraph("Optimal System Configuration", styles['Heading2']))
    story.append(Spacer(1, 6))
    
    config_data = [
        ['Parameter', 'Value', 'Unit'],
        ['PV System Size', f"{system_params['pv_size_kw']:.0f}", 'kW'],
        ['Battery Capacity', f"{system_params['battery_capacity_kwh']:.0f}", 'kWh'],
        ['Battery Power', f"{system_params['battery_power_kw']:.0f}", 'kW'],
        ['Inverter Size', f"{max(system_params['pv_size_kw'], system_params['battery_power_kw']):.0f}", 'kW'],
        ['System Reliability', f"{results['reliability']*100:.1f}", '%']
    ]
    
    config_table = Table(config_data, colWidths=[2*inch, 1.5*inch, 1*inch])
    config_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3B82F6')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#F9FAFB')),
        ('GRID', (0, 0), (-1, -1), 1, colors.grey),
    ]))
    story.append(config_table)
    story.append(Spacer(1, 12))
    
    # Financial Analysis
    story.append(Paragraph("Financial Analysis", styles['Heading2']))
    story.append(Spacer(1, 6))
    
    financial_data = [
        ['Metric', 'Value'],
        ['Total CAPEX', f"${results['total_capex']/1000:.1f}k"],
        ['NPV', f"${results['npv']/1000:.1f}k"],
        ['IRR', f"{results['irr']:.1f}%"],
        ['Payback Period', f"{results['payback_period']:.1f} years"],
        ['LCOE', f"${results['lcoe']:.3f}/kWh"],
        ['Annual Grid Savings', f"${results['annual_savings']/1000:.1f}k"]
    ]
    
    financial_table = Table(financial_data, colWidths=[2*inch, 2*inch])
    financial_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#10B981')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#D1FAE5')),
        ('GRID', (0, 0), (-1, -1), 1, colors.grey),
    ]))
    story.append(financial_table)
    story.append(Spacer(1, 12))
    
    # Environmental Impact
    story.append(Paragraph("Environmental Impact", styles['Heading2']))
    story.append(Spacer(1, 6))
    
    env_text = f"""
    The proposed hybrid energy system reduces grid energy consumption by {results['grid_reduction_percent']:.1f}%, 
    resulting in annual CO₂ emissions reduction of {results['emissions_reduction']/1000:.1f} tons.
    """
    story.append(Paragraph(env_text, styles['Normal']))
    
    # Generate the PDF
    doc.build(story)

def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown('<h1 class="main-header">⚡ Industrial Hybrid Energy System Designer</h1>', 
                unsafe_allow_html=True)
    st.markdown("Optimize PV and battery sizing for industrial applications with ≥95% reliability")
    
    # Sidebar for inputs
    with st.sidebar:
        st.markdown('<h3 class="sub-header">📊 System Inputs</h3>', unsafe_allow_html=True)
        
        # File uploads
        st.subheader("Data Upload")
        
        load_file = st.file_uploader("Upload Load Profile (CSV)", type=['csv'],
                                     help="CSV with columns: timestamp, load_kw (8760 rows)")
        
        weather_file = st.file_uploader("Upload Weather Data (CSV)", type=['csv'],
                                        help="CSV with columns: timestamp, irradiance_w_m2, temperature_c")
        
        use_sample_data = st.checkbox("Use sample data", value=True)
        
        # Grid parameters
        st.subheader("Grid Parameters")
        col1, col2 = st.columns(2)
        with col1:
            peak_tariff = st.number_input("Peak Tariff ($/kWh)", 
                                         min_value=0.0, max_value=1.0, 
                                         value=0.25, step=0.01)
        with col2:
            offpeak_tariff = st.number_input("Off-peak Tariff ($/kWh)",
                                           min_value=0.0, max_value=1.0,
                                           value=0.12, step=0.01)
        
        emission_factor = st.number_input("CO₂ Emission Factor (kg/kWh)",
                                        min_value=0.0, max_value=2.0,
                                        value=0.5, step=0.01)
        
        # Outage hours
        st.subheader("Grid Outage Hours")
        outage_input = st.text_input("Outage hours (comma-separated)",
                                   value="100,101,102,500,501,502",
                                   help="Hours when grid is unavailable (0-8759)")
        
        # Financial parameters
        st.subheader("Financial Parameters")
        
        col1, col2 = st.columns(2)
        with col1:
            pv_capex = st.number_input("PV CAPEX ($/kW)", value=800.0, step=50.0)
            battery_capex = st.number_input("Battery CAPEX ($/kWh)", value=350.0, step=25.0)
            discount_rate = st.number_input("Discount Rate (%)", value=8.0, step=0.5)
        
        with col2:
            installation_cost = st.number_input("Installation Cost (%)", value=15.0, step=1.0)
            om_cost = st.number_input("O&M Cost (% of CAPEX/year)", value=1.5, step=0.1)
            project_life = st.number_input("Project Lifetime (years)", value=25, step=1)
        
        tariff_escalation = st.number_input("Tariff Escalation (%/year)", 
                                          value=3.0, step=0.1)
        
        # Sensitivity analysis
        st.subheader("Sensitivity Analysis")
        battery_cost_sensitivity = st.slider("Battery Cost Sensitivity (%)",
                                           min_value=-50, max_value=50,
                                           value=0, step=10)
        
        tariff_sensitivity = st.slider("Tariff Sensitivity (%)",
                                     min_value=-30, max_value=30,
                                     value=0, step=5)
        
        # Run optimization button
        run_optimization = st.button("🚀 Run Optimization", type="primary", use_container_width=True)
    
    # Main content area
    if load_file is not None:
        load_data = pd.read_csv(load_file)
    elif use_sample_data:
        load_data = create_sample_load_profile()
        st.info("Using sample load profile data. Upload your own CSV for accurate analysis.")
    else:
        st.warning("Please upload load profile data or use sample data.")
        st.stop()
    
    if weather_file is not None:
        weather_data = pd.read_csv(weather_file)
    elif use_sample_data:
        weather_data = create_sample_irradiance_data()
        st.info("Using sample weather data. Upload your own CSV for accurate analysis.")
    else:
        st.warning("Please upload weather data or use sample data.")
        st.stop()
    
    # Process outage hours
    try:
        outage_hours = [int(h.strip()) for h in outage_input.split(',') if h.strip().isdigit()]
    except:
        outage_hours = []
    
    # Display data preview
    with st.expander("📈 Data Preview", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("Load Profile (first 24 hours):")
            st.dataframe(load_data.head(24), use_container_width=True)
        
        with col2:
            st.write("Weather Data (first 24 hours):")
            st.dataframe(weather_data.head(24), use_container_width=True)
    
    # Run optimization when button is clicked
    if run_optimization:
        with st.spinner("Running optimization... This may take a minute."):
            # Prepare data
            load_profile = load_data['load_kw'].values[:8760]
            irradiance = weather_data['irradiance_w_m2'].values[:8760]
            temperature = weather_data['temperature_c'].values[:8760]
            timestamp = pd.date_range(start='2024-01-01', periods=8760, freq='H')
            
            # Adjust parameters based on sensitivity
            adjusted_battery_capex = battery_capex * (1 + battery_cost_sensitivity/100)
            adjusted_peak_tariff = peak_tariff * (1 + tariff_sensitivity/100)
            adjusted_offpeak_tariff = offpeak_tariff * (1 + tariff_sensitivity/100)
            
            # Create financial parameters
            financial_params = FinancialParameters(
                pv_capex_per_kw=pv_capex,
                battery_capex_per_kwh=adjusted_battery_capex,
                inverter_capex_per_kw=150.0,  # Fixed
                installation_cost_percent=installation_cost/100,
                om_percent_per_year=om_cost/100,
                discount_rate_percent=discount_rate,
                project_lifetime_years=project_life,
                grid_tariff_peak=adjusted_peak_tariff,
                grid_tariff_offpeak=adjusted_offpeak_tariff,
                tariff_escalation_percent=tariff_escalation
            )
            
            # Run optimization
            optimizer = SystemOptimizer(
                load_profile, irradiance, temperature, timestamp,
                financial_params, (adjusted_peak_tariff, adjusted_offpeak_tariff),
                emission_factor, outage_hours
            )
            
            optimization_result = optimizer.optimize()
            
            if optimization_result['success']:
                # Simulate with optimal system
                pv_system = PVSystem(optimization_result['pv_size_kw'])
                battery = BatterySystem(
                    optimization_result['battery_capacity_kwh'],
                    optimization_result['battery_power_kw']
                )
                grid = GridConnection(adjusted_peak_tariff, adjusted_offpeak_tariff, emission_factor)
                
                simulator = EnergySystemSimulator(
                    load_profile, pv_system, battery, grid, outage_hours
                )
                
                simulation_results = simulator.simulate_hourly(
                    irradiance, temperature, timestamp
                )
                
                # Calculate financial metrics
                analyzer = FinancialAnalyzer(financial_params)
                
                # Calculate baseline grid cost
                baseline_grid = GridConnection(adjusted_peak_tariff, adjusted_offpeak_tariff)
                baseline_cost = 0
                for hour in range(len(load_profile)):
                    tariff = baseline_grid.get_tariff(hour)
                    baseline_cost += load_profile[hour] * tariff
                
                annual_savings = baseline_cost - simulation_results['grid_cost']
                annual_om = (optimization_result['pv_size_kw'] * pv_capex * 
                           financial_params.om_percent_per_year)
                
                npv, irr, payback, lcoe, total_capex = analyzer.calculate_npv(
                    optimization_result['pv_size_kw'],
                    optimization_result['battery_capacity_kwh'],
                    max(optimization_result['pv_size_kw'], optimization_result['battery_power_kw']),
                    annual_savings,
                    annual_om
                )
                
                # Calculate emissions reduction
                baseline_emissions = np.sum(load_profile) * emission_factor
                emissions_reduction = baseline_emissions - simulation_results['grid_emissions']
                grid_reduction_percent = (1 - simulation_results['grid_energy'] / np.sum(load_profile)) * 100
                
                # Display results
                st.markdown('<h3 class="sub-header">✅ Optimization Results</h3>', 
                          unsafe_allow_html=True)
                
                # Key metrics in columns
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Optimal PV Size", 
                            f"{optimization_result['pv_size_kw']:.0f} kW",
                            delta=None)
                
                with col2:
                    st.metric("Optimal Battery", 
                            f"{optimization_result['battery_capacity_kwh']:.0f} kWh",
                            delta=None)
                
                with col3:
                    st.metric("System Reliability", 
                            f"{simulation_results['reliability']*100:.1f}%",
                            delta=None)
                
                with col4:
                    st.metric("LCOE", 
                            f"${lcoe:.3f}/kWh",
                            delta=None)
                
                # Financial metrics
                st.markdown('<div class="sub-header">💰 Financial Analysis</div>', 
                          unsafe_allow_html=True)
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total CAPEX", f"${total_capex/1000:.1f}k")
                
                with col2:
                    st.metric("NPV", f"${npv/1000:.1f}k",
                            delta="Positive" if npv > 0 else "Negative")
                
                with col3:
                    st.metric("IRR", f"{irr:.1f}%",
                            delta="Good" if irr > discount_rate else "Poor")
                
                with col4:
                    st.metric("Payback Period", f"{payback:.1f} years")
                
                # Environmental impact
                st.markdown('<div class="sub-header">🌱 Environmental Impact</div>', 
                          unsafe_allow_html=True)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("CO₂ Reduction", 
                            f"{emissions_reduction/1000:.1f} tons/year")
                
                with col2:
                    st.metric("Grid Energy Reduction", 
                            f"{grid_reduction_percent:.1f}%")
                
                # Visualization
                st.markdown('<div class="sub-header">📊 System Performance</div>', 
                          unsafe_allow_html=True)
                
                # Create interactive plots
                tab1, tab2, tab3 = st.tabs(["Weekly Profile", "Annual Summary", "Battery Operation"])
                
                with tab1:
                    # Show one week of data
                    start_hour = 0
                    end_hour = 168  # One week
                    
                    fig = make_subplots(
                        rows=3, cols=1,
                        subplot_titles=('Power Balance', 'Battery State of Charge', 'Grid Power'),
                        vertical_spacing=0.1
                    )
                    
                    fig.add_trace(
                        go.Scatter(x=timestamp[start_hour:end_hour],
                                 y=load_profile[start_hour:end_hour],
                                 name='Load',
                                 line=dict(color='red', width=2)),
                        row=1, col=1
                    )
                    
                    fig.add_trace(
                        go.Scatter(x=timestamp[start_hour:end_hour],
                                 y=simulation_results['pv_generation'][start_hour:end_hour],
                                 name='PV Generation',
                                 line=dict(color='orange', width=2)),
                        row=1, col=1
                    )
                    
                    fig.add_trace(
                        go.Scatter(x=timestamp[start_hour:end_hour],
                                 y=simulation_results['battery_soc'][start_hour:end_hour],
                                 name='Battery SOC',
                                 line=dict(color='blue', width=2)),
                        row=2, col=1
                    )
                    
                    fig.add_trace(
                        go.Scatter(x=timestamp[start_hour:end_hour],
                                 y=simulation_results['grid_power'][start_hour:end_hour],
                                 name='Grid Power',
                                 line=dict(color='green', width=2),
                                 fill='tozeroy'),
                        row=3, col=1
                    )
                    
                    fig.update_layout(height=800, showlegend=True)
                    st.plotly_chart(fig, use_container_width=True)
                
                with tab2:
                    # Monthly aggregation
                    monthly_data = pd.DataFrame({
                        'timestamp': timestamp[:8760],
                        'load': load_profile[:8760],
                        'pv': simulation_results['pv_generation'][:8760],
                        'grid': simulation_results['grid_power'][:8760]
                    })
                    
                    monthly_data['month'] = monthly_data['timestamp'].dt.month
                    monthly_summary = monthly_data.groupby('month').agg({
                        'load': 'sum',
                        'pv': 'sum',
                        'grid': lambda x: x[x > 0].sum()  # Only positive grid consumption
                    }).reset_index()
                    
                    fig2 = go.Figure()
                    
                    fig2.add_trace(go.Bar(
                        x=monthly_summary['month'],
                        y=monthly_summary['load'] / 1000,
                        name='Load (MWh)',
                        marker_color='red'
                    ))
                    
                    fig2.add_trace(go.Bar(
                        x=monthly_summary['month'],
                        y=monthly_summary['pv'] / 1000,
                        name='PV Generation (MWh)',
                        marker_color='orange'
                    ))
                    
                    fig2.add_trace(go.Bar(
                        x=monthly_summary['month'],
                        y=monthly_summary['grid'] / 1000,
                        name='Grid Import (MWh)',
                        marker_color='green'
                    ))
                    
                    fig2.update_layout(
                        title='Monthly Energy Summary',
                        barmode='group',
                        xaxis_title='Month',
                        yaxis_title='Energy (MWh)',
                        height=500
                    )
                    
                    st.plotly_chart(fig2, use_container_width=True)
                
                with tab3:
                    # Battery degradation
                    fig3 = make_subplots(
                        rows=2, cols=1,
                        subplot_titles=('Battery Power', 'Capacity Degradation')
                    )
                    
                    fig3.add_trace(
                        go.Scatter(x=timestamp[:1000],
                                 y=simulation_results['battery_power'][:1000],
                                 name='Battery Power',
                                 line=dict(color='purple', width=1)),
                        row=1, col=1
                    )
                    
                    fig3.add_trace(
                        go.Scatter(x=timestamp[:1000],
                                 y=simulation_results['battery_degradation'][:1000] * 100,
                                 name='Capacity (%)',
                                 line=dict(color='brown', width=2)),
                        row=2, col=1
                    )
                    
                    fig3.update_layout(height=600, showlegend=True)
                    st.plotly_chart(fig3, use_container_width=True)
                
                # Cost breakdown
                st.markdown('<div class="sub-header">💵 Cost Breakdown</div>', 
                          unsafe_allow_html=True)
                
                cost_breakdown = {
                    'PV System': optimization_result['pv_size_kw'] * pv_capex,
                    'Battery Storage': optimization_result['battery_capacity_kwh'] * adjusted_battery_capex,
                    'Inverter': max(optimization_result['pv_size_kw'], 
                                  optimization_result['battery_power_kw']) * 150,
                    'Installation': total_capex * (installation_cost/100)
                }
                
                fig4 = go.Figure(data=[go.Pie(
                    labels=list(cost_breakdown.keys()),
                    values=list(cost_breakdown.values()),
                    hole=0.4,
                    marker_colors=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
                )])
                
                fig4.update_layout(title='CAPEX Breakdown')
                st.plotly_chart(fig4, use_container_width=True)
                
                # Generate PDF report
                st.markdown('<div class="sub-header">📄 Report Generation</div>', 
                          unsafe_allow_html=True)
                
                # Prepare results for PDF
                pdf_results = {
                    'reliability': simulation_results['reliability'],
                    'lcoe': lcoe,
                    'npv': npv,
                    'irr': irr,
                    'payback_period': payback,
                    'total_capex': total_capex,
                    'annual_savings': annual_savings,
                    'grid_reduction_percent': grid_reduction_percent,
                    'emissions_reduction': emissions_reduction
                }
                
                system_params = {
                    'pv_size_kw': optimization_result['pv_size_kw'],
                    'battery_capacity_kwh': optimization_result['battery_capacity_kwh'],
                    'battery_power_kw': optimization_result['battery_power_kw']
                }
                
                financial_params_dict = {
                    'pv_capex_per_kw': pv_capex,
                    'battery_capex_per_kwh': adjusted_battery_capex,
                    'discount_rate_percent': discount_rate
                }
                
                # Create download button for PDF
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
                    create_pdf_report(pdf_results, system_params, financial_params_dict, tmp.name)
                    
                    with open(tmp.name, 'rb') as f:
                        pdf_bytes = f.read()
                    
                    b64 = base64.b64encode(pdf_bytes).decode()
                    href = f'<a href="data:application/pdf;base64,{b64}" download="feasibility_report.pdf">📥 Download Feasibility Report (PDF)</a>'
                    st.markdown(href, unsafe_allow_html=True)
                
                st.success("Optimization completed successfully!")
                
            else:
                st.error(f"Optimization failed: {optimization_result['message']}")
    
    else:
        # Show introductory content
        st.markdown("""
        ### 🎯 Welcome to the Industrial Hybrid Energy System Designer
        
        This application helps you optimize the design of hybrid energy systems for industrial facilities.
        
        **Key Features:**
        
        1. **Load Profile Analysis**: Upload your hourly load data or use our sample data
        2. **PV & Battery Optimization**: Find the optimal system size to minimize LCOE
        3. **Financial Analysis**: Calculate NPV, IRR, Payback period, and LCOE
        4. **Reliability Assessment**: Ensure ≥95% supply reliability
        5. **Environmental Impact**: Calculate CO₂ emissions reduction
        6. **Sensitivity Analysis**: Test different cost and tariff scenarios
        
        **How to use:**
        
        1. Configure system parameters in the sidebar
        2. Upload your load profile and weather data (or use sample data)
        3. Click "Run Optimization" to find the optimal system
        4. Review results and download the feasibility report
        
        **Ready to start?** Configure your parameters in the sidebar and click "Run Optimization"!
        """)
        
        # Show sample visualizations
        if use_sample_data:
            st.markdown('<div class="sub-header">📊 Sample Data Visualization</div>', 
                      unsafe_allow_html=True)
            
            # Show sample load profile
            fig_sample = go.Figure()
            
            # First week of sample data
            sample_dates = pd.date_range(start='2024-01-01', periods=168, freq='H')
            sample_load = load_data['load_kw'].values[:168]
            sample_irradiance = weather_data['irradiance_w_m2'].values[:168]
            
            fig_sample.add_trace(go.Scatter(
                x=sample_dates,
                y=sample_load,
                name='Load (kW)',
                line=dict(color='red', width=2)
            ))
            
            fig_sample.add_trace(go.Scatter(
                x=sample_dates,
                y=sample_irradiance / 5,  # Scale for visualization
                name='Irradiance (W/m² ÷ 5)',
                line=dict(color='orange', width=2),
                yaxis='y2'
            ))
            
            fig_sample.update_layout(
                title='Sample Load Profile and Irradiance (One Week)',
                xaxis_title='Date',
                yaxis_title='Load (kW)',
                yaxis2=dict(
                    title='Irradiance (W/m² ÷ 5)',
                    overlaying='y',
                    side='right'
                ),
                height=400
            )
            
            st.plotly_chart(fig_sample, use_container_width=True)

if __name__ == "__main__":
    main()
