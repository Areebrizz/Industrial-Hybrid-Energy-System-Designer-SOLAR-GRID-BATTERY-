# ⚡ Industrial Hybrid Energy System Designer

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-FF4B4B)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Tests](https://github.com/yourusername/industrial-hybrid-energy-designer/actions/workflows/ci.yml/badge.svg)](https://github.com/yourusername/industrial-hybrid-energy-designer/actions/workflows/ci.yml)

A production-grade web application for designing and optimizing industrial hybrid energy systems with solar PV and battery storage.

## Features

- **Hourly Simulation**: 8760-hour energy balance simulation
- **Optimization**: Minimize LCOE while maintaining ≥95% reliability
- **Financial Analysis**: NPV, IRR, Payback, LCOE calculations
- **Environmental Impact**: CO₂ emissions reduction tracking
- **Battery Degradation**: Cycle-based aging model
- **Interactive UI**: Streamlit-based dashboard with real-time sensitivity analysis
- **Export Reports**: Generate PDF feasibility studies

## Quick Start

### Local Installation

```bash
# Clone repository
git clone https://github.com/yourusername/industrial-hybrid-energy-designer.git
cd industrial-hybrid-energy-designer

# Create virtual environment
python -m venv venv

# Activate environment
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run application
streamlit run src/app.py
