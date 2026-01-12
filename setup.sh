# setup.sh
#!/bin/bash
# Setup script for deployment

echo "Setting up Industrial Hybrid Energy System Designer..."

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Install requirements
pip install --upgrade pip
pip install -r requirements.txt

# Create necessary directories
mkdir -p data/uploads
mkdir -p reports

echo "Setup complete!"
echo "To run the application:"
echo "1. Activate virtual environment: source venv/bin/activate"
echo "2. Run Streamlit: streamlit run app.py"
