#!/bin/bash
# Setup script for Google VM

echo "Setting up trading bot environment..."

# Update system
sudo apt-get update
sudo apt-get upgrade -y

# Install Python and pip
sudo apt-get install -y python3-pip python3-venv

# Create virtual environment
python3 -m venv trading_env
source trading_env/bin/activate

# Install requirements
pip install --upgrade pip
pip install -r requirements.txt

# Create directory structure
mkdir -p trading_bot/{trading,utils,scripts,logs}

# Set up cron job for automatic start (optional)
# crontab -e
# Add: @reboot cd /path/to/trading_bot && source trading_env/bin/activate && python main.py >> logs/trading.log 2>&1

echo "Setup complete!"
echo "To run the bot:"
echo "1. cd trading_bot"
echo "2. source ../trading_env/bin/activate"
echo "3. python main.py"