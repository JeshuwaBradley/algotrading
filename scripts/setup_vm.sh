#!/bin/bash

# Trading Bot VM Setup Script

echo "Starting trading bot setup..."

# Update system
sudo apt-get update
sudo apt-get upgrade -y

# Install Python and pip
sudo apt-get install -y python3 python3-pip python3-venv

# Install Git
sudo apt-get install -y git

# Create project directory
mkdir -p ~/trading_bot
cd ~/trading_bot

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install requirements
pip install --upgrade pip
pip install -r requirements.txt

# Create necessary directories
mkdir -p trading
mkdir -p utils
mkdir -p scripts
mkdir -p logs
mkdir -p models

# Set up cron job for daily execution (optional)
# Add to crontab: 0 9 * * 1-5 cd /home/ubuntu/trading_bot && venv/bin/python main.py >> logs/trading.log 2>&1

echo "Setup complete!"
echo "To start trading:"
echo "1. cd ~/trading_bot"
echo "2. source venv/bin/activate"
echo "3. python main.py"