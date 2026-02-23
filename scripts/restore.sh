#!/bin/bash

# Restore script for trading bot state

if [ -z "$1" ]; then
    echo "Usage: ./restore.sh <backup_file>"
    echo "Available backups:"
    ls -1 backups/backup_*.tar.gz
    exit 1
fi

BACKUP_FILE=$1

if [ ! -f "$BACKUP_FILE" ]; then
    echo "Backup file not found: $BACKUP_FILE"
    exit 1
fi

# Stop the bot if running
pkill -f "python main.py" 2>/dev/null

# Create temp directory for restoration
TEMP_DIR=$(mktemp -d)
tar -xzf "$BACKUP_FILE" -C $TEMP_DIR

# Restore files
cp -r $TEMP_DIR/*/models . 2>/dev/null
cp $TEMP_DIR/*/portfolio_state.json . 2>/dev/null
cp $TEMP_DIR/*/last_update.txt . 2>/dev/null

rm -rf $TEMP_DIR

echo "Restored from $BACKUP_FILE"