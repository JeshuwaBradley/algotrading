#!/bin/bash

# Backup script for trading bot state

BACKUP_DIR="backups"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
BACKUP_PATH="$BACKUP_DIR/backup_$TIMESTAMP"

mkdir -p $BACKUP_PATH

# Copy model files
cp -r models $BACKUP_PATH/
cp portfolio_state.json $BACKUP_PATH/ 2>/dev/null
cp last_update.txt $BACKUP_PATH/ 2>/dev/null

# Create compressed archive
tar -czf "$BACKUP_PATH.tar.gz" -C $BACKUP_DIR "backup_$TIMESTAMP"
rm -rf $BACKUP_PATH

echo "Backup created: $BACKUP_PATH.tar.gz"

# Keep only last 10 backups
cd $BACKUP_DIR
ls -t backup_*.tar.gz | tail -n +11 | xargs -r rm
cd ..

echo "Old backups cleaned up"