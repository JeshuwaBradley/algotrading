import sqlite3
import pandas as pd
from datetime import datetime
import json
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PortfolioDB:
    """SQLite database for persistent portfolio storage"""
    
    def __init__(self, db_path="portfolio.db"):
        self.db_path = Path(db_path)
        self.init_database()
    
    def init_database(self):
        """Initialize database tables"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Portfolio snapshot table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS portfolio_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME NOT NULL,
                    total_value REAL NOT NULL,
                    cash REAL NOT NULL,
                    positions_json TEXT NOT NULL,
                    returns_daily REAL,
                    returns_cumulative REAL
                )
            ''')
            
            # Trades table for transaction history
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME NOT NULL,
                    symbol TEXT NOT NULL,
                    action TEXT NOT NULL,
                    quantity REAL NOT NULL,
                    price REAL NOT NULL,
                    value REAL NOT NULL,
                    snapshot_id INTEGER,
                    FOREIGN KEY (snapshot_id) REFERENCES portfolio_snapshots (id)
                )
            ''')
            
            # Daily summaries table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS daily_summaries (
                    date DATE PRIMARY KEY,
                    start_value REAL NOT NULL,
                    end_value REAL NOT NULL,
                    daily_return REAL NOT NULL,
                    trades_count INTEGER NOT NULL,
                    metadata_json TEXT
                )
            ''')
            
            conn.commit()
            logger.info("Database initialized successfully")
    
    def save_portfolio_snapshot(self, portfolio, returns=None):
        """
        Save current portfolio state
        portfolio: dict with keys like 'total_value', 'cash', 'positions'
        """
        with sqlite3.connect(self.db_path) as conn:
            # Convert positions to JSON
            positions_json = json.dumps(portfolio.get('positions', {}), default=str)
            
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO portfolio_snapshots 
                (timestamp, total_value, cash, positions_json, returns_daily, returns_cumulative)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                datetime.now(),
                portfolio.get('total_value', 0),
                portfolio.get('cash', 0),
                positions_json,
                returns.get('daily') if returns else None,
                returns.get('cumulative') if returns else None
            ))
            
            snapshot_id = cursor.lastrowid
            conn.commit()
            logger.info(f"Portfolio snapshot saved (ID: {snapshot_id})")
            return snapshot_id
    
    def save_trade(self, trade, snapshot_id=None):
        """Save individual trade"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO trades 
                (timestamp, symbol, action, quantity, price, value, snapshot_id)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                trade.get('timestamp', datetime.now()),
                trade['symbol'],
                trade['action'],
                trade['quantity'],
                trade['price'],
                trade['quantity'] * trade['price'],
                snapshot_id
            ))
            conn.commit()
    
    def save_daily_summary(self, date, start_value, end_value, trades_count, metadata=None):
        """Save end-of-day summary"""
        with sqlite3.connect(self.db_path) as conn:
            daily_return = (end_value - start_value) / start_value if start_value > 0 else 0
            
            cursor = conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO daily_summaries 
                (date, start_value, end_value, daily_return, trades_count, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                date,
                start_value,
                end_value,
                daily_return,
                trades_count,
                json.dumps(metadata) if metadata else None
            ))
            conn.commit()
    
    def get_portfolio_history(self, start_date=None, end_date=None):
        """Get historical portfolio snapshots"""
        query = "SELECT * FROM portfolio_snapshots"
        params = []
        
        if start_date and end_date:
            query += " WHERE timestamp BETWEEN ? AND ?"
            params = [start_date, end_date]
        elif start_date:
            query += " WHERE timestamp >= ?"
            params = [start_date]
        elif end_date:
            query += " WHERE timestamp <= ?"
            params = [end_date]
        
        query += " ORDER BY timestamp"
        
        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query(query, conn, params=params)
            # Parse positions JSON
            if not df.empty:
                df['positions'] = df['positions_json'].apply(json.loads)
            return df
    
    def get_latest_portfolio(self):
        """Get most recent portfolio snapshot"""
        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query('''
                SELECT * FROM portfolio_snapshots 
                ORDER BY timestamp DESC LIMIT 1
            ''', conn)
            
            if not df.empty:
                df['positions'] = df['positions_json'].apply(json.loads)
                return df.iloc[0].to_dict()
            return None