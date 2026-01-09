"""
Database Client - Read historical device states from Home Assistant MariaDB

This module handles all database interactions for the prediction service.
It reads historical state data that Home Assistant has stored in MariaDB.

Key Points:
- READ-ONLY: Never writes to the database
- Connects to the same MariaDB that Home Assistant uses
- Queries the 'states' and 'states_meta' tables
- Returns pandas DataFrames for ML processing
"""

import os
import logging
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import pandas as pd
import mysql.connector
from mysql.connector import Error
from mysql.connector.pooling import MySQLConnectionPool

logger = logging.getLogger(__name__)


class DatabaseClient:
    """
    Client for reading Home Assistant historical data from MariaDB.
    
    Home Assistant stores state changes in these tables:
    - states_meta: Entity metadata (entity_id mapped to metadata_id)
    - states: Actual state changes with timestamps
    
    We join these tables to get historical data for specific entities.
    """
    
    def __init__(self):
        """Initialize database connection pool"""
        self.config = {
            'host': os.getenv('MARIADB_HOST', 'localhost'),
            'port': int(os.getenv('MARIADB_PORT', '3306')),
            'user': os.getenv('MARIADB_USER', 'homeassistant'),
            'password': os.getenv('MARIADB_PASSWORD'),
            'database': os.getenv('MARIADB_DATABASE', 'homeassistant'),
        }
        
        # Create connection pool (reuse connections efficiently)
        try:
            self.pool = MySQLConnectionPool(
                pool_name="prediction_pool",
                pool_size=5,
                **self.config
            )
            logger.info(f"Database connection pool created: {self.config['host']}/{self.config['database']}")
        except Error as e:
            logger.error(f"Failed to create connection pool: {e}")
            self.pool = None
    
    def get_connection(self):
        """Get a connection from the pool"""
        if not self.pool:
            raise Exception("Database connection pool not initialized")
        return self.pool.get_connection()
    
    def load_historical_states(
        self, 
        entity_id: str, 
        days: int = 30,
        end_date: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Load historical state changes for a specific entity.
        
        OPTIMIZED for large databases (200M+ records):
        - Uses last_updated_ts (indexed column)
        - Limits max records to prevent memory issues
        
        Args:
            entity_id: Home Assistant entity ID (e.g., 'light.living_room')
            days: Number of days of history to fetch (default: 30)
            end_date: Optional end date (default: now)
        
        Returns:
            DataFrame with columns:
                - timestamp: When the state changed
                - state: The state value ('on', 'off', or numeric)
                - attributes: JSON attributes (if any)
        
        Example:
            df = client.load_historical_states('light.living_room', days=30)
            # Returns DataFrame with all state changes in last 30 days
        """
        if not end_date:
            end_date = datetime.now()
        
        start_date = end_date - timedelta(days=days)
        
        # Optimized query using indexed timestamp
        # LIMIT to 10000 records max to prevent memory issues
        query = """
        SELECT 
            FROM_UNIXTIME(s.last_updated_ts) as timestamp,
            s.state,
            s.attributes
        FROM states s
        FORCE INDEX (ix_states_last_updated_ts)
        INNER JOIN states_meta sm ON s.metadata_id = sm.metadata_id
        WHERE sm.entity_id = %s
            AND s.last_updated_ts >= UNIX_TIMESTAMP(%s)
            AND s.last_updated_ts <= UNIX_TIMESTAMP(%s)
        ORDER BY s.last_updated_ts ASC
        LIMIT 10000
        """
        
        try:
            conn = self.get_connection()
            cursor = conn.cursor(dictionary=True)
            
            cursor.execute(query, (entity_id, start_date, end_date))
            results = cursor.fetchall()
            
            cursor.close()
            conn.close()
            
            if not results:
                logger.warning(f"No historical data found for {entity_id}")
                return pd.DataFrame()
            
            # Convert to DataFrame
            df = pd.DataFrame(results)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            logger.info(f"Loaded {len(df)} records for {entity_id} from {start_date} to {end_date}")
            return df
            
        except Error as e:
            logger.error(f"Database error loading history for {entity_id}: {e}")
            return pd.DataFrame()
    
    def get_available_entities(self, limit: int = 100, days: int = 1) -> List[str]:
        """
        Get list of entities that have recent state changes.
        
        OPTIMIZED FOR HIGH-VOLUME: Uses only 1 day scan by default
        (Changed from 7 days due to 6.8M records/week = ~970K records/day)
        
        Args:
            limit: Maximum number of entities to return
            days: Number of recent days to check (default: 1)
        
        Returns:
            List of entity IDs that have data in recent days
        """
        # OPTIMIZATION: Only scan last 1 day (not 7)
        # With 970K records/day, we need minimal scan window
        # This still finds all active entities
        query = """
        SELECT sm.entity_id, COUNT(*) as state_count
        FROM (
            SELECT metadata_id
            FROM states
            FORCE INDEX (ix_states_last_updated_ts)
            WHERE last_updated_ts >= UNIX_TIMESTAMP(DATE_SUB(NOW(), INTERVAL %s DAY))
            LIMIT 5000
        ) s
        INNER JOIN states_meta sm ON s.metadata_id = sm.metadata_id
        GROUP BY sm.entity_id
        HAVING state_count > 5
        ORDER BY state_count DESC
        LIMIT %s
        """
        
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            logger.info(f"Discovering entities (scanning last {days} day, max 5000 records)...")
            cursor.execute(query, (days, limit))
            results = cursor.fetchall()
            
            cursor.close()
            conn.close()
            
            entities = [row[0] for row in results]
            logger.info(f"Found {len(entities)} entities with sufficient data")
            return entities
            
        except Error as e:
            logger.error(f"Database error getting available entities: {e}")
            # Fallback to even simpler query
            return self._get_available_entities_simple(limit)
    
    def _get_available_entities_simple(self, limit: int) -> List[str]:
        """
        Ultra-simple fallback: Just get random entities from states_meta
        """
        query = """
        SELECT entity_id
        FROM states_meta
        WHERE entity_id LIKE 'light.%'
           OR entity_id LIKE 'switch.%'
           OR entity_id LIKE 'binary_sensor.%'
        LIMIT %s
        """
        
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            cursor.execute(query, (limit,))
            results = cursor.fetchall()
            
            cursor.close()
            conn.close()
            
            entities = [row[0] for row in results]
            logger.info(f"Found {len(entities)} entities (simple fallback)")
            return entities
            
        except Error as e:
            logger.error(f"Simple fallback also failed: {e}")
            return []
    
    def get_entities_by_pattern(self, pattern: str = 'light.%', limit: int = 50) -> List[str]:
        """
        Get entities by pattern match (bypasses state history scan).
        
        FASTEST METHOD: Queries states_meta only, no historical scan.
        Use this if you know what type of entities you want to train.
        
        Args:
            pattern: SQL LIKE pattern (e.g., 'light.%', 'switch.%', 'binary_sensor.%')
            limit: Maximum entities to return
        
        Returns:
            List of entity IDs matching pattern
        
        Examples:
            # Get all lights
            lights = client.get_entities_by_pattern('light.%')
            
            # Get all switches
            switches = client.get_entities_by_pattern('switch.%')
            
            # Get specific device
            living_room = client.get_entities_by_pattern('%.living_room')
        """
        query = """
        SELECT entity_id
        FROM states_meta
        WHERE entity_id LIKE %s
        ORDER BY entity_id
        LIMIT %s
        """
        
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            cursor.execute(query, (pattern, limit))
            results = cursor.fetchall()
            
            cursor.close()
            conn.close()
            
            entities = [row[0] for row in results]
            logger.info(f"Found {len(entities)} entities matching pattern '{pattern}'")
            return entities
            
        except Error as e:
            logger.error(f"Error getting entities by pattern: {e}")
            return []
    
    def get_entity_state_count(self, entity_id: str, days: int = 30) -> int:
        """
        Count how many state changes an entity has in recent history.
        
        OPTIMIZED: Uses last_updated_ts (indexed) instead of last_changed
        
        Useful for determining if there's enough data to train a model.
        We need at least 100-200 state changes for decent predictions.
        
        Args:
            entity_id: Entity to check
            days: Number of days to look back
        
        Returns:
            Count of state changes
        """
        # Use last_updated_ts instead of last_changed (faster with large tables)
        query = """
        SELECT COUNT(*) as count
        FROM states s
        FORCE INDEX (ix_states_last_updated_ts)
        INNER JOIN states_meta sm ON s.metadata_id = sm.metadata_id
        WHERE sm.entity_id = %s
            AND s.last_updated_ts >= UNIX_TIMESTAMP(DATE_SUB(NOW(), INTERVAL %s DAY))
        """
        
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            cursor.execute(query, (entity_id, days))
            result = cursor.fetchone()
            
            cursor.close()
            conn.close()
            
            count = result[0] if result else 0
            return count
            
        except Error as e:
            logger.error(f"Database error counting states for {entity_id}: {e}")
            return 0
    
    def health_check(self) -> Dict[str, any]:
        """
        Check if database connection is healthy.
        
        Returns:
            Dict with connection status and basic info
        """
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            # Test query
            cursor.execute("SELECT VERSION()")
            version = cursor.fetchone()[0]
            
            # Check states table size
            cursor.execute("SELECT COUNT(*) FROM states WHERE last_changed >= DATE_SUB(NOW(), INTERVAL 7 DAY)")
            recent_states = cursor.fetchone()[0]
            
            cursor.close()
            conn.close()
            
            return {
                "status": "healthy",
                "database": self.config['database'],
                "host": self.config['host'],
                "version": version,
                "recent_states_7d": recent_states
            }
            
        except Error as e:
            logger.error(f"Database health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e)
            }


# Example usage (for testing)
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Create client
    client = DatabaseClient()
    
    # Health check
    health = client.health_check()
    print(f"Database health: {health}")
    
    # Get available entities
    entities = client.get_available_entities(limit=10)
    print(f"Available entities: {entities[:5]}...")
    
    # Load data for first entity
    if entities:
        df = client.load_historical_states(entities[0], days=7)
        print(f"\nSample data for {entities[0]}:")
        print(df.head())
        print(f"\nTotal records: {len(df)}")