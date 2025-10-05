"""
Snowflake EDA Agent - Production-Ready Exploratory Data Analysis
Comprehensive EDA tool for Snowflake tables with automated metric computation
"""

from .agent import SnowflakeEDAAgent
from .database import SnowflakeConnection, SchemaManager, ResultsStorage

__version__ = "1.0.0"
__all__ = ["SnowflakeEDAAgent", "SnowflakeConnection", "SchemaManager", "ResultsStorage"]
