"""
Helper functions for EDA Agent
"""
import re
import json
import math
from typing import Any, Optional, List, Dict
from datetime import datetime
from collections import Counter


def generate_analysis_id() -> str:
    """
    Generate a unique analysis ID based on timestamp
    
    Returns:
        str: Unique analysis ID
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    # Add microseconds for uniqueness
    microseconds = datetime.now().microsecond
    return f"EDA_{timestamp}_{microseconds}"


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Safely divide two numbers, returning default if denominator is zero
    
    Args:
        numerator: Numerator value
        denominator: Denominator value
        default: Default value if division by zero
        
    Returns:
        float: Result of division or default
    """
    try:
        if denominator == 0 or denominator is None:
            return default
        result = numerator / denominator
        # Check for inf or nan
        if math.isnan(result) or math.isinf(result):
            return default
        return result
    except (ZeroDivisionError, TypeError):
        return default


def format_variant_value(value: Any) -> str:
    """
    Format a value for storage as VARIANT in Snowflake
    
    Args:
        value: Value to format
        
    Returns:
        str: JSON-formatted string
    """
    try:
        if value is None:
            return 'NULL'
        if isinstance(value, (dict, list)):
            return json.dumps(value)
        if isinstance(value, (int, float, bool)):
            return json.dumps(value)
        return json.dumps(str(value))
    except Exception:
        return 'NULL'


def compute_entropy(values: List[Any]) -> float:
    """
    Compute Shannon entropy for a list of values
    
    Args:
        values: List of values
        
    Returns:
        float: Entropy value
    """
    try:
        if not values:
            return 0.0
        
        # Count occurrences
        counter = Counter(values)
        total = len(values)
        
        # Calculate entropy
        entropy = 0.0
        for count in counter.values():
            if count > 0:
                probability = count / total
                entropy -= probability * math.log2(probability)
        
        return entropy
    except Exception:
        return 0.0


def detect_patterns(values: List[str]) -> Dict[str, int]:
    """
    Detect common patterns in string values (email, phone, URL, etc.)
    
    Args:
        values: List of string values
        
    Returns:
        dict: Pattern counts
    """
    patterns = {
        'email': 0,
        'phone': 0,
        'url': 0,
        'numeric': 0,
        'alphanumeric': 0
    }
    
    # Regex patterns
    email_pattern = re.compile(r'^[\w\.-]+@[\w\.-]+\.\w+$')
    phone_pattern = re.compile(r'^[\+\d]?[\d\s\-\(\)]{7,}$')
    url_pattern = re.compile(r'^https?://[^\s]+$')
    numeric_pattern = re.compile(r'^\d+$')
    alphanumeric_pattern = re.compile(r'^[a-zA-Z0-9]+$')
    
    for value in values:
        if value is None or not isinstance(value, str):
            continue
        
        value_str = str(value).strip()
        if not value_str:
            continue
        
        if email_pattern.match(value_str):
            patterns['email'] += 1
        elif phone_pattern.match(value_str):
            patterns['phone'] += 1
        elif url_pattern.match(value_str):
            patterns['url'] += 1
        elif numeric_pattern.match(value_str):
            patterns['numeric'] += 1
        elif alphanumeric_pattern.match(value_str):
            patterns['alphanumeric'] += 1
    
    return patterns


def parse_snowflake_type(data_type: str) -> Dict[str, Any]:
    """
    Parse Snowflake data type string into components
    
    Args:
        data_type: Snowflake data type string (e.g., "VARCHAR(100)", "NUMBER(10,2)")
        
    Returns:
        dict: Parsed type information
    """
    type_info = {
        'base_type': data_type,
        'precision': None,
        'scale': None
    }
    
    # Extract base type and parameters
    match = re.match(r'(\w+)(?:\((\d+)(?:,(\d+))?\))?', data_type)
    if match:
        type_info['base_type'] = match.group(1)
        if match.group(2):
            type_info['precision'] = int(match.group(2))
        if match.group(3):
            type_info['scale'] = int(match.group(3))
    
    return type_info


def truncate_string(text: str, max_length: int = 1000) -> str:
    """
    Truncate string to maximum length
    
    Args:
        text: Input text
        max_length: Maximum length
        
    Returns:
        str: Truncated string
    """
    if text is None:
        return ""
    text_str = str(text)
    if len(text_str) <= max_length:
        return text_str
    return text_str[:max_length] + "..."


def format_number(value: Optional[float], decimals: int = 2) -> Optional[float]:
    """
    Format a number to specified decimal places
    
    Args:
        value: Number to format
        decimals: Number of decimal places
        
    Returns:
        float or None: Formatted number
    """
    if value is None:
        return None
    try:
        if math.isnan(value) or math.isinf(value):
            return None
        return round(value, decimals)
    except (TypeError, ValueError):
        return None


def create_value_frequency_dict(values: List[tuple], total_count: int) -> List[Dict[str, Any]]:
    """
    Create frequency distribution dictionary from query results
    
    Args:
        values: List of (value, count) tuples
        total_count: Total number of records
        
    Returns:
        list: List of dictionaries with value, count, and percentage
    """
    result = []
    for value, count in values:
        result.append({
            'value': value,
            'count': int(count) if count is not None else 0,
            'percentage': safe_divide(count * 100, total_count, 0.0)
        })
    return result


def merge_dicts(*dicts: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge multiple dictionaries, with later dicts taking precedence
    
    Args:
        *dicts: Variable number of dictionaries
        
    Returns:
        dict: Merged dictionary
    """
    result = {}
    for d in dicts:
        if d:
            result.update(d)
    return result


def convert_to_sql_null(value: Any) -> str:
    """
    Convert Python value to SQL NULL representation if needed
    
    Args:
        value: Value to convert
        
    Returns:
        str: SQL value or 'NULL'
    """
    if value is None:
        return 'NULL'
    if isinstance(value, str):
        # Escape single quotes by doubling them
        escaped = value.replace("'", "''")
        return f"'{escaped}'"
    if isinstance(value, bool):
        return str(value).upper()
    if isinstance(value, (int, float)):
        if math.isnan(value) or math.isinf(value):
            return 'NULL'
        return str(value)
    # For any other type, convert to string and quote it
    return f"'{str(value).replace(chr(39), chr(39)+chr(39))}'"
