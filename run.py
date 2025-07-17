#!/usr/bin/env python3
"""
Simple test version of run.py for debugging
"""

import sys
import os
from datetime import datetime

def main():
    """Simple test function"""
    
    print("🗳️ Forsyth County Voter Analysis - Simple Test")
    print(f"Python version: {sys.version}")
    print(f"Working directory: {os.getcwd()}")
    print(f"Timestamp: {datetime.now()}")
    
    # List files in current directory
    print("\nFiles in current directory:")
    try:
        for file in os.listdir('.'):
            print(f"  📄 {file}")
    except Exception as e:
        print(f"Error listing files: {e}")
    
    # Test imports
    print("\nTesting imports...")
    
    try:
        import pandas as pd
        print(f"✅ pandas {pd.__version__}")
    except ImportError as e:
        print(f"❌ pandas: {e}")
    
    try:
        from google.cloud import bigquery
        print("✅ google-cloud-bigquery")
        
        # Test BigQuery connection
        client = bigquery.Client(project="demsncforsythcp")
        print("✅ BigQuery client created")
        
        # Simple test query
        test_query = "SELECT 1 as test"
        result = client.query(test_query).result()
        print("✅ BigQuery connection successful")
        
    except Exception as e:
        print(f"❌ BigQuery: {e}")
    
    # Test if our main script exists
    if os.path.exists('voter_movement_analyzer.py'):
        print("✅ voter_movement_analyzer.py found")
        try:
            import voter_movement_analyzer
            print("✅ voter_movement_analyzer imported successfully")
        except Exception as e:
            print(f"❌ voter_movement_analyzer import failed: {e}")
    else:
        print("❌ voter_movement_analyzer.py not found")
    
    print("\n✅ Simple test completed!")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
