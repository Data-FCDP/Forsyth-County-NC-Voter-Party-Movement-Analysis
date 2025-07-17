"""
Forsyth County NC Voter Party Movement Analysis
For Google Cloud Platform - Cloud Source Repositories

This module analyzes voter party affiliation changes in Forsyth County, NC
using BigQuery data and provides comprehensive reporting and visualizations.

Author: Data Analytics Team
Created: 2025-07-16
"""

import pandas as pd
from google.cloud import bigquery
from google.cloud import storage
import os
import logging
from datetime import datetime, date
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for server environments
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Tuple, Dict, List
import json


class VoterMovementAnalyzer:
    """
    Analyzes voter party movement patterns for Forsyth County, NC
    """
    
    def __init__(self, project_id: str, dataset_id: str = "analytics_nc_split", 
                 table_name: str = "person", county: str = "FORSYTH"):
        """
        Initialize the analyzer with BigQuery connection details
        
        Args:
            project_id: Google Cloud project ID
            dataset_id: BigQuery dataset name
            table_name: BigQuery table name
            county: County name to analyze (default: FORSYTH)
        """
        self.project_id = project_id
        self.dataset_id = dataset_id
        self.table_name = table_name
        self.county = county.upper()
        
        # Initialize BigQuery client
        self.client = bigquery.Client(project=project_id)
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Party codes mapping
        self.party_codes = {
            'DEM': 'Democratic',
            'REP': 'Republican', 
            'UNA': 'Unaffiliated',
            'LIB': 'Libertarian',
            'GRE': 'Green',
            'CST': 'Constitution'
        }
        
        # Analysis metadata
        self.processing_date = datetime.now().strftime("%m/%d/%Y")
        self.analysis_timestamp = datetime.now()
        
    def get_party_name(self, party_code: str) -> str:
        """Convert party code to readable name"""
        return self.party_codes.get(party_code, party_code)
    
    def build_movement_query(self, start_date: str = "2022-01-01", 
                           end_date: str = "2022-12-31") -> str:
        """
        Build the SQL query for voter movement analysis
        
        Args:
            start_date: Start date for analysis period
            end_date: End date for analysis period
            
        Returns:
            SQL query string
        """
        
        query = f"""
        SELECT DISTINCT *
        FROM (
            SELECT 
                r1.sos_id,
                r1.county_voter_id,
                r1.county_name,
                r1.last_name,
                r1.first_name,
                r1.party_name_vf as original_party,
                r2.party_name_vf as new_party,
                r1.registration_date as original_reg_date,
                r2.registration_date as new_reg_date,
                
                -- Democratic party movements
                IF(r1.party_name_vf = 'DEM' AND r2.party_name_vf = 'REP', 'true', NULL) as dem_2_rep,
                IF(r1.party_name_vf = 'DEM' AND r2.party_name_vf = 'UNA', 'true', NULL) as dem_2_una,
                IF(r1.party_name_vf = 'DEM' AND r2.party_name_vf = 'LIB', 'true', NULL) as dem_2_lib,
                IF(r1.party_name_vf = 'DEM' AND r2.party_name_vf NOT IN ('DEM', 'REP', 'UNA', 'LIB'), 'true', NULL) as dem_2_other,
                IF(r1.party_name_vf = 'DEM' AND r2.party_name_vf NOT IN ('DEM'), 'true', NULL) as dem_2_all,
                
                -- Republican party movements
                IF(r1.party_name_vf = 'REP' AND r2.party_name_vf = 'DEM', 'true', NULL) as rep_2_dem,
                IF(r1.party_name_vf = 'REP' AND r2.party_name_vf = 'UNA', 'true', NULL) as rep_2_una,
                IF(r1.party_name_vf = 'REP' AND r2.party_name_vf = 'LIB', 'true', NULL) as rep_2_lib,
                IF(r1.party_name_vf = 'REP' AND r2.party_name_vf NOT IN ('DEM', 'REP', 'UNA', 'LIB'), 'true', NULL) as rep_2_other,
                IF(r1.party_name_vf = 'REP' AND r2.party_name_vf NOT IN ('REP'), 'true', NULL) as rep_2_all,
                
                -- Unaffiliated party movements
                IF(r1.party_name_vf = 'UNA' AND r2.party_name_vf = 'DEM', 'true', NULL) as una_2_dem,
                IF(r1.party_name_vf = 'UNA' AND r2.party_name_vf = 'REP', 'true', NULL) as una_2_rep,
                IF(r1.party_name_vf = 'UNA' AND r2.party_name_vf = 'LIB', 'true', NULL) as una_2_lib,
                IF(r1.party_name_vf = 'UNA' AND r2.party_name_vf NOT IN ('DEM', 'REP', 'UNA', 'LIB'), 'true', NULL) as una_2_other,
                IF(r1.party_name_vf = 'UNA' AND r2.party_name_vf NOT IN ('UNA'), 'true', NULL) as una_2_all,
                
                -- Libertarian party movements
                IF(r1.party_name_vf = 'LIB' AND r2.party_name_vf = 'DEM', 'true', NULL) as lib_2_dem,
                IF(r1.party_name_vf = 'LIB' AND r2.party_name_vf = 'REP', 'true', NULL) as lib_2_rep,
                IF(r1.party_name_vf = 'LIB' AND r2.party_name_vf = 'UNA', 'true', NULL) as lib_2_una,
                IF(r1.party_name_vf = 'LIB' AND r2.party_name_vf NOT IN ('DEM', 'REP', 'UNA', 'LIB'), 'true', NULL) as lib_2_other,
                IF(r1.party_name_vf = 'LIB' AND r2.party_name_vf NOT IN ('LIB'), 'true', NULL) as lib_2_all,
                
                -- Other party movements
                IF(r1.party_name_vf NOT IN ('DEM', 'REP', 'UNA', 'LIB') AND r2.party_name_vf = 'DEM', 'true', NULL) as other_2_dem,
                IF(r1.party_name_vf NOT IN ('DEM', 'REP', 'UNA', 'LIB') AND r2.party_name_vf = 'REP', 'true', NULL) as other_2_rep,
                IF(r1.party_name_vf NOT IN ('DEM', 'REP', 'UNA', 'LIB') AND r2.party_name_vf = 'UNA', 'true', NULL) as other_2_una,
                IF(r1.party_name_vf NOT IN ('DEM', 'REP', 'UNA', 'LIB') AND r2.party_name_vf = 'LIB', 'true', NULL) as other_2_lib,
                IF(r1.party_name_vf NOT IN ('DEM', 'REP', 'UNA', 'LIB') AND r2.party_name_vf IN ('DEM', 'REP', 'UNA', 'LIB'), 'true', NULL) as other_2_all
                
            FROM
                `{self.project_id}.{self.dataset_id}.{self.table_name}` AS r1
            LEFT JOIN
                `{self.project_id}.{self.dataset_id}.{self.table_name}` AS r2
            ON
                r1.sos_id = r2.sos_id
            WHERE
                r1.county_name = '{self.county}'
                AND r1.reg_voterfile_status = 'A'
                AND r2.county_name = '{self.county}'
                AND r2.reg_voterfile_status = 'A'
                AND r1.registration_date <= '{start_date}'
                AND r2.registration_date <= '{end_date}'
                AND r1.registration_date < r2.registration_date
                AND r1.party_name_vf != r2.party_name_vf
        )
        WHERE 
            dem_2_rep = 'true' OR dem_2_una = 'true' OR dem_2_lib = 'true' OR dem_2_other = 'true' OR dem_2_all = 'true'
            OR rep_2_dem = 'true' OR rep_2_una = 'true' OR rep_2_lib = 'true' OR rep_2_other = 'true' OR rep_2_all = 'true'
            OR una_2_dem = 'true' OR una_2_rep = 'true' OR una_2_lib = 'true' OR una_2_other = 'true' OR una_2_all = 'true'
            OR lib_2_dem = 'true' OR lib_2_rep = 'true' OR lib_2_una = 'true' OR lib_2_other = 'true' OR lib_2_all = 'true'
            OR other_2_dem = 'true' OR other_2_rep = 'true' OR other_2_una = 'true' OR other_2_lib = 'true' OR other_2_all = 'true'
        """
        
        return query
    
    def run_analysis(self, start_date: str = "2022-01-01", 
                    end_date: str = "2022-12-31") -> Optional[pd.DataFrame]:
        """
        Execute the voter movement analysis
        
        Args:
            start_date: Analysis start date
            end_date: Analysis end date
            
        Returns:
            DataFrame with analysis results or None if failed
        """
        try:
            self.logger.info(f"Starting {self.county} County voter movement analysis")
            self.logger.info(f"Analysis period: {start_date} to {end_date}")
            
            # Build and execute query
            query = self.build_movement_query(start_date, end_date)
            query_job = self.client.query(query)
            df = query_job.to_dataframe()
            
            self.logger.info(f"Query completed. Retrieved {len(df)} records")
            
            if df.empty:
                self.logger.warning("No movement records found")
                return None
            
            # Add readable party names
            df['original_party_name'] = df['original_party'].apply(self.get_party_name)
            df['new_party_name'] = df['new_party'].apply(self.get_party_name)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Analysis failed: {str(e)}")
            return None
    
    def generate_movement_summary(self, df: pd.DataFrame) -> Dict[str, int]:
        """
        Generate summary statistics for party movements
        
        Args:
            df: Analysis results DataFrame
            
        Returns:
            Dictionary with movement counts
        """
        summary = {}
        
        movement_columns = [col for col in df.columns if col.endswith('_2_dem') or 
                           col.endswith('_2_rep') or col.endswith('_2_una') or 
                           col.endswith('_2_lib') or col.endswith('_2_other')]
        
        for col in movement_columns:
            count = df[col].notna().sum()
            if count > 0:
                parts = col.split('_2_')
                from_party = parts[0].upper()
                to_party = parts[1].upper()
                movement_key = f"{from_party}_to_{to_party}"
                summary[movement_key] = count
        
        return summary
    
    def analyze_movement_patterns(self, df: pd.DataFrame) -> Dict[str, any]:
        """
        Perform detailed analysis of movement patterns
        
        Args:
            df: Analysis results DataFrame
            
        Returns:
            Dictionary with detailed analysis results
        """
        if df is None or df.empty:
            return {}
        
        analysis_results = {}
        
        # Most common movements
        movements = []
        for _, row in df.iterrows():
            movements.append(f"{row['original_party_name']} → {row['new_party_name']}")
        
        movement_counts = pd.Series(movements).value_counts()
        analysis_results['top_movements'] = movement_counts.head(10).to_dict()
        
        # Timeline analysis
        if 'new_reg_date' in df.columns:
            try:
                df['reg_month'] = pd.to_datetime(df['new_reg_date']).dt.to_period('M')
                monthly_counts = df['reg_month'].value_counts().sort_index()
                analysis_results['monthly_timeline'] = {
                    str(month): count for month, count in monthly_counts.items()
                }
            except Exception as e:
                self.logger.warning(f"Timeline analysis failed: {e}")
                analysis_results['monthly_timeline'] = {}
        
        # Summary statistics
        analysis_results['total_movements'] = len(df)
        analysis_results['unique_voters'] = df['sos_id'].nunique() if 'sos_id' in df.columns else 0
        analysis_results['analysis_date'] = self.processing_date
        analysis_results['county'] = self.county
        
        return analysis_results
    
    def create_visualizations(self, df: pd.DataFrame, output_dir: str = "output") -> List[str]:
        """
        Create visualizations for movement data
        
        Args:
            df: Analysis results DataFrame
            output_dir: Directory to save visualizations
            
        Returns:
            List of created file paths
        """
        if df is None or df.empty:
            return []
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        created_files = []
        
        # Set plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        try:
            # Movement summary
            movements = []
            for _, row in df.iterrows():
                movements.append(f"{row['original_party_name']} → {row['new_party_name']}")
            
            movement_counts = pd.Series(movements).value_counts()
            
            # Bar chart of top movements
            plt.figure(figsize=(12, 8))
            top_movements = movement_counts.head(10)
            bars = plt.bar(range(len(top_movements)), top_movements.values)
            plt.xticks(range(len(top_movements)), top_movements.index, rotation=45, ha='right')
            plt.title(f'Top 10 Party Movements in {self.county} County, NC (2022)', fontsize=14, fontweight='bold')
            plt.ylabel('Number of Voters', fontsize=12)
            plt.xlabel('Party Movement', fontsize=12)
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                        f'{int(height)}', ha='center', va='bottom')
            
            plt.tight_layout()
            movement_chart_path = os.path.join(output_dir, f'{self.county.lower()}_party_movements_chart.png')
            plt.savefig(movement_chart_path, dpi=300, bbox_inches='tight')
            plt.close()
            created_files.append(movement_chart_path)
            
            # Timeline visualization
            if 'new_reg_date' in df.columns:
                try:
                    df['reg_month'] = pd.to_datetime(df['new_reg_date']).dt.to_period('M')
                    monthly_counts = df['reg_month'].value_counts().sort_index()
                    
                    plt.figure(figsize=(12, 6))
                    plt.plot(range(len(monthly_counts)), monthly_counts.values, marker='o', linewidth=2, markersize=6)
                    plt.xticks(range(len(monthly_counts)), [str(m) for m in monthly_counts.index], rotation=45)
                    plt.title(f'Party Movement Timeline - {self.county} County, NC (2022)', fontsize=14, fontweight='bold')
                    plt.xlabel('Month', fontsize=12)
                    plt.ylabel('Number of Party Changes', fontsize=12)
                    plt.grid(True, alpha=0.3)
                    plt.tight_layout()
                    
                    timeline_chart_path = os.path.join(output_dir, f'{self.county.lower()}_movement_timeline.png')
                    plt.savefig(timeline_chart_path, dpi=300, bbox_inches='tight')
                    plt.close()
                    created_files.append(timeline_chart_path)
                except Exception as e:
                    self.logger.warning(f"Timeline visualization failed: {e}")
            
        except Exception as e:
            self.logger.error(f"Visualization creation failed: {e}")
        
        return created_files
    
    def save_results(self, df: pd.DataFrame, analysis_results: Dict[str, any], 
                    output_dir: str = "output") -> Dict[str, str]:
        """
        Save analysis results to files
        
        Args:
            df: Analysis results DataFrame
            analysis_results: Analysis summary dictionary
            output_dir: Output directory
            
        Returns:
            Dictionary of saved file paths
        """
        os.makedirs(output_dir, exist_ok=True)
        saved_files = {}
        
        if df is not None and not df.empty:
            # Save CSV data
            timestamp = self.analysis_timestamp.strftime('%Y%m%d_%H%M%S')
            csv_filename = f"{self.county.lower()}_voter_party_movers_{timestamp}.csv"
            csv_path = os.path.join(output_dir, csv_filename)
            
            # Reorder columns for better readability
            column_order = ['sos_id', 'county_voter_id', 'county_name', 'last_name', 'first_name', 
                           'original_party', 'original_party_name', 'new_party', 'new_party_name',
                           'original_reg_date', 'new_reg_date']
            
            movement_cols = [col for col in df.columns if '_2_' in col and col not in column_order]
            column_order.extend(movement_cols)
            available_cols = [col for col in column_order if col in df.columns]
            df_ordered = df[available_cols]
            
            df_ordered.to_csv(csv_path, index=False)
            saved_files['csv'] = csv_path
            
            # Save JSON summary
            json_filename = f"{self.county.lower()}_movement_analysis_{timestamp}.json"
            json_path = os.path.join(output_dir, json_filename)
            
            with open(json_path, 'w') as f:
                json.dump(analysis_results, f, indent=2, default=str)
            saved_files['json'] = json_path
            
            # Save text summary
            txt_filename = f"{self.county.lower()}_movement_summary_{timestamp}.txt"
            txt_path = os.path.join(output_dir, txt_filename)
            
            with open(txt_path, 'w') as f:
                f.write(f"{self.county} County NC Voter Party Movement Analysis\n")
                f.write("=" * 60 + "\n")
                f.write(f"Generated: {self.analysis_timestamp.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Total Movement Records: {len(df):,}\n")
                f.write(f"Unique Voters: {df['sos_id'].nunique():,}\n\n")
                
                if 'top_movements' in analysis_results:
                    f.write("Top Party Movements:\n")
                    f.write("-" * 30 + "\n")
                    for movement, count in analysis_results['top_movements'].items():
                        f.write(f"{movement}: {count:,} voters\n")
            
            saved_files['summary'] = txt_path
        
        return saved_files
    
    def upload_to_gcs(self, local_files: Dict[str, str], bucket_name: str, 
                     gcs_prefix: str = "") -> Dict[str, str]:
        """
        Upload results to Google Cloud Storage
        
        Args:
            local_files: Dictionary of local file paths
            bucket_name: GCS bucket name
            gcs_prefix: Optional prefix for GCS object names
            
        Returns:
            Dictionary of GCS object paths
        """
        try:
            storage_client = storage.Client(project=self.project_id)
            bucket = storage_client.bucket(bucket_name)
            
            gcs_paths = {}
            
            for file_type, local_path in local_files.items():
                if os.path.exists(local_path):
                    filename = os.path.basename(local_path)
                    gcs_object_name = f"{gcs_prefix}/{filename}" if gcs_prefix else filename
                    
                    blob = bucket.blob(gcs_object_name)
                    blob.upload_from_filename(local_path)
                    
                    gcs_paths[file_type] = f"gs://{bucket_name}/{gcs_object_name}"
                    self.logger.info(f"Uploaded {file_type} to {gcs_paths[file_type]}")
            
            return gcs_paths
            
        except Exception as e:
            self.logger.error(f"GCS upload failed: {e}")
            return {}


def main():
    """
    Main execution function for production use
    """
    # Configuration - Update these values for your environment
    config = {
        'project_id': 'demsncdurhamcp',
        'dataset_id': 'analytics_nc_split',
        'table_name': 'person',
        'county': 'FORSYTH',
        'analysis_start_date': '2022-01-01',
        'analysis_end_date': '2022-12-31',
        'output_directory': 'voter_analysis_output',
        'gcs_bucket': None,  # Set to your bucket name if you want GCS upload
        'gcs_prefix': 'voter_analysis'
    }
    
    # Initialize analyzer
    analyzer = VoterMovementAnalyzer(
        project_id=config['project_id'],
        dataset_id=config['dataset_id'],
        table_name=config['table_name'],
        county=config['county']
    )
    
    # Run analysis
    print(f"🗳️  Starting {config['county']} County voter movement analysis...")
    df = analyzer.run_analysis(
        start_date=config['analysis_start_date'],
        end_date=config['analysis_end_date']
    )
    
    if df is not None and not df.empty:
        print(f"✅ Analysis completed. Found {len(df):,} movement records")
        
        # Generate detailed analysis
        analysis_results = analyzer.analyze_movement_patterns(df)
        print(f"📊 Analyzed {analysis_results.get('total_movements', 0)} total movements")
        
        # Create visualizations
        print("📈 Creating visualizations...")
        viz_files = analyzer.create_visualizations(df, config['output_directory'])
        print(f"Created {len(viz_files)} visualization files")
        
        # Save results
        print("💾 Saving analysis results...")
        saved_files = analyzer.save_results(df, analysis_results, config['output_directory'])
        print(f"Saved {len(saved_files)} result files")
        
        # Upload to GCS if configured
        if config['gcs_bucket']:
            print(f"☁️  Uploading to GCS bucket: {config['gcs_bucket']}")
            all_files = {**saved_files}
            for i, viz_file in enumerate(viz_files):
                all_files[f'visualization_{i}'] = viz_file
            
            gcs_paths = analyzer.upload_to_gcs(
                all_files, 
                config['gcs_bucket'], 
                config['gcs_prefix']
            )
            print(f"Uploaded {len(gcs_paths)} files to GCS")
        
        # Print summary
        print("\n" + "="*60)
        print(f"🎉 Analysis Complete - {config['county']} County, NC")
        print("="*60)
        print(f"Total party movements: {analysis_results.get('total_movements', 0):,}")
        print(f"Unique voters affected: {analysis_results.get('unique_voters', 0):,}")
        print(f"Output directory: {config['output_directory']}")
        
        if 'top_movements' in analysis_results:
            print("\nTop 5 movement types:")
            for i, (movement, count) in enumerate(list(analysis_results['top_movements'].items())[:5], 1):
                print(f"  {i}. {movement}: {count:,} voters")
        
        return True
    else:
        print("❌ Analysis failed or returned no results")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
