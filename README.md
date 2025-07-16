# Forsyth-County-NC-Voter-Party-Movement-Analysis
This repository contains code for analyzing voter party affiliation changes in Forsyth County, North Carolina using Google BigQuery data.
Overview
The analysis identifies voters who changed their party affiliation during a specified time period (default: 2022) and provides comprehensive reporting including:

Summary statistics of party movements
Detailed breakdowns by movement type (DEM→REP, REP→UNA, etc.)
Timeline analysis showing movement patterns over time
Visualizations and charts
Export capabilities to CSV, JSON, and Cloud Storage

Features

🗳️ Comprehensive Analysis: Tracks all major party movements including Democratic, Republican, Unaffiliated, Libertarian, and other parties
📊 Rich Visualizations: Generates charts showing top movements and timeline trends
💾 Multiple Output Formats: Saves results as CSV, JSON summary, and text reports
☁️ Cloud Integration: Optional upload to Google Cloud Storage
🔧 Configurable: Easy configuration through YAML file
📈 Production Ready: Robust error handling and logging

Prerequisites

Google Cloud Platform account with BigQuery access
Voter registration data in BigQuery
Python 3.8+ environment
Required packages (see requirements.txt
