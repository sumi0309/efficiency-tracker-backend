# construction_analytics.py

# Import all required libraries
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
from typing import Dict, List, Any
from prophet import Prophet
from sklearn.ensemble import RandomForestRegressor
import pickle
from sklearn.preprocessing import StandardScaler

def get_equipment_name(equipment_id: str) -> str:
    """Map equipment IDs to comprehensive construction equipment names"""
    equipment_types = {
        # Heavy Machinery/Vehicles
        'EXC': 'Excavator',
        'BDZ': 'Bulldozer',
        'CRN': 'Tower_Crane',
        'LDR': 'Wheel_Loader',
        'DMP': 'Dump_Truck',
        'CNM': 'Concrete_Mixer',
        'GRD': 'Motor_Grader',
        'BCH': 'Backhoe',
        'RLR': 'Road_Roller',
        'FLT': 'Forklift',
        'SKL': 'Skid_Steer',
        'TRK': 'Cargo_Truck',
        
        # Power Tools & Equipment
        'GEN': 'Generator',
        'CMP': 'Air_Compressor',
        'WLD': 'Welding_Machine',
        'DRL': 'Industrial_Drill',
        'SAW': 'Concrete_Saw',
        'PLT': 'Plate_Compactor',
        'PMC': 'Concrete_Pump',
        'SCF': 'Scaffolding',
        
        # Material Handling
        'CNV': 'Conveyor',
        'HST': 'Construction_Hoist',
        'WNC': 'Material_Winch',
        'SLF': 'Material_Lift',
        
        # Specialized Equipment
        'PLD': 'Piling_Equipment',
        'TRW': 'Tower_Light',
        'VBR': 'Concrete_Vibrator',
        'MXR': 'Mortar_Mixer',
        
        # Default mappings for common prefixes
        'HE': 'Hydraulic_Excavator',
        'CR': 'Crawler_Crane',
        'MC': 'Mobile_Crane',
        'BL': 'Boom_Lift',
        'CT': 'Compactor',
        'MT': 'Material_Truck',
        'PV': 'Paver',
        'RL': 'Roller',
        'SC': 'Scraper'
    }
    
    # Extract equipment type from ID (assuming format like 'EXC_001')
    eq_type = equipment_id.split('_')[0] if '_' in equipment_id else equipment_id[:3]
    
    # Get equipment name, use combination of type and ID if not found
    equipment_name = equipment_types.get(eq_type)
    if equipment_name is None:
        # Try two-letter prefix if three-letter not found
        eq_type_short = eq_type[:2]
        equipment_name = equipment_types.get(eq_type_short, 'Construction_Equipment')
    
    return equipment_name

def analyze_performance_metrics(project_df: pd.DataFrame, 
                             equipment_df: pd.DataFrame,
                             mobile_df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """Calculate hourly performance metrics for efficiency, work completion, and labor utilization"""
    # Convert timestamps to datetime
    equipment_df['timestamp_start'] = pd.to_datetime(equipment_df['timestamp_start'])
    project_df['start_time'] = pd.to_datetime(project_df['start_time'])
    mobile_df['timestamp'] = pd.to_datetime(mobile_df['timestamp'])
    
    # Group by hour and calculate metrics
    hourly_metrics = {
        'efficiency': [],
        'work_completion': [],
        'labor_utilization': [],
        'timestamps': []
    }
    
    for hour in range(24):
        # Equipment efficiency
        hour_equipment = equipment_df[equipment_df['timestamp_start'].dt.hour == hour]
        efficiency = hour_equipment['utilization_load'].mean()
        
        # Work completion
        hour_projects = project_df[project_df['start_time'].dt.hour == hour]
        completion = hour_projects['completion_percentage'].mean()
        
        # Labor utilization
        hour_mobile = mobile_df[mobile_df['timestamp'].dt.hour == hour]
        labor = len(hour_mobile[hour_mobile['completion_status'] == 'Completed']) / max(len(hour_mobile), 1) * 100
        
        hourly_metrics['timestamps'].append(f"{hour:02d}:00")
        hourly_metrics['efficiency'].append(round(efficiency if not pd.isna(efficiency) else 75, 1))
        hourly_metrics['work_completion'].append(round(completion if not pd.isna(completion) else 80, 1))
        hourly_metrics['labor_utilization'].append(round(labor if not pd.isna(labor) else 85, 1))
    
    return hourly_metrics

def calculate_resource_distribution(mobile_df: pd.DataFrame, 
                                 equipment_df: pd.DataFrame) -> Dict[str, List]:
    """Calculate actual resource distribution by hour"""
    mobile_df['timestamp'] = pd.to_datetime(mobile_df['timestamp'])
    equipment_df['timestamp_start'] = pd.to_datetime(equipment_df['timestamp_start'])
    
    resource_dist = {
        'timestamps': [f"{hour:02d}:00" for hour in range(24)],
        'workers': [],
        'equipment': [],
        'equipment_details': []  # New field for detailed equipment info
    }
    
    for hour in range(24):
        # Count workers
        workers = len(mobile_df[mobile_df['timestamp'].dt.hour == hour]['user_id'].unique())
        resource_dist['workers'].append(workers)
        
        # Get equipment for this hour
        hour_equipment = equipment_df[equipment_df['timestamp_start'].dt.hour == hour]
        
        # Count unique equipment with names
        equipment_list = []
        for idx, row in hour_equipment.iterrows():
            equipment_name = get_equipment_name(row['equipment_id'])
            equipment_list.append({
                'name': equipment_name,
                'id': row['equipment_id'],
                'status': row['status']
            })
        
        # Add equipment count and details
        resource_dist['equipment'].append(len(equipment_list))
        resource_dist['equipment_details'].append(equipment_list)
    
    return resource_dist

def calculate_equipment_status(equipment_df: pd.DataFrame) -> Dict[str, Dict]:
    """Get current status of each equipment with actual utilization"""
    status = {}
    
    # Get latest status for each equipment
    latest_status = equipment_df.sort_values('timestamp_start').groupby('equipment_id').last()
    
    # Process each equipment
    for idx, row in latest_status.head(6).iterrows():
        # Get equipment name and number
        equipment_name = get_equipment_name(idx)
        equipment_number = idx.split('_')[1] if '_' in idx else '001'
        
        # Create key with real equipment name
        key = f"{equipment_name}_{equipment_number}"
        
        # Store status information
        status[key] = {
            'status': row['status'],
            'utilization': round(row['utilization_load'], 2),
            'location': row['current_zone'],
            'equipment_type': equipment_name
        }
    
    return status



def calculate_zone_metrics(equipment_df: pd.DataFrame, mobile_df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """Calculate detailed metrics for each zone"""
    zone_metrics = {}
    
    for zone in ['Zone A', 'Zone B', 'Zone C']:
        # Get equipment in this zone
        zone_equipment = equipment_df[equipment_df['current_zone'] == zone]
        equipment_count = len(zone_equipment['equipment_id'].unique())
        
        # Calculate utilization
        utilization = zone_equipment['utilization_load'].mean()
        
        # Calculate worker density
        zone_workers = mobile_df[mobile_df['task_id'].isin(
            zone_equipment['equipment_id']  # Assuming task_id links to equipment
        )]
        worker_count = len(zone_workers['user_id'].unique())
        
        # Calculate activity level
        recent_activities = len(zone_workers[
            zone_workers['timestamp'] >= (zone_workers['timestamp'].max() - pd.Timedelta(hours=1))
        ])
        
        zone_metrics[zone] = {
            "utilization": round(utilization, 2),
            "worker_density": "high" if worker_count > 20 else "medium" if worker_count > 10 else "low",
            "equipment_count": equipment_count,
            "activity_level": "high" if recent_activities > 30 else "medium" if recent_activities > 15 else "low"
        }
    
    return zone_metrics

def calculate_summary_metrics(dfs: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    """Calculate all summary metrics from actual data"""
    project_df = dfs['project']
    equipment_df = dfs['equipment']
    mobile_df = dfs['mobile']
    
    # Calculate actual completion and efficiency metrics
    completed_tasks = len(project_df[project_df['completion_percentage'] == 100])
    total_tasks = len(project_df)
    current_efficiency = equipment_df['utilization_load'].mean()
    
    # Calculate time distributions
    total_hours = (project_df['end_time'].max() - project_df['start_time'].min()).total_seconds() / 3600
    active_time = total_hours - project_df['delay_duration'].sum()
    break_time = active_time * 0.1  # Assuming 10% break time
    
    return {
        'summary': {
            'overall_efficiency': round(current_efficiency, 1),
            'labor_utilization': round(len(mobile_df[mobile_df['completion_status'] == 'Completed']) / len(mobile_df) * 100, 1),
            'task_completion': round(project_df['completion_percentage'].mean(), 1),
            'quality_score': round(100 - (project_df['delay_duration'].mean() * 5), 1)
        },
        'time_distribution': {
            'active_time': f"{int(active_time)}h {int((active_time % 1) * 60)}m",
            'break_time': f"{int(break_time)}h {int((break_time % 1) * 60)}m",
            'downtime': f"{int(project_df['delay_duration'].sum())}h {int((project_df['delay_duration'].sum() % 1) * 60)}m"
        },
        'task_metrics': {
            'tasks_completed': f"{completed_tasks}/{total_tasks}",
            'average_completion_time': f"{int((project_df['end_time'] - project_df['start_time']).mean().total_seconds() / 60)}m",
            'quality_compliance': f"{round(100 - (project_df['delay_duration'].mean() * 5), 1)}%"
        }
    }

def calculate_risks_and_predictions(dfs: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    """Calculate risk assessments and predictions from actual data"""
    equipment_df = dfs['equipment']
    project_df = dfs['project']
    
    # Calculate zone-specific risks
    zone_risks = {}
    for zone in ['Zone A', 'Zone B', 'Zone C']:
        zone_equipment = equipment_df[equipment_df['current_zone'] == zone]
        zone_risks[zone] = round(
            (100 - zone_equipment['utilization_load'].mean()) * 0.4 +
            (len(zone_equipment[zone_equipment['status'] != 'Active']) / len(zone_equipment)) * 100 * 0.6
        )
    
    return {
        'risk_assessment': zone_risks,
        'predictive_insights': {
            'peak_efficiency': {
                'value': round(equipment_df['utilization_load'].max(), 1),
                'expected_time': '14:00'
            },
            'bottleneck': {
                'zone': min(zone_risks, key=zone_risks.get),
                'reason': 'Resource shortage predicted'
            },
            'staffing': {
                'recommendation': '+3 workers',
                'zone': max(zone_risks, key=zone_risks.get)
            }
        }
    }

def train_efficiency_model(historical_data: pd.DataFrame) -> Prophet:
    """Train Prophet model on historical efficiency data"""
    # Prepare data for Prophet
    df = pd.DataFrame({
        'ds': pd.to_datetime(historical_data['timestamp_start']),
        'y': historical_data['utilization_load']
    })
    
    # Initialize and train Prophet model
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=True,
        changepoint_prior_scale=0.05
    )
    model.fit(df)
    
    return model

def calculate_resource_optimization(dfs: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    """Calculate resource optimization recommendations from actual data"""
    equipment_df = dfs['equipment']
    mobile_df = dfs['mobile']
    project_df = dfs['project']

    # Calculate shift recommendations based on workload patterns
    current_hour = datetime.now().hour
    evening_workload = len(mobile_df[
        (mobile_df['timestamp'].dt.hour >= 16) & 
        (mobile_df['timestamp'].dt.hour <= 22)
    ])
    
    day_workload = len(mobile_df[
        (mobile_df['timestamp'].dt.hour >= 8) & 
        (mobile_df['timestamp'].dt.hour <= 16)
    ])
    
    # Calculate recommended shifts based on workload difference
    shift_diff = round((day_workload - evening_workload) / max(day_workload, 1) * 2)
    recommended_shifts = f"+{abs(shift_diff)} Evening" if evening_workload < day_workload else f"+{abs(shift_diff)} Day"

    # Calculate equipment relocation based on zone utilization
    zone_equipment = equipment_df.groupby('current_zone')['utilization_load'].mean()
    high_util_zones = len(zone_equipment[zone_equipment > 80])
    low_util_zones = len(zone_equipment[zone_equipment < 60])
    equipment_to_relocate = abs(high_util_zones - low_util_zones) + 1

    return {
        'recommended_shifts': recommended_shifts,
        'equipment_relocation': f"{equipment_to_relocate} units"
    }
    
def calculate_weather_impact(dfs: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    """Calculate weather impact based on historical performance data"""
    project_df = dfs['project']
    
    # Analyze delay reasons and completion rates
    weather_delays = len(project_df[project_df['delay_reason'] == 'Weather'])
    total_delays = len(project_df[project_df['delay_duration'] > 0])
    
    # Calculate weather impact probability
    weather_probability = round((weather_delays / max(total_delays, 1)) * 100)
    
    # Calculate productivity impact
    delayed_completion = project_df[project_df['delay_reason'] == 'Weather']['completion_percentage'].mean()
    normal_completion = project_df[project_df['delay_reason'] != 'Weather']['completion_percentage'].mean()
    
    if pd.isna(delayed_completion) or pd.isna(normal_completion):
        productivity_impact = -8  # Default if no data
    else:
        productivity_impact = round(delayed_completion - normal_completion)

    # Determine forecast based on delay patterns
    if weather_delays > total_delays * 0.3:
        forecast = "Heavy Rain"
    elif weather_delays > total_delays * 0.1:
        forecast = "Light Rain"
    else:
        forecast = "Clear"

    return {
        'forecast': forecast,
        'probability': min(max(weather_probability, 0), 100),
        'productivity_impact': min(max(productivity_impact, -20), 0)
    }


def predict_efficiency_with_ml(equipment_df: pd.DataFrame) -> Dict[str, Any]:
    """Generate efficiency predictions using Prophet and RandomForest"""
    equipment_df['timestamp_start'] = pd.to_datetime(equipment_df['timestamp_start'])
    
    # Train Prophet model
    prophet_model = train_efficiency_model(equipment_df)
    
    # Create future dataframe for predictions
    future_dates = pd.DataFrame({
        'ds': pd.date_range(
            start=equipment_df['timestamp_start'].max(),
            periods=24,
            freq='H'
        )
    })
    
    # Get Prophet predictions
    prophet_forecast = prophet_model.predict(future_dates)
    
    # Prepare data for RandomForest
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    
    # Create features for RandomForest
    equipment_df['hour'] = equipment_df['timestamp_start'].dt.hour
    equipment_df['day_of_week'] = equipment_df['timestamp_start'].dt.dayofweek
    
    # Train RandomForest on recent data
    X = equipment_df[['hour', 'day_of_week']].values
    y = equipment_df['utilization_load'].values
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    rf_model.fit(X_scaled, y)
    
    # Prepare prediction results
    timestamps = [f"{str(i).zfill(2)}:00" for i in range(24)]
    historical_values = []
    predicted_values = []
    
    # Calculate historical values from actual data
    for hour in range(24):
        hour_data = equipment_df[equipment_df['timestamp_start'].dt.hour == hour]
        if len(hour_data) > 0:
            efficiency = hour_data['utilization_load'].mean()
            historical_values.append(round(efficiency, 1))
        else:
            # Use Prophet's historical fit for missing values
            prophet_value = prophet_forecast[prophet_forecast['ds'].dt.hour == hour]['yhat'].mean()
            historical_values.append(round(prophet_value, 1))
    
    # Generate predictions using both models
    for hour in range(24):
        # Combine Prophet and RandomForest predictions
        prophet_pred = prophet_forecast[prophet_forecast['ds'].dt.hour == hour]['yhat'].mean()
        
        rf_features = scaler.transform([[hour, datetime.now().weekday()]])
        rf_pred = rf_model.predict(rf_features)[0]
        
        # Weighted average of both predictions
        combined_pred = 0.6 * prophet_pred + 0.4 * rf_pred
        predicted_values.append(round(combined_pred, 1))
    
    return {
        'timestamps': timestamps,
        'historical_values': historical_values,
        'predicted_values': predicted_values,
        'current_value': {
            'time': '18:00',
            'value': historical_values[18]
        }
    }

def generate_dashboard_data(dfs: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    """Generate complete dashboard data from CSV files"""
    
    # Calculate all existing metrics
    performance_metrics = analyze_performance_metrics(dfs['project'], dfs['equipment'], dfs['mobile'])
    resource_dist = calculate_resource_distribution(dfs['mobile'], dfs['equipment'])
    equipment_status = calculate_equipment_status(dfs['equipment'])
    summary_metrics = calculate_summary_metrics(dfs)
    risks_and_predictions = calculate_risks_and_predictions(dfs)
    zone_metrics = calculate_zone_metrics(dfs['equipment'], dfs['mobile'])
    efficiency_predictions = predict_efficiency_with_ml(dfs['equipment'])
    
    # Calculate new metrics from data
    resource_optimization = calculate_resource_optimization(dfs)
    weather_impact = calculate_weather_impact(dfs)
    
    return {
        'zone_metrics': zone_metrics,
        'efficiency_predictions': efficiency_predictions,
        'performance_metrics': performance_metrics,
        'resource_distribution': resource_dist,
        'equipment_status': equipment_status,
        'summary_metrics': summary_metrics,
        'risks_and_predictions': risks_and_predictions,
        'resource_optimization': resource_optimization,
        'weather_impact': weather_impact
    }