# api.py

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
import uvicorn
from datetime import datetime
import json
from fastapi.middleware.cors import CORSMiddleware
import os
from typing import Dict, Optional
from dotenv import load_dotenv
from db_utils import MongoDBClient

# Import our existing analytics code
from run_analytics import main as run_analysis

load_dotenv()

# Initialize MongoDB client
mongodb_client = MongoDBClient(
    connection_string=os.getenv('MONGODB_URI'),
    database_name=os.getenv('MONGODB_DATABASE', 'construction_analytics'),
    collection_name=os.getenv('MONGODB_COLLECTION', 'reports')
)

app = FastAPI(
    title="Construction Analytics API",
    description="API for running construction site analytics including video analysis",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with your frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Store for analysis status
analysis_status = {
    "last_run": None,
    "is_running": False,
    "last_error": None,
    "last_report_id": None
}

async def run_analysis_task():
    """Background task to run the analysis"""
    try:
        analysis_status["is_running"] = True
        analysis_status["last_error"] = None
        
        # Run the analysis
        dashboard_data = run_analysis()
        
        # Save locally
        with open('dashboard_output.json', 'w') as f:
            json.dump(dashboard_data, f, indent=2)
        
        # Upload to MongoDB
        report_id = mongodb_client.upload_report(dashboard_data)
        analysis_status["last_report_id"] = report_id
        
        analysis_status["last_run"] = datetime.now().isoformat()
        analysis_status["is_running"] = False
        
    except Exception as e:
        analysis_status["last_error"] = str(e)
        analysis_status["is_running"] = False
        raise

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "name": "Construction Analytics API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": [
            "/run-analysis",
            "/status",
            "/results",
            "/reports"
        ]
    }

@app.post("/run-analysis")
async def trigger_analysis(background_tasks: BackgroundTasks):
    """Trigger a new analysis run"""
    if analysis_status["is_running"]:
        raise HTTPException(
            status_code=409,
            detail="Analysis is already running"
        )
    
    # Check if required files exist
    required_files = [
        'construction_project_data.csv',
        'equipment_tracking.csv',
        'iot_sensor_data.csv',
        'material_management_data.csv',
        'mobile_activity_data.csv',
        'project_management_data.csv'
    ]
    
    missing_files = [f for f in required_files if not os.path.exists(f)]
    if missing_files:
        raise HTTPException(
            status_code=400,
            detail=f"Missing required files: {', '.join(missing_files)}"
        )
    
    # Start analysis in background
    background_tasks.add_task(run_analysis_task)
    
    return {
        "status": "Analysis started",
        "message": "Analysis is running in the background"
    }

@app.get("/status")
async def get_status():
    """Get the current status of the analysis"""
    return {
        "is_running": analysis_status["is_running"],
        "last_run": analysis_status["last_run"],
        "last_error": analysis_status["last_error"],
        "last_report_id": analysis_status["last_report_id"]
    }

@app.get("/results")
async def get_results():
    """Get the latest analysis results"""
    try:
        # Get latest result from MongoDB
        latest_report = mongodb_client.get_latest_report()
        if latest_report:
            return latest_report
        
        # Fallback to local file if MongoDB is empty
        with open('dashboard_output.json', 'r') as f:
            results = json.load(f)
        return results
    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail="No analysis results found. Run analysis first."
        )

@app.get("/reports")
async def get_all_reports():
    """Get all reports from MongoDB"""
    return mongodb_client.get_all_reports()


if __name__ == "__main__":
    uvicorn.run(
        "api:app", 
        host="0.0.0.0", 
        port=8000, 
        reload=False 
    )