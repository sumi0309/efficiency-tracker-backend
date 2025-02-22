# api.py

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
import uvicorn
from datetime import datetime
import json
import os
from typing import Dict, Optional

# Import our existing analytics code
from run_analytics import main as run_analysis

app = FastAPI(
    title="Construction Analytics API",
    description="API for running construction site analytics including video analysis",
    version="1.0.0"
)

# Store for analysis results and status
analysis_status = {
    "last_run": None,
    "is_running": False,
    "last_error": None
}

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
            "/results"
        ]
    }

async def run_analysis_task():
    """Background task to run the analysis"""
    try:
        analysis_status["is_running"] = True
        analysis_status["last_error"] = None
        
        # Run the analysis
        dashboard_data = run_analysis()
        
        # Save results
        with open('dashboard_output.json', 'w') as f:
            json.dump(dashboard_data, f, indent=2)
        
        analysis_status["last_run"] = datetime.now().isoformat()
        analysis_status["is_running"] = False
        
    except Exception as e:
        analysis_status["last_error"] = str(e)
        analysis_status["is_running"] = False
        raise

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
        "last_error": analysis_status["last_error"]
    }

@app.get("/results")
async def get_results():
    """Get the latest analysis results"""
    try:
        with open('dashboard_output.json', 'r') as f:
            results = json.load(f)
        return results
    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail="No analysis results found. Run analysis first."
        )

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)