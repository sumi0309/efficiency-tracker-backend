from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
from datetime import datetime
import json
from typing import Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MongoDBClient:
    def __init__(self, connection_string: str, database_name: str = 'construction_analytics', 
                 collection_name: str = 'reports'):
        """Initialize MongoDB client with proper error handling"""
        try:
            self.client = MongoClient(
                connection_string,
                serverSelectionTimeoutMS=5000,
                connectTimeoutMS=10000,
                socketTimeoutMS=10000
            )
            # Test connection
            self.client.admin.command('ping')
            logger.info("Successfully connected to MongoDB")
            
            self.db = self.client[database_name]
            self.reports_collection = self.db[collection_name]
            
        except ConnectionFailure as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error while connecting to MongoDB: {e}")
            raise

    def upload_report(self, report_data: Dict[str, Any]) -> str:
        """Upload report to MongoDB Atlas with error handling"""
        try:
            # Add metadata
            report_data['metadata'] = {
                'timestamp': datetime.now().isoformat(),
                'report_id': f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                'version': '1.0'
            }
            
            # Convert any non-serializable objects to strings
            serialized_data = json.loads(json.dumps(report_data, default=str))
            
            # Insert into MongoDB
            result = self.reports_collection.insert_one(serialized_data)
            logger.info(f"Successfully uploaded report with ID: {result.inserted_id}")
            
            return str(result.inserted_id)
            
        except Exception as e:
            logger.error(f"Error uploading report to MongoDB: {e}")
            raise

    def get_report(self, report_id: str) -> Dict[str, Any]:
        """Retrieve a specific report"""
        try:
            return self.reports_collection.find_one({"metadata.report_id": report_id})
        except Exception as e:
            logger.error(f"Error retrieving report {report_id}: {e}")
            raise

    def get_all_reports(self) -> list:
        """Get all reports"""
        try:
            return list(self.reports_collection.find({}, {'_id': 0}))
        except Exception as e:
            logger.error(f"Error retrieving all reports: {e}")
            raise

    def get_latest_report(self) -> Dict[str, Any]:
        """Get the most recent report"""
        try:
            return self.reports_collection.find_one(
                {},
                sort=[('metadata.timestamp', -1)]
            )
        except Exception as e:
            logger.error(f"Error retrieving latest report: {e}")
            raise