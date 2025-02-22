# setup_db.py

from pymongo import MongoClient, ASCENDING, DESCENDING
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
from config import MONGODB_CONFIG
import sys
import time

def test_connection(client):
    """Test MongoDB connection"""
    try:
        # The ismaster command is cheap and does not require auth.
        client.admin.command('ismaster')
        return True
    except ConnectionFailure:
        return False

def setup_database():
    """Setup MongoDB collections and indexes"""
    print("Connecting to MongoDB Atlas...")
    
    try:
        # Create client with increased timeout
        client = MongoClient(
            MONGODB_CONFIG['connection_string'],
            serverSelectionTimeoutMS=10000,  # 10 second timeout
            connectTimeoutMS=20000,
            socketTimeoutMS=20000,
            ssl=True,
        )
        
        # Test connection
        if not test_connection(client):
            print("Failed to connect to MongoDB Atlas.")
            return False
            
        print("Successfully connected to MongoDB Atlas!")
        
        # Get database and collection
        db = client[MONGODB_CONFIG['database_name']]
        collection = db[MONGODB_CONFIG['collection_name']]
        
        # Create indexes
        print("Creating indexes...")
        collection.create_index([("timestamp", DESCENDING)])
        collection.create_index([("report_id", ASCENDING)], unique=True)
        
        print("\nDatabase setup completed successfully!")
        print(f"Database: {MONGODB_CONFIG['database_name']}")
        print(f"Collection: {MONGODB_CONFIG['collection_name']}")
        
        # Test write operation
        print("\nTesting write operation...")
        test_doc = {"test": "connection", "timestamp": time.time()}
        result = collection.insert_one(test_doc)
        if result.inserted_id:
            print("Write test successful!")
            # Clean up test document
            collection.delete_one({"_id": result.inserted_id})
        
        client.close()
        return True
        
    except ServerSelectionTimeoutError as e:
        print(f"\nError connecting to MongoDB Atlas: {e}")
        print("\nPlease check:")
        print("1. Your internet connection")
        print("2. The connection string in config.py")
        print("3. Your IP address is whitelisted in MongoDB Atlas")
        print("4. Your username and password are correct")
        return False
        
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        return False

if __name__ == "__main__":
    success = setup_database()
    if not success:
        sys.exit(1)  # Exit with error code
    sys.exit(0)  # Exit successfully