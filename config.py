# config.py

# MongoDB Configuration
MONGODB_CONFIG = {
    # Replace with your actual values from MongoDB Atlas
    'connection_string': 'mongodb+srv://dkoushik:dkoushik@efficiency-tracker.s6yak.mongodb.net/?retryWrites=true&w=majority&appName=efficiency-tracker',
    'database_name': 'construction_analytics',
    'collection_name': 'reports'
}

# API Configuration
API_CONFIG = {
    'host': '0.0.0.0',
    'port': 8000,
    'debug': True
}