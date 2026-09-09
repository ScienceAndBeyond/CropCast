# Keeps the configuration parameters used across programs
import logging
import os
from pathlib import Path

from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

DATA_PATH = Path("../data")

# API Endpoints
NASS_BASE_URL = "https://quickstats.nass.usda.gov/api/api_GET/"

load_dotenv()
# API Keys
GEE_PROJECT_ID=os.getenv("GEE_PROJECT_ID")  
NASS_API_KEY = os.getenv("NASS_API_KEY")
