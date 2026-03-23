from typing import List
from datetime import datetime

import torch

TIMESTAMP: datetime = datetime.now().strftime("%Y%m%d_%H%M%S")

# Data Ingestion Constants

ARTIFACT_DIR:str = 'artifacts'

BUCKET_NAME:str = 'chest-x-ray-images-1'

S3_DATA_FOLDER:str = 'data'
