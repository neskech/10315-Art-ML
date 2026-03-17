import os
from pathlib import Path
import pandas as pd
import torch
import torch.nn.functional as F
from pose_module.interpret_mhr_params import PoseData

CURRENT_DIR = Path(__file__).parent.resolve()
DATA_PATH = CURRENT_DIR.parent / "data"

def getClassificationFeatureMetric():
    parquet_path = os.path.join(DATA_PATH, "classification_features.parquet")
    df = pd.read_parquet(parquet_path)
    
    feature_map = dict(zip(df['image_path'], df['classification_features']))

    def classificationFeatureMetric(pose1: PoseData, pose2: PoseData):
        feat1 = feature_map.get(pose1.relative_image_path)
        feat2 = feature_map.get(pose2.relative_image_path)
        

        t1 = torch.tensor(feat1).float()
        t2 = torch.tensor(feat2).float()
        
        sim = F.cosine_similarity(t1.unsqueeze(0), t2.unsqueeze(0))
        return (1.0 - sim.item())

    return classificationFeatureMetric