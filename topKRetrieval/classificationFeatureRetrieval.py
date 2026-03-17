import os
from pathlib import Path
import pandas as pd
import torch
import torch.nn.functional as F
from classification_features.inference import PoseClassificationFeatures
from pose_module.interpret_mhr_params import PoseData

CURRENT_DIR = Path(__file__).parent.resolve()
DATA_PATH = CURRENT_DIR.parent / "data"

def getClassificationFeatureMetric():
    parquet_path = os.path.join(DATA_PATH, "classification_features.parquet")
    df = pd.read_parquet(parquet_path)
    
    feature_map = dict(zip(df['image_path'], df['classification_features']))

    extractor = PoseClassificationFeatures()
    cached_feats = {}

    def classificationFeatureMetric(pose1: PoseData, pose2: PoseData):
        feat1 = feature_map.get(pose1.relative_image_path)
        feat2 = feature_map.get(pose2.relative_image_path)

        if feat1 is None:
            if pose1.relative_image_path in cached_feats:
                feat1 = cached_feats[pose1.relative_image_path]
            else:
                feat1 = extractor.extract_embedding(pose1.pred_keypoints_2d_dict)
                cached_feats[pose1.relative_image_path] = feat1
        elif feat2 is None:
            if pose2.relative_image_path in cached_feats:
                feat2 = cached_feats[pose2.relative_image_path]
            else:
                feat2 = extractor.extract_embedding(pose2.pred_keypoints_2d_dict)
                cached_feats[pose2.relative_image_path] = feat2
        

        t1 = torch.tensor(feat1).float()
        t2 = torch.tensor(feat2).float()
        
        sim = F.cosine_similarity(t1.unsqueeze(0), t2.unsqueeze(0))
        return (1.0 - sim.item())

    return classificationFeatureMetric