import torch
from loss import STANetLoss
from torch.utils.data import DataLoader
from utils import *
from data.dataset import VideoQualityDataset
import argparse

# Argument parsing
parser = argparse.ArgumentParser(description="Training script for STANet")
parser.add_argument('--device', type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use for training (cuda/cpu).")
parser.add_argument('--save_path', type=str, default="./exp/stanet/", help="Path to save the trained models.")
parser.add_argument('--json_path', type=str, default='./VMAFplus/subj_score.json', help="Path to the JSON file for the VideoQualityDataset.")
parser.add_argument('--batch_size', type=int, default=1, help="Batch size for training.")
parser.add_argument('--pretrained_model_path', type=str, default='./exp/FR_model', help="Path to the pretrained model for the Extractor.")

args = parser.parse_args()

def getscore(dataloader, device):
    score1 = []
    score2 = []
    Feature_layer_combine_1 = []
    Feature_layer_combine_2 = []
    global dis_path_1, dis_path_2
    for dis_path_1, dis_path_2, patch_quality_score1, patch_quality_score2, Feature_layer_combine1, Feature_layer_combine2 in dataloader:
        score1.append(patch_quality_score1)
        score2.append(patch_quality_score2)
        Feature_layer_combine_1.append(Feature_layer_combine1)
        Feature_layer_combine_2.append(Feature_layer_combine2)
    return dis_path_1, dis_path_2


device = torch.device(args.device)
save_path = args.save_path

dataset = VideoQualityDataset(json_path=args.json_path)
dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

# Initialize models and optimizer
for mos1, mos2, patch_quality_score1, patch_quality_score2,Feature_layer_combine1,Feature_layer_combine2 in dataloader:
    data_to_save = {
        'patch_quality_score1': patch_quality_score1,
        'patch_quality_score2': patch_quality_score2,
        'mos1': mos1,
        'mos2': mos2,
        'Feature_layer_combine_1': Feature_layer_combine1,
        'Feature_layer_combine_2': Feature_layer_combine2
    }
    # Save to the same file
    with open('./data_VMAFplus.pkl', 'ab') as f:  # 'ab' mode means append in binary mode
        pickle.dump(data_to_save, f)