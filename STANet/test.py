import torch
from networks.network import STANet
from data.dataset_test import VideoQualityDatasettest
from networks.multi_scale import Extractor
from torch.utils.data import DataLoader
import argparse
import os
from scipy.stats import spearmanr, pearsonr
import numpy as np

# Argument parsing
parser = argparse.ArgumentParser(description="Testing script for STANet")
parser.add_argument('--model_path', type=str, required=True, help="Path to the trained model.")
parser.add_argument('--json_path', type=str, nargs='+', required=True, help="List of paths to JSON files for testing datasets.")
parser.add_argument('--device', type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use for testing (cuda/cpu).")
parser.add_argument('--batch_size', type=int, default=1, help="Batch size for testing.")
args = parser.parse_args()

def test(model, dataloader, device):
    model.eval()
    predictions, ground_truths = [], []
    with torch.no_grad():
        for Feature_layer_combine1, mos1, patch_quality_score1 in dataloader:
            mos1 = mos1.float().to(device)
            clip_num = len(patch_quality_score1) // (10 * 9 * 16)
            stanet_output = []
            scores1 = torch.cat(patch_quality_score1, dim=0).to(device)
            for i in range(clip_num):
                start_idx = i * (10 * 9 * 16)
                end_idx = start_idx + (10 * 9 * 16)
                scores1_segment = scores1[start_idx:end_idx]
                Feature_layer_combine1_segment = Feature_layer_combine1[start_idx:end_idx]
                stanet_output1 = model(scores1_segment, Feature_layer_combine1_segment)
                stanet_output.append(stanet_output1)
            stanet_output_np = [output.cpu().numpy() for output in stanet_output]
            
            mean_score = np.mean(np.concatenate(stanet_output_np))
            print(mean_score)

            predictions.extend([mean_score])
            ground_truths.extend(mos1.cpu().numpy())


    # Calculate metrics
    srocc, _ = spearmanr(predictions, ground_truths)
    plcc, _ = pearsonr(predictions, ground_truths)
    rmse = np.sqrt(((np.array(predictions) - np.array(ground_truths)) ** 2).mean())

    return srocc, plcc, rmse

# Initialize model and load trained weights

stanet = STANet().to(args.device)
stanet.load_state_dict(torch.load(args.model_path))
stanet.eval()

# Test the model on multiple datasets
for json_path in args.json_path:
    dataset = VideoQualityDatasettest(json_path=json_path)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    srocc, plcc, rmse = test(stanet, dataloader, args.device)
    print(f"Testing on {os.path.basename(json_path)}, SROCC: {srocc:.4f}, PLCC: {plcc:.4f}, RMSE: {rmse:.4f}")