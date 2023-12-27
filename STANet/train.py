import torch
import torch.nn as nn
from networks.network import STANet
from loss import STANetLoss
import torch.optim as optim
from utils import *
import argparse
import torch.nn.init as init

# Argument parsing
parser = argparse.ArgumentParser(description="Training script for STANet")
parser.add_argument('--learning_rate', type=float, default=0.001, help="Learning rate for the optimizer.")
parser.add_argument('--epochs', type=int, default=3000, help="Number of epochs to train.")
parser.add_argument('--device', type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use for training (cuda/cpu).")
parser.add_argument('--save_path', type=str, default="./exp/stanet/", help="Path to save the trained models.")
parser.add_argument('--json_path', type=str, default='path_to_json_file.json', help="Path to the JSON file for the VideoQualityDataset.")
parser.add_argument('--batch_size', type=int, default=1, help="Batch size for training.")
parser.add_argument('--pretrained_model_path', type=str, default='./exp/FR_model', help="Path to the pretrained model for the Extractor.")

args = parser.parse_args()

def he_initialization(layer):
    if isinstance(layer, (nn.Conv2d, nn.Conv3d, nn.Linear)):
        init.kaiming_normal_(layer.weight, nonlinearity='relu')
        if layer.bias is not None:
            init.constant_(layer.bias, 0)


def load_pretrained_weights(extractor, path, device):

    pretrained_state_dict = torch.load(path, map_location=device)

    extractor_state_dict = {k.replace('net.moduleExtractor.', ''): v for k, v in pretrained_state_dict.items() if 'net.moduleExtractor' in k}

    extractor = extractor.load_state_dict(extractor_state_dict, strict=False)
    print("Loaded pretrained weights for Extractor.")
    return extractor
def train(model, optimizer, criterion, device):
    total_loss = 0
    count = 0
    with open('./data_modi_multi_up1.pkl', 'rb') as f:
        while True:
            try:
                optimizer.zero_grad()
                count += 1
                data = pickle.load(f)
                feature1 = data.get('Feature_layer_combine_1', None)
                feature2 = data.get('Feature_layer_combine_2', None)
                mos1 = data.get('mos1', None)
                mos2 = data.get('mos2', None)
                score1 = data.get('patch_quality_score1', None)
                score2 = data.get('patch_quality_score2', None)

                mos1, mos2 = mos1.float(), mos2.float()
                mos1.requires_grad_(True)
                mos2.requires_grad_(True)
                mos1 = mos1.to(device)
                mos2 = mos2.to(device)

                feature_list1 = []
                feature_list2 = []

                for f1, f2 in zip(feature1, feature2):
                    feature_list1.append(f1)
                    feature_list2.append(f2)


                scores1 = []
                scores2 = []
                for S1, S2 in zip(score1, score2):
                    scores1.append(S1.data.cpu().numpy().flatten())
                    scores2.append(S2.data.cpu().numpy().flatten())
                scores1 = torch.tensor(scores1)
                scores2 = torch.tensor(scores2)
                scores1_np = np.array(scores1) 
                scores1 = torch.tensor(scores1_np)
                scores2_np = np.array(scores2) 
                scores2 = torch.tensor(scores2_np)

                scores1, scores2 = scores1.to(device), scores2.to(device)

                stanet_output1 = model(scores1, feature_list1)
                stanet_output2 = model(scores2, feature_list2)


                loss = criterion(stanet_output1, stanet_output2, mos1, mos2)

                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            except EOFError:
                break

    return total_loss / count


# Training parameters
learning_rate = args.learning_rate
epochs = args.epochs
device = torch.device(args.device)
save_path = args.save_path

if not os.path.exists(save_path):
    os.mkdir(save_path)

stanet = STANet().to(device)
loss_fn = STANetLoss().to(device)
optimizer = optim.Adam(stanet.parameters(), lr=learning_rate)

# Train the model
for epoch in range(epochs):
    torch.set_grad_enabled(True)
    stanet.train()
    loss_fn.train()
    avg_loss = train(stanet, optimizer, loss_fn, device)
    print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
    # Save the model weights after each epoch
    torch.save(stanet.state_dict(), os.path.join(save_path, f'stanet_epoch_{epoch+1}.pth'))