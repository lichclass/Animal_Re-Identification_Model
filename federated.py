from tqdm import tqdm
from dataset import SeaTurtleDataset
from utils import (
    extract_embeddings, 
    compute_rank1_rank5_map, 
    build_model, 
    build_backbone,
)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torch.amp import autocast_mode

class FederatedClient:
    def __init__(self, client_id, train_df, args):
        self.client_id = client_id
        self.args = args
        self.device = args['device']
        self.train_df = train_df.copy()

        unique_local_ids = sorted(self.train_df['identity'].unique().tolist())
        self.num_local_classes = len(unique_local_ids)
        self.local_id_map = {identity: idx for idx, identity in enumerate(unique_local_ids)}
        self.inv_local_id_map = {idx: identity for identity, idx in self.local_id_map.items()}
        self.train_df['train_label'] = self.train_df['identity'].map(self.local_id_map)

        head_kwargs = {}
        if args['head'] == 'arcface':
            head_kwargs = {'s': args['s'], 'm': args['m']}
        elif args['head'] == 'adaface':
            head_kwargs = {'m': args['m'], 'h': args['h'], 's': args['s'], 't_alpha': args['t_alpha']}
        
        self.model = build_model(
            embedding_dim=args['embedding_dim'],
            num_classes=self.num_local_classes,
            backbone_type=args['backbone'],
            head_type=args['head'],
            pretrained_backbone=True,
            dropout=args['dropout'],
            **head_kwargs
        ).to(self.device)

        print(f"[Client {self.client_id}] Images: {len(self.train_df)} | Identities: {self.num_local_classes}")

    def get_loader(self):
        transform = T.Compose([
            T.Resize((384, 384)),
            T.RandomHorizontalFlip(p=0.5),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            T.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        dataset = SeaTurtleDataset(self.train_df, self.args['split_mode'], 'train', transform)
        return DataLoader(
            dataset, 
            batch_size=self.args['batch_size'], 
            shuffle=True, 
            num_workers=self.args['num_workers'], 
            pin_memory=self.args['pin_memory']
        )

    def train(self, global_backbone_state, global_prototypes, current_lr):
        self.model.backbone.load_state_dict(global_backbone_state)
        self.model.train()
        
        criterion = nn.CrossEntropyLoss()
        loader = self.get_loader()
        
        optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=current_lr, 
            weight_decay=self.args['weight_decay']
        )
        
        for epoch in range(self.args['local_epochs']):
            epoch_loss = 0.0
            epoch_cls_loss = 0.0
            epoch_proto_loss = 0.0
            
            iterator = tqdm(
                loader,
                desc=f'Client {self.client_id} | Epoch {epoch+1}/{self.args["local_epochs"]}'
            )
            
            for images, train_labels, _, _ in iterator:
                images, train_labels = images.to(self.device), train_labels.to(self.device)
                
                with autocast_mode.autocast(device_type='cuda'):
                    logits, emb = self.model(images, train_labels)
                    loss_cls = criterion(logits, train_labels)
                    
                    # Federated Prototype Loss
                    loss_proto = torch.tensor(0.0, device=self.device)
                    if global_prototypes:
                        valid_terms = []
                        for i, local_idx in enumerate(train_labels):
                            # Map local index to global identity string
                            identity_str = self.inv_local_id_map[local_idx.item()]
                            
                            if identity_str in global_prototypes:
                                g_proto = global_prototypes[identity_str].to(self.device)
                                # MSE between normalized local embedding and global prototype
                                valid_terms.append(F.mse_loss(emb[i], g_proto))
                        
                        if valid_terms:
                            loss_proto = torch.stack(valid_terms).mean() * self.args['lambda_proto']
                    
                    # Combine Losses
                    total_loss = loss_cls + loss_proto
                
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
                
                epoch_loss += total_loss.item()
                epoch_cls_loss += loss_cls.item()
                epoch_proto_loss += loss_proto.item()
                
                iterator.set_postfix({
                    'Loss': f'{total_loss.item():.4f}',
                    'Cls': f'{loss_cls.item():.4f}',
                    'Proto': f'{loss_proto.item():.4f}'
                })
            
            avg_loss = epoch_loss / len(loader)
            avg_cls = epoch_cls_loss / len(loader)
            avg_proto = epoch_proto_loss / len(loader)
            print(f'  [Client {self.client_id}] Epoch {epoch+1}: Loss={avg_loss:.4f} (Cls={avg_cls:.4f}, Proto={avg_proto:.4f})')
        
        new_prototypes = self._compute_local_prototypes(loader)
        
        # Return backbone weights only
        return self.model.backbone.state_dict(), new_prototypes

    @torch.no_grad()
    def _compute_local_prototypes(self, loader):
        self.model.eval()
        prototype_sums = {}
        prototype_counts = {}
        
        for images, train_labels, _, _ in loader:
            images = images.to(self.device)
            
            # Get Raw Embeddings, no normalization
            raw_emb = self.model.backbone.forward_raw(images)
            
            for i in range(raw_emb.size(0)):
                local_idx = train_labels[i].item()
                identity_str = self.inv_local_id_map[local_idx]
                
                if identity_str not in prototype_sums:
                    prototype_sums[identity_str] = raw_emb[i].detach().cpu()
                    prototype_counts[identity_str] = 1
                else:
                    prototype_sums[identity_str] += raw_emb[i].detach().cpu()
                    prototype_counts[identity_str] += 1
        
        prototypes = {}
        for identity_str in prototype_sums:
            mean_vec = prototype_sums[identity_str] / prototype_counts[identity_str]
            prototypes[identity_str] = F.normalize(mean_vec, p=2, dim=0)
        
        return prototypes
    

class FederatedServer:
    def __init__(self, args):
        self.args = args
        self.device = args['device']
        
        self.global_backbone = build_backbone(
            embedding_dim=args['embedding_dim'], 
            model_type=args['backbone'], 
            pretrained=True,
            dropout=args.get('dropout', 0.1)
        ).to(self.device)

        self.global_prototypes = {}

    def _get_eval_model(self):
        backbone = self.global_backbone
        
        class EvalWrapper(nn.Module):
            def __init__(self, backbone):
                super().__init__()
                self.backbone = backbone
            
            def forward(self, x, labels=None):
                emb = self.backbone(x, return_norms=False)
                return None, emb  # (logits, embeddings) tuple
        
        return EvalWrapper(backbone)

    def aggregate_weights(self, client_weights_list):
        print("[Server] Aggregating Backbone Weights...")
        avg_weights = {}
        num_clients = len(client_weights_list)
        
        ref_keys = client_weights_list[0].keys()
        for key in ref_keys:
            avg_weights[key] = sum(cw[key] for cw in client_weights_list) / num_clients
        
        self.global_backbone.load_state_dict(avg_weights)
        return avg_weights

    def aggregate_prototypes(self, client_protos_list):
        print("[Server] Aggregating Prototypes...")
        
        round_sums = {}
        round_counts = {}
        
        for c_protos in client_protos_list:
            for identity_str, vec in c_protos.items():
                if identity_str not in round_sums:
                    round_sums[identity_str] = vec
                    round_counts[identity_str] = 1
                else:
                    round_sums[identity_str] += vec
                    round_counts[identity_str] += 1
        
        momentum = self.args['proto_momentum']
        updated_count = 0
        new_count = 0
        
        for identity_str, vec_sum in round_sums.items():
            current_avg = vec_sum / round_counts[identity_str]
            current_avg = F.normalize(current_avg, p=2, dim=0)
            
            if identity_str in self.global_prototypes:
                old_proto = self.global_prototypes[identity_str]
                new_proto = (old_proto * momentum) + (current_avg * (1 - momentum))
                self.global_prototypes[identity_str] = F.normalize(new_proto, p=2, dim=0)
                updated_count += 1
            else:
                self.global_prototypes[identity_str] = current_avg
                new_count += 1
        
        print(f"[Server] Updated: {updated_count} | New: {new_count} | Total: {len(self.global_prototypes)}")
        return self.global_prototypes

    def evaluate(self, loader, set_name="Val"):
        print(f"[Server] Evaluating on {set_name} Set...")
        eval_model = self._get_eval_model()
        embs, labels, encounters = extract_embeddings(
            eval_model, loader, self.device, set_name
        )
        
        r1, r5, mAP = compute_rank1_rank5_map(
            embs, labels, encounters, 
            embs, labels, encounters, 
            self.device, encounter_based=None
        )
        return r1, r5, mAP