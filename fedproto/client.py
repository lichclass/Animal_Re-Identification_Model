import torch
import torch.nn.functional as F
import numpy as np

from utils.tasksampler import FewShotTaskSampler
from modules.resnet_aspp import ResNet18ASPPEncoder
from modules.resnet18 import ResNet18Encoder
from modules.protonetloss import PrototypicalLoss, compute_prototypes

# Import the improved trainer
import sys
sys.path.append('.')
from utils.trainer import train_one as train_fn

class FedProtoClientApp:
    def __init__(
        self, 
        cid, 
        train_dataset, 
        train_loader,
        n_way, 
        k_shot, 
        n_samples, 
        episodes,
        model="resnet18_aspp",
        optimizer="adam",
        embedding_dim=256,
        lambda_align=0.5,
        lambda_triplet=0.3,
        lr=1e-4
    ):
        self.cid = cid
        self.local_prototypes = None  
        self.local_proto_counts = None
        self.global_prototypes = None  

        # Few Shot Configs
        self.n_way = n_way
        self.k_shot = k_shot
        self.n_samples = n_samples
        self.episodes = episodes
        self.lambda_align = lambda_align
        self.lambda_triplet = lambda_triplet

        # Model Configs
        self.embedding_dim = embedding_dim
        self.loss_fn = PrototypicalLoss(k_shot)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = self.__build_model__(model).to(self.device)

        self.optimizer = (
            torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
            if optimizer == "adam"
            else torch.optim.SGD(self.model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
        )

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=10, eta_min=lr/10
        )

        self.train_dataset = train_dataset
        self.train_loader = train_loader
        self.train_sampler = self.__build_sampler__()

    # Getters and Setters
    def set_cid(self, cid):
        self.cid = cid

    def get_cid(self):
        message = {
            'client': self.cid
        }
        return message

    def set_local_prototypes(self, prototypes):
        self.local_prototypes = prototypes
        
    def get_local_prototypes(self):
        message = {
            'client': self.cid, 
            'num_prototypes': len(self.local_prototypes) if self.local_prototypes is not None else 0,
            'prototypes': self.local_prototypes,
            'counts': self.local_proto_counts
        }
        return message

    def set_model_weights(self, message):
        weights = message['model_weights']
        self.model.load_state_dict(weights)

    def get_model_weights(self):
        message = {
            'client': self.cid, 
            'model_weights': self.model.state_dict()
        }
        return message

    def set_global_prototypes(self, global_prototypes_dict):
        self.global_prototypes = global_prototypes_dict

    def get_global_prototypes(self):
        message = {
            'client': self.cid,
            'num_global_prototypes': len(self.global_prototypes) if self.global_prototypes is not None else 0,
            'global_prototypes': self.global_prototypes
        }
        return message

    # Training
    def fit(self):
        proto_loss, triplet_loss, acc = train_fn(
                model=self.model,
                train_dataset=self.train_dataset,
                task_sampler=self.train_sampler,
                loss_fn=self.loss_fn,
                optimizer=self.optimizer,
                device=self.device,
                global_prototypes=self.global_prototypes,  # Pass dict directly
                client_id=self.cid,
                lambda_align=self.lambda_align,
                lambda_triplet=self.lambda_triplet,
            )
        self.scheduler.step()
        self.__build_local_prototypes__()
        message = {'client': self.cid,  'proto_loss': proto_loss, 'triplet_loss': triplet_loss, 'acc': acc}
        return message

    def __build_sampler__(self):
        base_dataset = self.train_dataset
        if isinstance(self.train_dataset, torch.utils.data.Subset):
            base_dataset = self.train_dataset.dataset
            subset_indices = self.train_dataset.indices
        else:
            base_dataset = self.train_dataset
            subset_indices = None
            
        all_labels = base_dataset.df["identity"].map(base_dataset.identity_to_idx).values
        
        if subset_indices is not None:
            train_labels = all_labels[subset_indices]
        else:
            train_labels = all_labels
            
        unique = sorted(set(train_labels))
        local_map = {c: i for i, c in enumerate(unique)}
        mapped_labels = np.array([local_map[l] for l in train_labels])
        
        train_sampler = FewShotTaskSampler(
            labels=mapped_labels,
            n_way=self.n_way,
            n_samples=self.n_samples,
            iterations=self.episodes,
            allow_replacement=False
        )
        return train_sampler

    def __build_model__(self, model):
        if model == "resnet18_aspp":
            return ResNet18ASPPEncoder(embedding_dim=self.embedding_dim)
        elif model == "resnet18":
            return ResNet18Encoder(embedding_dim=self.embedding_dim)
        else:
            raise ValueError(f"Unknown model: {model}")

    def __build_local_prototypes__(self):
        """
        Build prototypes using MULTIPLE episodes for stability.
        Returns dict[identity_str -> normalized_tensor]
        """
        base_dataset = (self.train_dataset.dataset 
                       if isinstance(self.train_dataset, torch.utils.data.Subset)
                       else self.train_dataset)
        idx_to_identity = base_dataset.idx_to_identity
        
        self.model.eval()
        
        # Accumulate embeddings per identity across multiple episodes
        identity_embeddings = {}
        
        with torch.no_grad():
            # Sample 3-5 episodes to get stable prototypes
            for _ in range(min(5, len(self.train_sampler))):
                batch_indices = next(iter(self.train_sampler))
                imgs = []
                true_labels = []
                
                for idx in batch_indices:
                    img, label = base_dataset[idx]
                    imgs.append(img)
                    true_labels.append(label)
                    
                imgs = torch.stack(imgs).to(self.device)
                true_labels = torch.tensor(true_labels, dtype=torch.long, device=self.device)
                
                embeddings = self.model(imgs)
                
                # Group by identity
                for emb, label in zip(embeddings, true_labels):
                    identity = idx_to_identity[int(label)]
                    if identity not in identity_embeddings:
                        identity_embeddings[identity] = []
                    identity_embeddings[identity].append(emb.cpu())
        
        # Average and normalize
        local_prototypes = {}
        local_proto_counts = {}
        for identity, embs in identity_embeddings.items():
            avg_proto = torch.stack(embs).mean(dim=0)
            norm_proto = F.normalize(avg_proto, dim=0)
            local_prototypes[identity] = norm_proto
            local_proto_counts[identity] = len(embs)

        self.set_local_prototypes(local_prototypes)
        self.local_proto_counts = local_proto_counts
        return local_prototypes