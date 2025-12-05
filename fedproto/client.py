import torch
import numpy as np

from tasksampler import FewShotTaskSampler
from modules.resnet_aspp import ResNet18ASPPEncoder
from modules.resnet18 import ResNet18Encoder
from modules.protonetloss import PrototypicalLoss, compute_prototypes

# Import the improved trainer
import sys
sys.path.append('.')
from trainer import train_one_epoch as train_fn

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
        lr=1e-4,
        use_triplet_loss=True,  # NEW
    ):
        self.cid = cid
        self.local_prototypes = None  # dict[identity_str -> tensor]
        self.global_prototypes = None  # dict[identity_str -> tensor]
        self.use_triplet_loss = use_triplet_loss

        # Few Shot Configs
        self.n_way = n_way
        self.k_shot = k_shot
        self.n_samples = n_samples
        self.episodes = episodes
        self.lambda_align = lambda_align

        # Model Configs
        self.embedding_dim = embedding_dim
        self.loss_fn = PrototypicalLoss(k_shot)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.model = self.__build_model__(model).to(self.device)

        # Use AdamW with weight decay for better generalization
        self.optimizer = (
            torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
            if optimizer == "adam"
            else torch.optim.SGD(self.model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
        )

        # Cosine annealing scheduler (better than StepLR)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=10, eta_min=lr/10
        )

        self.train_dataset = train_dataset
        self.train_loader = train_loader
        self.train_sampler = self.__build_sampler__()

    # Getters and Setters
    def set_local_prototypes(self, prototypes):
        self.local_prototypes = prototypes
        
    def get_local_prototypes(self):
        return self.local_prototypes

    def set_model_weights(self, weights):
        self.model.load_state_dict(weights)

    def get_model_weights(self):
        return self.model.state_dict()

    def set_global_prototypes(self, global_prototypes_dict):
        """
        Receives dict[identity_str -> tensor] directly from server.
        No conversion needed!
        """
        self.global_prototypes = global_prototypes_dict

    def get_global_prototypes(self):
        return self.global_prototypes

    # Training
    def fit(self):
        (
            total_loss,
            proto_loss,
            triplet_loss,
            align_loss,
            train_acc
        ) = train_fn(
            model=self.model,
            train_dataset=self.train_dataset,
            task_sampler=self.train_sampler,
            loss_fn=self.loss_fn,
            optimizer=self.optimizer,
            device=self.device,
            global_prototypes=self.global_prototypes,  # Pass dict directly
            client_id=self.cid,
            lambda_align=self.lambda_align,
            tqdm_position=self.cid,
        )
        self.scheduler.step()
        
        # Update local prototypes after training
        self.local_prototypes = self.__build_local_prototypes__()
        
        return total_loss, train_acc

    def __build_sampler__(self):
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
                    img, label = self.train_dataset[idx]
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
        for identity, embs in identity_embeddings.items():
            avg_proto = torch.stack(embs).mean(dim=0)
            # L2 normalization for better Re-ID performance
            norm_proto = torch.nn.functional.normalize(avg_proto, dim=0)
            local_prototypes[identity] = norm_proto
            
        self.model.train()
        return local_prototypes