import torch
import numpy as np

from tasksampler import FewShotTaskSampler
from modules.resnet_aspp import ResNet18ASPPEncoder
from modules.resnet18 import ResNet18Encoder
from modules.protonetloss import PrototypicalLoss, compute_prototypes
from trainer import train_one_epoch as train_fn
from evaluator import evaluate_one_epoch as eval_fn

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
        lr=1e-4,
    ):
        self.cid = cid
        self.local_prototypes = None
        self.global_prototypes = None
        self.global_prototype_classes = None

        # Few Shot Configs
        self.n_way = n_way
        self.k_shot = k_shot
        self.n_samples = n_samples
        self.episodes = episodes

        # Model Configs
        self.embedding_dim = embedding_dim
        self.loss_fn = PrototypicalLoss(k_shot)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.model = self.__build_model__(model).to(self.device)

        self.optimizer = (
            torch.optim.Adam(self.model.parameters(), lr=lr)
            if optimizer == "adam"
            else torch.optim.SGD(self.model.parameters(), lr=lr)
        )

        self.train_dataset = train_dataset
        self.train_loader = train_loader
        self.train_sampler = self.__build_sampler__()

    # ---------------------------
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
        global_prototypes_dict: {identity_str: embedding[D]} from server.

        We convert this into:
            - self.global_prototypes        : [C, D]
            - self.global_prototype_classes : [C] (int labels in this client's label space)
        Only classes that this client actually knows (present in its dataset)
        are kept for alignment.
        """
        self.global_prototypes_dict = global_prototypes_dict

        # 1) Get base dataset (unwrap Subset if needed)
        if isinstance(self.train_dataset, torch.utils.data.Subset):
            base_dataset = self.train_dataset.dataset  # SeaTurtleDataset
        else:
            base_dataset = self.train_dataset

        id2idx = base_dataset.identity_to_idx  # {'t001': 0, ...}

        proto_list = []
        class_list = []

        # We fix an order over identities (sorted for determinism)
        for identity in sorted(global_prototypes_dict.keys()):
            if identity in id2idx:
                class_idx = id2idx[identity]  # int label in this dataset
                class_list.append(class_idx)
                proto_list.append(global_prototypes_dict[identity].to(self.device))

        if len(proto_list) == 0:
            # No overlap between this client and global identities (rare but possible)
            self.global_prototypes = None
            self.global_prototype_classes = None
        else:
            self.global_prototypes = torch.stack(proto_list, dim=0).to(self.device)         # [C, D]
            self.global_prototype_classes = torch.tensor(class_list, dtype=torch.long, device=self.device)

    def get_global_prototypes(self):
        # return the tensor version by default (what training uses)
        return self.global_prototypes, self.global_prototype_classes

    def get_sampled_embeddings(self, n_min=3, n_max=10, n_default=5):
        return self.__build_sample_embeddings__(n_min, n_max, n_default)

    # ---------------------------
    # Available Methods

    def fit(self):
        g_protos, g_classes = self.global_prototypes, self.global_prototype_classes

        train_loss, train_acc = train_fn(
            model=self.model,
            train_dataset=self.train_dataset,
            task_sampler=self.train_sampler,
            loss_fn=self.loss_fn,
            optimizer=self.optimizer,
            device=self.device,
            global_prototypes=g_protos,
            global_prototype_classes=g_classes,
            client_id=self.cid,
            tqdm_position=self.cid,
        )

        # Build local prototypes for sending to server
        self.local_prototypes = self.__build_local_prototypes__()
        return train_loss, train_acc
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
        batch_indices = next(iter(self.train_sampler))
        imgs = []
        true_labels = []
        for idx in batch_indices:
            img, label = self.train_dataset[idx]  # label = TRUE global label
            imgs.append(img)
            true_labels.append(label)
        imgs = torch.stack(imgs).to(self.device)
        true_labels = torch.tensor(true_labels, dtype=torch.long, device=self.device)
        embeddings = self.model(imgs).cpu()
        prototypes, classes = compute_prototypes(
            embeddings,
            true_labels,
            self.k_shot
        )
        local_prototypes = {
            self.train_dataset.dataset.idx_to_identity[int(c)]: prototypes[i]
            for i, c in enumerate(classes)
        }
        return local_prototypes

    def __build_sample_embeddings__(self, n_min, n_max, n_default):
        base_k = max(n_min, min(n_default, n_max))
        base_dataset = self.train_dataset.dataset
        idx_to_identity = base_dataset.idx_to_identity
        self.model.eval()
        label_to_embeds = {}
        with torch.no_grad():
            for imgs, labels in self.train_loader:
                imgs = imgs.to(self.device)
                labels = labels.to(self.device)
                emb = self.model(imgs)
                emb = emb.detach().cpu()
                for e, lbl in zip(emb, labels):
                    lbl_int = int(lbl.item())
                    if lbl_int not in label_to_embeds:
                        label_to_embeds[lbl_int] = []
                    label_to_embeds[lbl_int].append(e)
        sampled_embeds = {}
        sampled_labels = {}
        for lbl_int, embs in label_to_embeds.items():
            num = len(embs)
            if num == 0:
                continue
            K = min(base_k, num)
            perm = torch.randperm(num)[:K].tolist()
            chosen = [embs[i] for i in perm]
            emb_tensor = torch.stack(chosen, dim=0)
            label_tensor = torch.full((K,), lbl_int, dtype=torch.long, device=self.device)
            identity_str = idx_to_identity[lbl_int]
            sampled_embeds[identity_str] = emb_tensor
            sampled_labels[identity_str] = label_tensor

        return sampled_embeds, sampled_labels


