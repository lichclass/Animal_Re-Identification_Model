import torch
import torch.nn.functional as F
import copy


class FedProtoServerApp:
    """
    Server-side controller for Federated Prototypical Networks.
    Adds:
        - Global encoder updated via prototype regression
        - Learnable pseudo-inputs (one per identity)
    """

    def __init__(self, backbone, device="cpu"):
        self.best_client = None 
        self.global_prototypes = None      
        self.device = device
        self.global_encoder = backbone

    # ------------------------------------------------------------------
    # GLOBAL PROTOTYPE SETTERS / GETTERS
    # ------------------------------------------------------------------

    def set_global_prototypes(self, global_prototypes):
        self.global_prototypes = global_prototypes

    def set_global_model_weights(self, weights):
        self.global_encoder.load_state_dict(weights)

    def get_global_model_weights(self):
        return self.global_encoder.state_dict()

    # ------------------------------------------------------------------
    # AGGREGATION
    # ------------------------------------------------------------------

    def aggregate(self, client_protos):
        proto_bucket = {}
        for _, proto_dict in client_protos.items():
            for identity, emb in proto_dict.items():
                emb = emb.detach().cpu()
                if identity not in proto_bucket:
                    proto_bucket[identity] = []
                proto_bucket[identity].append(emb)
        global_prototypes = {}
        for identity, embs in proto_bucket.items():
            avg_proto = torch.mean(torch.stack(embs), dim=0)
            norm_proto = F.normalize(avg_proto, dim=0)
            global_prototypes[identity] = norm_proto
        self.set_global_prototypes(global_prototypes)
        return global_prototypes
    
    def fedAvg(self, clients):
        client_states = []
        client_sizes = []
        for client in clients:
            state = client.get_model_weights()
            state_cpu = {k: v.cpu() for k, v in state.items()}
            client_states.append(state_cpu)
            client_sizes.append(len(client.train_dataset))
        total_samples = sum(client_sizes)
        client_weights = [n / total_samples for n in client_sizes]
        global_state = copy.deepcopy(client_states[0])
        for key in global_state.keys():
            avg_param = 0
            for w, state in zip(client_weights, client_states):
                avg_param += w * state[key]
            global_state[key] = avg_param
        self.set_global_model_weights(global_state)
        self.global_encoder.to(self.device)
        for client in clients:
            client.set_model_weights(self.get_global_model_weights())
        return global_state

    @torch.no_grad()
    def evaluate(self, dataset, batch_size=64):
        device = self.device
        model = self.global_encoder.to(device)
        model.eval()

        # ----------------------------
        # 1. Build query/gallery split
        # ----------------------------
        query_imgs, query_ids = [], []
        gallery_imgs, gallery_ids = [], []

        # Convert full dataset into per-identity dictionary
        identity_groups = {}
        for idx in range(len(dataset)):
            _, id_ = dataset[idx]
            identity_groups.setdefault(id_, []).append(idx)

        for id_, indices in identity_groups.items():

            if len(indices) < 2:
                # Need at least 1 query and 1 gallery
                continue

            # First image = query
            q = indices[0]
            query_imgs.append(q)
            query_ids.append(id_)

            # Remaining = gallery
            for g in indices[1:]:
                gallery_imgs.append(g)
                gallery_ids.append(id_)

        # ----------------------------
        # 2. Encode all embeddings
        # ----------------------------

        def encode_indices(idxs):
            loader = torch.utils.data.DataLoader(
                idxs, batch_size=batch_size, shuffle=False,
                collate_fn=lambda batch: [dataset[i] for i in batch]
            )

            embs, labels = [], []
            for batch in loader:
                imgs, ids = zip(*batch)
                imgs = torch.stack(imgs).to(device)
                emb = model(imgs)
                embs.append(emb.cpu())
                labels.extend(ids)

            return torch.cat(embs, dim=0), torch.tensor(labels)

        query_emb, query_labels = encode_indices(query_imgs)      # [Q, D]
        gallery_emb, gallery_labels = encode_indices(gallery_imgs) # [G, D]

        # --------------------------------
        # 3. Compute pairwise distances
        # --------------------------------
        # Use cosine similarity or Euclidean distance — both common in Re-ID
        from torch.nn.functional import pairwise_distance

        # Distance matrix [Q, G]
        dist_mat = torch.cdist(query_emb, gallery_emb, p=2)   # Euclidean

        # --------------------------------
        # 4. Compute Rank-k and mAP
        # --------------------------------

        def compute_map(dist_mat, q_labels, g_labels):
            """
            Derived from standard Market1501/DukeMTMC evaluation.
            """

            num_q = dist_mat.size(0)
            APs = []
            correct_rank1 = 0
            correct_rank5 = 0

            for i in range(num_q):
                dist = dist_mat[i]
                gt = q_labels[i].item()

                # Sorted gallery indices
                sorted_idx = torch.argsort(dist)

                # Rank-1
                if g_labels[sorted_idx[0]].item() == gt:
                    correct_rank1 += 1

                # Rank-5
                if gt in g_labels[sorted_idx[:5]]:
                    correct_rank5 += 1

                # mAP computation
                match = (g_labels[sorted_idx] == gt).numpy()
                num_rel = match.sum()
                if num_rel == 0:
                    continue

                precision_list = []
                hit = 0
                for rank, is_match in enumerate(match, start=1):
                    if is_match:
                        hit += 1
                        precision_list.append(hit / rank)

                APs.append(sum(precision_list) / num_rel)

            return (
                sum(APs) / len(APs),
                correct_rank1 / num_q,
                correct_rank5 / num_q,
            )

        mAP, rank1, rank5 = compute_map(dist_mat, query_labels, gallery_labels)

        print("\n===== 📊 Re-ID Evaluation =====")
        print(f"Rank-1 Accuracy : {rank1:.4f}")
        print(f"Rank-5 Accuracy : {rank5:.4f}")
        print(f"mAP             : {mAP:.4f}")

        return mAP, rank1, rank5

