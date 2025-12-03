import torch
import numpy as np
from collections import defaultdict

def build_federated_splits(
    full_dataset,
    num_clients=3,
    overlap_ratio=0.20,               # 20% of identities appear in multiple clients
    min_clients_per_identity=2,       # overlapped identities appear in 2 clients
    seed=42
):
    np.random.seed(seed)

    df = full_dataset.df
    identities = np.array(df["identity"].unique())
    num_identities = len(identities)

    # -------------------------------------------
    # 1) Randomize identities once using seed
    # -------------------------------------------
    rng = np.random.default_rng(seed)
    rng.shuffle(identities)

    # -------------------------------------------
    # 2) Select overlapped identities
    # -------------------------------------------
    num_overlap = int(num_identities * overlap_ratio)
    overlapped_identities = set(identities[:num_overlap])
    base_identities = identities[num_overlap:]

    # -------------------------------------------
    # 3) Assign base identities (distinct per client)
    # -------------------------------------------
    base_splits = np.array_split(base_identities, num_clients)

    # -------------------------------------------
    # 4) Assign overlapping identities
    # Each identity will appear in >=2 clients
    # -------------------------------------------
    client_identities = [set(split) for split in base_splits]

    for ident in overlapped_identities:
        # Randomly choose k clients to receive this identity
        k = min(min_clients_per_identity, num_clients)
        chosen = rng.choice(num_clients, size=k, replace=False)
        for cid in chosen:
            client_identities[cid].add(ident)

    # -------------------------------------------
    # 5) Build dataset subsets
    # -------------------------------------------
    client_datasets = []
    for cid in range(num_clients):
        ids = list(client_identities[cid])
        idxs = df[df["identity"].isin(ids)].index.values
        subset = torch.utils.data.Subset(full_dataset, idxs)
        client_datasets.append(subset)

    # Debug print
    print("\n===== FEDERATED SPLIT SUMMARY =====")
    print(f"Total identities: {num_identities}")
    print(f"Overlapped identities: {len(overlapped_identities)}")
    for cid in range(num_clients):
        print(f"Client {cid}: {len(client_identities[cid])} identities "
              f"({len(client_datasets[cid])} samples)")
    print("===================================\n")

    return client_datasets