from collections import defaultdict
import random
import torch


class FewShotTaskSampler:
    def __init__(self, labels, n_way, n_samples, iterations, allow_replacement=False):
        self.labels = torch.as_tensor(labels) if not isinstance(labels, torch.Tensor) else labels
        self.n_way = int(n_way)
        self.n_samples_per_class = int(n_samples)
        self.iterations = int(iterations)
        self.allow_replacement = bool(allow_replacement)
        
        self.idxs_by_class = defaultdict(list)
        for idx, label in enumerate(self.labels):
            label_item = label.item() if isinstance(label, torch.Tensor) else label
            self.idxs_by_class[label_item].append(idx)
        
        self.classes = list(self.idxs_by_class.keys())
        
        # Filter classes that have enough samples
        self.valid_classes = [
            cls for cls in self.classes 
            if len(self.idxs_by_class[cls]) >= self.n_samples_per_class
        ]
        
        if len(self.valid_classes) < self.n_way:
            raise ValueError(
                f"Not enough classes with {self.n_samples_per_class} samples. "
                f"Need {self.n_way} classes, but only {len(self.valid_classes)} are available."
            )
    
    def __len__(self):
        return self.iterations
    
    def __iter__(self):
        for _ in range(self.iterations):
            # Sample n_way classes from valid classes
            sampled_classes = random.sample(self.valid_classes, self.n_way)
            batch_indices = []
            
            for cls in sampled_classes:
                available_indices = self.idxs_by_class[cls]
                
                if self.allow_replacement:
                    sampled = random.choices(available_indices, k=self.n_samples_per_class)
                else:
                    # Always sample exactly n_samples_per_class (guaranteed by valid_classes)
                    sampled = random.sample(available_indices, self.n_samples_per_class)
                
                batch_indices.extend(sampled)
            
            yield batch_indices