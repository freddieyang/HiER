"""Label-free CSR atom gathering aligned with the published-dictionary means."""
from pathlib import Path
import json

import torch


class AtomicMoleculeFeatureStore:
    """Wrap the original store without changing means, weights or Morgan rows.

    Persistent atoms use FP16; corrections subtract their own uniform average
    before being added to original FP32 means. Component/atom indices are used
    only for alignment and CSR grouping, never supplied to learned embeddings.
    """
    def __init__(self, base, values, offsets):
        values = torch.as_tensor(values)
        offsets = torch.as_tensor(offsets, dtype=torch.long, device='cpu')
        if (values.ndim != 2 or values.shape[1] != base.molecule_input_dim
                or values.requires_grad or not torch.isfinite(values).all()
                or offsets.shape != (len(base.vectors)+1,) or offsets[0] != 0
                or offsets[-1] != len(values) or (offsets.diff() <= 0).any()):
            raise ValueError('Finite frozen atoms with complete molecule CSR required')
        self.base = base
        self.values = values.to(device=base.device, dtype=torch.float16)
        if not bool(torch.isfinite(self.values).all()):
            raise ValueError('Atomic values overflow FP16 storage')
        self.atom_offsets = offsets
        self.atom_lengths = offsets.diff()
        self.component_rows = base.rows.cpu()

    def __getattr__(self, name):
        return getattr(self.base, name)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, indices):
        if isinstance(indices, torch.Tensor):
            indices = indices.detach().cpu()
        indices = torch.as_tensor(indices, dtype=torch.long).reshape(-1)
        data = self.base[indices]
        width = data['mask'].shape[1]
        cpu_mask = torch.arange(width)[None, :] < self.base.lengths[indices, None]
        molecule_ids = self.component_rows[indices, :width][cpu_mask]
        lengths_cpu = self.atom_lengths[molecule_ids]
        pointer_cpu = torch.cat((torch.zeros(1, dtype=torch.long), lengths_cpu.cumsum(0)))
        count = int(pointer_cpu[-1])
        pointer = pointer_cpu.to(self.device)
        lengths = lengths_cpu.to(self.device)
        owner = torch.repeat_interleave(torch.arange(len(molecule_ids), device=self.device),
                                       lengths, output_size=count)
        starts = self.atom_offsets[molecule_ids].to(self.device)
        positions = starts[owner] + torch.arange(count, device=self.device) - pointer[owner]
        data.update(atomic_values=self.values[positions], atomic_pointer=pointer,
                    component_flat_rows=cpu_mask.reshape(-1).nonzero(as_tuple=True)[0].to(self.device))
        return data


def attach_atomic_features(base, atom_cache, source_dir):
    """Fail closed on vocabulary/means/cache mismatch; no statistics are fitted."""
    atom_cache, source_dir = Path(atom_cache), Path(source_dir)
    summary = json.loads((atom_cache/'summary.json').read_text())
    if summary['status'] != 'completed_integrity_checked' or not summary['all_corrected_means_exact_raw_atom_means']:
        raise ValueError('Completed published-atom cache required')
    mean_manifest = json.loads((source_dir/'reaction_feature_manifest.json').read_text())
    smiles = []
    for entry in mean_manifest['shards']:
        smiles.extend(torch.load(entry['path'], map_location='cpu', weights_only=False)['smiles'])
    chunks, lengths, loaded_smiles = [], [], []
    means = base.vectors.cpu()
    for file in sorted((atom_cache/'paired_shards').glob('molecules_*.pt')):
        shard = torch.load(file, map_location='cpu', weights_only=False)
        first = len(loaded_smiles)
        last = first+len(shard['smiles'])
        if list(shard['global_indices']) != list(range(first, last)):
            raise ValueError('Raw atom molecule order differs from mean cache')
        if not torch.equal(shard['published_mean'], means[first:last]):
            raise ValueError('Only identical published-dictionary molecule means may anchor this readout')
        atoms, offsets = shard['atomic_values'], shard['offsets']
        if atoms.dtype != torch.float32 or offsets[0] != 0 or offsets[-1] != len(atoms):
            raise ValueError('Invalid published-atom shard')
        for i, (start, stop) in enumerate(zip(offsets[:-1].tolist(), offsets[1:].tolist())):
            if not 1 <= stop-start <= 256 or not torch.equal(atoms[start:stop].mean(0), means[first+i]):
                raise ValueError('Raw atoms do not reconstruct the original published mean')
        chunks.append(atoms.to(dtype=torch.float16))
        lengths.append(offsets.diff())
        loaded_smiles.extend(shard['smiles'])
    if (loaded_smiles != smiles or len(smiles) != summary['molecules']
            or sum(len(x) for x in chunks) != summary['true_retained_atoms']):
        raise ValueError('Incomplete/misaligned atomic cache')
    values = torch.cat(chunks)
    offsets = torch.cat((torch.zeros(1, dtype=torch.long), torch.cat(lengths).cumsum(0)))
    store = AtomicMoleculeFeatureStore(base, values, offsets)
    audit = dict(source=str(atom_cache.resolve()), molecules=len(smiles), atoms=len(values),
        storage_dtype='float16', original_mean_dtype='float32',
        exact_original_fp32_means_verified=True, raw_atom_uniform_subtraction=True,
        cached_dictionary='published', atom_boundaries='actual published special token IDs excluded',
        pretrained_encoder_calls=0, uses_enzyme_or_ec_labels=False, uses_pair_associations=False,
        fitted_statistics=False, persistent_atom_bytes=store.values.numel()*store.values.element_size())
    return store, audit
