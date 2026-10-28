"""Reaction-context readout of frozen, correctly tokenized Uni-Mol atoms.

Scaled dot-product attention follows Vaswani et al. (2017),
https://arxiv.org/abs/1706.03762. Here there is no Transformer stack or
enzyme-conditioned cross encoder: the masked reaction context supplies one
query, and each component's atoms supply keys and unchanged values. Explicit
uniform subtraction preserves the original molecule mean at initialization.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F
from torch_scatter import segment_csr

from .functional_set_retriever import molecule_moments
from .residue_readout import ResidueReadoutRetriever


def bounded_component_logits(logits, pointer, owner, bound):
    """Smoothly bound within-component contrast without fixing its logit origin.

    For n atoms, any two softmax weights have ratio at most exp(2 * bound).
    This capacity control is an experiment, not a reaction-center annotation.
    ``bound`` is validated by the caller; the disabled path never calls here.
    """
    center = segment_csr(logits, pointer, reduce='mean')  # [Ncomponents]
    return bound * torch.tanh((logits - center[owner]) / bound)  # [A]


class ConditionalAtomicReadout(nn.Module):
    """[B,C] context and CSR [A,F] atoms -> [Ncomponents,F] mean corrections."""

    def __init__(self, context_dim, atom_dim, hidden_dim=64, logit_bound=0.):
        super().__init__()
        if min(context_dim, atom_dim, hidden_dim) < 1:
            raise ValueError('Positive atomic readout dimensions required')
        if not math.isfinite(logit_bound) or logit_bound < 0:
            raise ValueError('Atomic logit bound must be finite and nonnegative; zero disables it')
        self.query = nn.Linear(context_dim, hidden_dim, bias=False)
        self.key = nn.Linear(atom_dim, hidden_dim, bias=False)
        nn.init.zeros_(self.query.weight)
        self.hidden_dim = hidden_dim
        self.logit_bound = float(logit_bound)

    def forward(self, context, values, pointer, component_reactions):
        if values.requires_grad:
            raise ValueError('Atomic features must remain frozen')
        if (values.ndim != 2 or values.shape[1] != self.key.in_features
                or context.ndim != 2 or context.shape[1] != self.query.in_features
                or pointer.ndim != 1 or pointer.dtype != torch.long
                or component_reactions.shape != (len(pointer)-1,)
                or component_reactions.dtype != torch.long):
            raise ValueError('Invalid atomic/context dimensions or CSR metadata')
        lengths = pointer.diff()
        if (len(lengths) == 0 or int(pointer[0]) != 0 or int(pointer[-1]) != len(values)
                or bool((lengths <= 0).any()) or bool((component_reactions < 0).any())
                or bool((component_reactions >= len(context)).any())):
            raise ValueError('Nonempty complete CSR components required')
        with torch.autocast(device_type=values.device.type, enabled=False):
            x = values.float()
            if not bool(torch.isfinite(x).all()) or not bool(torch.isfinite(context).all()):
                raise ValueError('Atomic values and reaction context must be finite')
            keys = self.key(F.layer_norm(x, (x.shape[-1],), eps=1e-5))  # [A,H]
            queries = self.query(context.float())  # [B,H]
            owner = torch.repeat_interleave(torch.arange(len(lengths), device=x.device),
                                           lengths, output_size=len(x))
            logits = (keys * queries[component_reactions[owner]]).sum(-1) / math.sqrt(self.hidden_dim)
            if self.logit_bound:
                logits = bounded_component_logits(logits, pointer, owner, self.logit_bound)
            maxima = segment_csr(logits.detach(), pointer, reduce='max')
            unnormalized = (logits-maxima[owner]).exp()
            denominator = segment_csr(unnormalized, pointer, reduce='sum')
            attention = unnormalized / denominator[owner].clamp_min(1e-12)
            uniform = lengths.float().reciprocal()[owner]
            delta = segment_csr((attention-uniform)[:, None]*x, pointer, reduce='sum')
            entropy = segment_csr(-attention*attention.clamp_min(1e-12).log(), pointer, reduce='sum')
        return delta, entropy.exp()/lengths.float(), attention


class AtomicReadoutRetriever(ResidueReadoutRetriever):
    """Refine molecule means before TRAIN-IDF moments; retain the original index.

    The query sees only already-masked reaction views. Molecular values retain
    all cached atoms, including hydrogen; no EC/center/side annotation is input.
    """
    def __init__(self, *, atomic_readout='conditional', atom_attention_dim=64,
                 atomic_mix=.5, atomic_logit_bound=0., **kwargs):
        if (atomic_readout != 'conditional' or kwargs.get('pooling') != 'moments'
                or kwargs.get('scoring', 'explicit') != 'explicit'
                or kwargs.get('drop_unimol_probability', 0.) != 0
                or kwargs.get('ec_posterior') != 'leaf_marginal' or atomic_mix != .5):
            raise ValueError('Atomic control requires moments, marginal EC, explicit scoring and intact Uni-Mol')
        super().__init__(**kwargs)
        self.atomic_mix = float(atomic_mix)
        # Extra module initialization must not change common initialization or
        # any later training dropout/candidate-sampling random stream.
        with torch.random.fork_rng(devices=[]):
            self.atomic_readout = ConditionalAtomicReadout(
                self.global_input_dim+2*self.molecule_input_dim,
                self.molecule_input_dim, atom_attention_dim, atomic_logit_bound)

    def _reaction_feature_keys(self):
        return super()._reaction_feature_keys() | {'atomic_values', 'atomic_pointer', 'component_flat_rows'}

    def _refine_molecule_readout(self, features, molecules, mask, global_state, molecule_state):
        if self.view_dropout.unimol_probability != 0:
            raise ValueError('Atomic experiment cannot bypass a dropped Uni-Mol view')
        flat = features['component_flat_rows']
        expected = mask.reshape(-1).nonzero(as_tuple=True)[0]
        if flat.dtype != torch.long or not torch.equal(flat, expected):
            raise ValueError('Atomic components must exactly align with all valid molecule slots')
        delta, fraction, _ = self.atomic_readout(
            torch.cat((global_state, molecule_state), -1), features['atomic_values'],
            features['atomic_pointer'], flat // molecules.shape[1])
        correction = torch.zeros_like(molecules.float().reshape(-1, self.molecule_input_dim))
        correction = correction.index_copy(0, flat, delta).view_as(molecules)
        refined = molecules.float() + self.atomic_mix*correction
        moments, _ = molecule_moments(refined, mask, features['weights'])
        return self.moments_norm(moments), dict(
            atomic_effective_fraction=fraction,
            atomic_refinement_norm=delta.norm(dim=-1)*self.atomic_mix)
