"""Parameter sharing for EC-conditioned fusion in one dual-encoder model.

The reaction and protein encoders remain modality-specific. The EC router is
already shared. This controlled variant ties only the two RouteFusion modules;
it never feeds a reaction into a candidate encoder or a ground-truth EC into
either encoder.
"""
from __future__ import annotations

from .functional_set_retriever import FunctionalSetRetriever


class SharedFusionFunctionalSetRetriever(FunctionalSetRetriever):
    """One fusion rule, called independently for each modality.

    ``independent_matched`` starts two separate fusion modules with identical
    random weights. ``shared`` starts with the same weights but keeps just one
    registered module, accumulating both modalities' gradients. All other
    initial weights and RNG consumption match between these two controls.

    The original independent-random model is unchanged in its original class.
    No old learned checkpoint is averaged, loaded, or converted here.
    """

    def __init__(self, *, fusion_sharing: str = "shared", **kwargs):
        if fusion_sharing not in {"independent_matched", "shared"}:
            raise ValueError("Unknown EC fusion sharing policy")
        # Construct the original modules first to preserve the initialization
        # and RNG of every common parameter in the paired experiment.
        super().__init__(**kwargs)
        self.fusion_sharing = fusion_sharing
        if fusion_sharing == "independent_matched":
            self.protein_fusion.load_state_dict(self.reaction_fusion.state_dict(), strict=True)
        else:
            fusion = self.reaction_fusion
            del self.reaction_fusion
            del self.protein_fusion
            self.shared_fusion = fusion

    def __getattr__(self, name):
        # Inherited encoders keep their clear side-specific call sites. These
        # aliases resolve to ONE registered module, not two checkpoint copies.
        modules = self.__dict__.get("_modules", {})
        if name in {"reaction_fusion", "protein_fusion"} and "shared_fusion" in modules:
            return modules["shared_fusion"]
        return super().__getattr__(name)
