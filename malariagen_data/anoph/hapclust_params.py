"""Parameters for haplotype clustering functions."""

from typing import Annotated

from typing_extensions import TypeAlias

from .clustering_params import linkage_method

linkage_method_default: linkage_method = "single"

distance_threshold: TypeAlias = Annotated[
    float,
    """
    The distance threshold used to form flat clusters from the hierarchical
    linkage tree. Haplotypes whose pairwise distance is less than or equal
    to this value are placed in the same cluster. The units are the same as
    the distance matrix — for haplotypes this is the absolute Hamming
    distance (number of differing SNPs). A value of ``None`` means no
    threshold is applied and all haplotypes are assigned to a single cluster.
    """,
]
