import random
import pytest
from pytest_cases import parametrize_with_cases

from malariagen_data import af1 as _af1
from malariagen_data import ag3 as _ag3
from malariagen_data.anoph.hapclust import AnophelesHapClustAnalysis


@pytest.fixture
def ag3_sim_api(ag3_sim_fixture):
    return AnophelesHapClustAnalysis(
        url=ag3_sim_fixture.url,
        public_url=ag3_sim_fixture.url,
        config_path=_ag3.CONFIG_PATH,
        major_version_number=_ag3.MAJOR_VERSION_NUMBER,
        major_version_path=_ag3.MAJOR_VERSION_PATH,
        pre=True,
        aim_metadata_dtype={
            "aim_species_fraction_arab": "float64",
            "aim_species_fraction_colu": "float64",
            "aim_species_fraction_colu_no2l": "float64",
            "aim_species_gambcolu_arabiensis": object,
            "aim_species_gambiae_coluzzii": object,
            "aim_species": object,
        },
        gff_gene_type="gene",
        gff_gene_name_attribute="Name",
        gff_default_attributes=("ID", "Parent", "Name", "description"),
        default_phasing_analysis="gamb_colu_arab",
        results_cache=ag3_sim_fixture.results_cache_path.as_posix(),
        taxon_colors=_ag3.TAXON_COLORS,
        virtual_contigs=_ag3.VIRTUAL_CONTIGS,
    )


@pytest.fixture
def af1_sim_api(af1_sim_fixture):
    return AnophelesHapClustAnalysis(
        url=af1_sim_fixture.url,
        public_url=af1_sim_fixture.url,
        config_path=_af1.CONFIG_PATH,
        major_version_number=_af1.MAJOR_VERSION_NUMBER,
        major_version_path=_af1.MAJOR_VERSION_PATH,
        pre=False,
        gff_gene_type="protein_coding_gene",
        gff_gene_name_attribute="Note",
        gff_default_attributes=("ID", "Parent", "Note", "description"),
        default_phasing_analysis="funestus",
        results_cache=af1_sim_fixture.results_cache_path.as_posix(),
        taxon_colors=_af1.TAXON_COLORS,
    )


# N.B., here we use pytest_cases to parametrize tests. Each
# function whose name begins with "case_" defines a set of
# inputs to the test functions. See the documentation for
# pytest_cases for more information, e.g.:
#
# https://smarie.github.io/python-pytest-cases/#basic-usage
#
# We use this approach here because we want to use fixtures
# as test parameters, which is otherwise hard to do with
# pytest alone.


def case_ag3_sim(ag3_sim_fixture, ag3_sim_api):
    return ag3_sim_fixture, ag3_sim_api


def case_af1_sim(af1_sim_fixture, af1_sim_api):
    return af1_sim_fixture, af1_sim_api


@parametrize_with_cases("fixture,api", cases=".")
def test_plot_haplotype_clustering(fixture, api: AnophelesHapClustAnalysis):
    # Set up test parameters.
    all_sample_sets = api.sample_sets()["sample_set"].to_list()
    linkage_methods = (
        "single",
        "complete",
        "average",
        "weighted",
        "centroid",
        "median",
        "ward",
    )
    sample_queries = (None, "sex_call == 'F'")
    hapclust_params = dict(
        region=fixture.random_region_str(region_size=5000),
        sample_sets=[random.choice(all_sample_sets)],
        linkage_method=random.choice(linkage_methods),
        sample_query=random.choice(sample_queries),
        show=False,
    )

    # Run checks.
    api.plot_haplotype_clustering(**hapclust_params)


@parametrize_with_cases("fixture,api", cases=".")
def test_haplotype_cluster_labels(fixture, api: AnophelesHapClustAnalysis):
    import numpy as np
    import pandas as pd

    all_sample_sets = api.sample_sets()["sample_set"].to_list()
    region = fixture.random_region_str(region_size=5000)
    sample_sets = [random.choice(all_sample_sets)]

    # ------------------------------------------------------------------ #
    # Case 1: no threshold — all haplotypes should be in cluster 1.
    # ------------------------------------------------------------------ #
    df = api.haplotype_cluster_labels(
        region=region,
        sample_sets=sample_sets,
        distance_threshold=None,
    )

    # Return type.
    assert isinstance(df, pd.DataFrame)

    # Columns.
    assert set(df.columns) == {"haplotype_id", "sample_id", "cluster"}

    # Each sample contributes exactly two haplotypes.
    expected_n_haps = df["sample_id"].nunique() * 2
    assert len(df) == expected_n_haps

    # haplotype_id format: "{sample_id}_1" or "{sample_id}_2".
    for _, row in df.iterrows():
        sid = row["sample_id"]
        hid = row["haplotype_id"]
        assert hid in (
            sid + "_1",
            sid + "_2",
        ), f"Unexpected haplotype_id {hid!r} for sample {sid!r}"

    # No threshold → all cluster labels are 1.
    assert (df["cluster"] == 1).all()

    # Cluster labels are positive integers.
    assert df["cluster"].dtype in (
        np.dtype("int64"),
        np.dtype("intp"),
        np.dtype("int32"),
    )
    assert (df["cluster"] >= 1).all()

    # ------------------------------------------------------------------ #
    # Case 2: large threshold — all haplotypes also end up in one cluster.
    # ------------------------------------------------------------------ #
    df_large = api.haplotype_cluster_labels(
        region=region,
        sample_sets=sample_sets,
        distance_threshold=1e9,  # effectively no cut
    )
    assert isinstance(df_large, pd.DataFrame)
    assert (df_large["cluster"] >= 1).all()
    assert len(df_large) == len(df)

    # ------------------------------------------------------------------ #
    # Case 3: zero / tiny threshold — every haplotype in its own cluster.
    # ------------------------------------------------------------------ #
    df_zero = api.haplotype_cluster_labels(
        region=region,
        sample_sets=sample_sets,
        distance_threshold=0.0,
    )
    assert isinstance(df_zero, pd.DataFrame)
    assert len(df_zero) == len(df)
    # All cluster labels positive.
    assert (df_zero["cluster"] >= 1).all()

    # ------------------------------------------------------------------ #
    # Case 4: reproducibility — same params → identical results.
    # ------------------------------------------------------------------ #
    df_repeat = api.haplotype_cluster_labels(
        region=region,
        sample_sets=sample_sets,
        distance_threshold=None,
    )
    assert (df["cluster"].values == df_repeat["cluster"].values).all()
    assert (df["haplotype_id"].values == df_repeat["haplotype_id"].values).all()

    # ------------------------------------------------------------------ #
    # Case 5: linkage_method parameter is respected.
    # ------------------------------------------------------------------ #
    df_complete = api.haplotype_cluster_labels(
        region=region,
        sample_sets=sample_sets,
        distance_threshold=5.0,
        linkage_method="complete",
    )
    assert isinstance(df_complete, pd.DataFrame)
    assert len(df_complete) == len(df)
    assert (df_complete["cluster"] >= 1).all()
