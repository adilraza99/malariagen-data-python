import pandas as pd
import pytest

from malariagen_data import Af1, Ag3


@pytest.fixture
def ag3_sim_api(ag3_sim_fixture, tmp_path):
    data_path = ag3_sim_fixture.bucket_path.as_posix()
    return Ag3(
        url=data_path,
        public_url=data_path,
        pre=True,
        check_location=False,
        bokeh_output_notebook=False,
        results_cache=tmp_path.as_posix(),
    )


@pytest.fixture
def af1_sim_api(af1_sim_fixture, tmp_path):
    data_path = af1_sim_fixture.bucket_path.as_posix()
    return Af1(
        url=data_path,
        public_url=data_path,
        pre=True,
        check_location=False,
        bokeh_output_notebook=False,
        results_cache=tmp_path.as_posix(),
    )


def test_load_inversion_tags(ag3_sim_api):
    df = ag3_sim_api.load_inversion_tags(inversion="2Rb")
    assert isinstance(df, pd.DataFrame)
    assert set(df.columns) >= {"inversion", "contig", "position", "alt_allele"}
    assert (df["inversion"] == "2Rb").all()
    assert (df["contig"] == "2R").all()
    assert len(df) > 0


def test_load_inversion_tags_2la(ag3_sim_api):
    df = ag3_sim_api.load_inversion_tags(inversion="2La")
    assert isinstance(df, pd.DataFrame)
    assert (df["inversion"] == "2La").all()
    assert (df["contig"] == "2L").all()
    assert len(df) > 0


def test_load_inversion_tags_invalid(ag3_sim_api):
    with pytest.raises(ValueError, match="Unknown inversion"):
        ag3_sim_api.load_inversion_tags(inversion="X_x")


def test_load_inversion_tags_not_implemented(af1_sim_api):
    with pytest.raises(NotImplementedError):
        af1_sim_api.load_inversion_tags(inversion="2La")


def test_karyotype(ag3_sim_api):
    df = ag3_sim_api.karyotype(inversion="2Rb")
    assert isinstance(df, pd.DataFrame)
    expected_cols = {
        "sample_id",
        "inversion",
        "karyotype_2Rb_mean",
        "karyotype_2Rb",
        "total_tag_snps",
    }
    assert set(df.columns) == expected_cols
    assert (df["inversion"] == "2Rb").all()
    assert all(df["karyotype_2Rb"].isin([0, 1, 2]))
    assert all(df["karyotype_2Rb_mean"].between(0, 2))


def test_karyotype_invalid_inversion(ag3_sim_api):
    with pytest.raises(ValueError, match="Unknown inversion"):
        ag3_sim_api.karyotype(inversion="X_x")


def test_karyotype_not_implemented(af1_sim_api):
    with pytest.raises(NotImplementedError):
        af1_sim_api.karyotype(inversion="2La")
