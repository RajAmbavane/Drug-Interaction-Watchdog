"""Tests for CYP450 feature extraction, the main mechanistic signal in the ML model."""

import pandas as pd
import pytest

from ml.features.cyp450_features import (
    CYP450FeatureExtractor,
    DrugCYPProfile,
)

parse = CYP450FeatureExtractor._parse_cyp_roles_from_text


# ── Text parsing ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "text, enzyme, role",
    [
        ("Ketoconazole is a strong inhibitor of CYP3A4.", "CYP3A4", "inhibitor"),
        ("Rifampin induces CYP 3A4 expression.", "CYP3A4", "inducer"),
        ("Codeine is metabolized by CYP2D6 to morphine.", "CYP2D6", "substrate"),
        ("Warfarin is a substrate of cyp2c9.", "CYP2C9", "substrate"),
    ],
)
def test_parse_detects_enzyme_and_role(text, enzyme, role):
    found = parse(text)
    assert (found[0][0], found[0][1]) == (enzyme, role)


def test_parse_ignores_mentions_without_a_role():
    assert parse("CYP2C19 genotype was recorded for all participants.") == []


def test_parse_ignores_text_without_cyp_mentions():
    assert parse("Take with food to reduce stomach upset.") == []


# ── Per-drug profile ─────────────────────────────────────────────────────────

def test_profile_normalises_enzyme_names():
    p = DrugCYPProfile("simvastatin")
    p.add_role("cyp 3a4", "substrate", "snippet")
    p.add_role("CYP2C9", "inhibitor")
    p.add_role("CYP1A2", "unknown-role")
    assert p.substrates == {"CYP3A4"}
    assert p.inhibitors == {"CYP2C9"}
    assert p.all_enzymes == {"CYP3A4", "CYP2C9"}
    assert p.snippets == ["snippet"]


# ── Pairwise features ────────────────────────────────────────────────────────

@pytest.fixture
def extractor():
    ext = CYP450FeatureExtractor()
    ket = ext._get_profile("Ketoconazole")
    ket.add_role("CYP3A4", "inhibitor", "ketoconazole inhibits CYP3A4")
    sim = ext._get_profile("Simvastatin")
    sim.add_role("CYP3A4", "substrate")
    rif = ext._get_profile("Rifampin")
    rif.add_role("CYP3A4", "inducer")
    ome = ext._get_profile("Omeprazole")
    ome.add_role("CYP2C19", "substrate")
    return ext


def test_inhibitor_substrate_pair_is_flagged(extractor):
    f = extractor._pair_features("ketoconazole", "simvastatin")
    assert f["inhibitor_substrate_pairs"] == 1
    assert f["inducer_substrate_pairs"] == 0
    assert f["cyp3a4_involved"] == 1
    assert f["max_cyp_risk_score"] == 3
    assert f["cyp_description_snippet"].startswith("ketoconazole")


def test_pair_features_are_symmetric(extractor):
    ab = extractor._pair_features("rifampin", "simvastatin")
    ba = extractor._pair_features("simvastatin", "rifampin")
    assert ab["inducer_substrate_pairs"] == ba["inducer_substrate_pairs"] == 1


def test_unrelated_enzymes_produce_no_interaction_signal(extractor):
    f = extractor._pair_features("ketoconazole", "omeprazole")
    assert f["inhibitor_substrate_pairs"] == 0
    assert f["shared_cyp_count"] == 0
    assert f["n_enzymes_involved"] == 2
    assert f["cyp2c19_involved"] == 1


def test_transform_requires_fit():
    with pytest.raises(RuntimeError):
        CYP450FeatureExtractor().transform(pd.DataFrame({"drug_a_name": ["a"], "drug_b_name": ["b"]}))


def test_unknown_drugs_yield_zero_features(extractor):
    f = extractor._pair_features("unknowndrug", "anotherdrug")
    assert f["any_cyp_interaction"] == 0
    assert f["max_cyp_risk_score"] == 0
    assert f["cyp_description_snippet"] == ""


def test_transform_appends_feature_columns():
    ext = CYP450FeatureExtractor()
    ext._get_profile("a").add_role("CYP2D6", "inhibitor")
    ext._get_profile("b").add_role("CYP2D6", "substrate")
    ext._fitted = True
    pairs = pd.DataFrame({"drug_a_name": ["a"], "drug_b_name": ["b"], "label": [1]})
    out = ext.transform(pairs)
    assert out.loc[0, "label"] == 1
    assert out.loc[0, "cyp2d6_involved"] == 1
    assert out.loc[0, "inhibitor_substrate_pairs"] == 1
