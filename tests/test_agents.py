"""Tests for the rule-based alert routing agent."""

from types import SimpleNamespace

import pytest

from tests.conftest import load_module

routing = load_module("agents/alert_routing_agent.py")
AlertRoutingAgent = routing.AlertRoutingAgent
Destination = routing.Destination
Urgency = routing.Urgency


def prediction(drug_a="drug_x", drug_b="drug_y", severity=0, adjusted=None):
    return SimpleNamespace(drug_a=drug_a, drug_b=drug_b, severity=severity, adjusted_severity=adjusted)


def patient(**kwargs):
    defaults = dict(is_pregnant=False, is_dialysis=False, inr=None, prior_alerts=[])
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


@pytest.fixture
def agent(monkeypatch):
    a = AlertRoutingAgent()
    # Never send email/webhooks from tests
    monkeypatch.setattr(a, "_dispatch", lambda *args, **kwargs: None)
    return a


# ── Urgency classification ───────────────────────────────────────────────────

@pytest.mark.parametrize(
    "severity, expected",
    [(0, Urgency.ROUTINE), (1, Urgency.ROUTINE), (2, Urgency.PRIORITY), (3, Urgency.URGENT)],
)
def test_base_urgency_follows_severity(severity, expected):
    urgency, reason = AlertRoutingAgent._classify_urgency(severity, prediction(), None)
    assert urgency == expected
    assert reason == ""


def test_pregnancy_with_contraindicated_drug_is_critical():
    urgency, reason = AlertRoutingAgent._classify_urgency(
        1, prediction("warfarin", "paracetamol"), patient(is_pregnant=True)
    )
    assert urgency == Urgency.CRITICAL
    assert "Pregnancy" in reason


def test_dialysis_with_renally_cleared_drug_is_critical():
    urgency, reason = AlertRoutingAgent._classify_urgency(
        0, prediction("metformin", "drug_y"), patient(is_dialysis=True)
    )
    assert urgency == Urgency.CRITICAL
    assert "Dialysis" in reason


def test_high_inr_with_anticoagulant_is_critical():
    urgency, reason = AlertRoutingAgent._classify_urgency(
        2, prediction("Warfarin", "aspirin"), patient(inr=4.2)
    )
    assert urgency == Urgency.CRITICAL
    assert "INR=4.2" in reason


def test_normal_inr_does_not_escalate():
    urgency, _ = AlertRoutingAgent._classify_urgency(2, prediction("warfarin", "aspirin"), patient(inr=2.5))
    assert urgency == Urgency.PRIORITY


def test_unacknowledged_prior_alert_bumps_one_tier():
    prior = [{"drug_a": "drug_y", "drug_b": "drug_x", "was_acknowledged": False}]
    urgency, reason = AlertRoutingAgent._classify_urgency(1, prediction(), patient(prior_alerts=prior))
    assert urgency == Urgency.PRIORITY
    assert "Prior alert" in reason


def test_acknowledged_prior_alert_is_ignored():
    prior = [{"drug_a": "drug_x", "drug_b": "drug_y", "was_acknowledged": True}]
    urgency, _ = AlertRoutingAgent._classify_urgency(1, prediction(), patient(prior_alerts=prior))
    assert urgency == Urgency.ROUTINE


# ── Routing decisions ────────────────────────────────────────────────────────

def test_minor_interaction_only_reaches_log_and_patient(agent):
    decision = agent.route(object(), prediction(severity=1), patient(), [])
    assert decision.destinations == [Destination.LOG, Destination.PATIENT]


def test_major_interaction_reaches_full_care_team(agent):
    decision = agent.route(object(), prediction(severity=3), patient(), [])
    assert set(decision.delivered_to()) == {"log", "patient", "pharmacist", "physician", "ehr"}


def test_adjusted_severity_takes_precedence(agent):
    decision = agent.route(object(), prediction(severity=1, adjusted=2), patient(), [])
    assert decision.severity == 2
    assert Destination.PHARMACIST in decision.destinations


def test_critical_override_escalates_low_severity_to_physician(agent):
    decision = agent.route(object(), prediction("warfarin", "drug_y", severity=0), patient(is_pregnant=True), [])
    assert decision.urgency == Urgency.CRITICAL
    assert {Destination.PHYSICIAN, Destination.EHR}.issubset(decision.destinations)


def test_run_without_prediction_leaves_state_unchanged(agent):
    state = {"dual_report": None}
    assert agent.run(state) == {"dual_report": None}


def test_decision_serialises_to_dict(agent):
    data = agent.route(object(), prediction(severity=2), None, []).to_dict()
    assert data["urgency"] == "priority"
    assert data["destinations"] == ["log", "patient", "pharmacist"]
