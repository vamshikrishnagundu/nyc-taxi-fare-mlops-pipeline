"""Unit tests for the Teams webhook forwarder's pure payload helpers.

``extract_text`` normalises the two payload shapes the forwarder receives (our
own Airflow ``{"text": ...}`` alerts and Grafana's ``title``/``message`` alerts),
and ``to_adaptive_card`` wraps a message in the Adaptive Card envelope that Power
Automate's "When a Teams webhook request is received" trigger requires.
"""
from app import extract_text, to_adaptive_card


def test_extract_text_prefers_text_key():
    """Our own alerts send {"text": ...} and should pass straight through."""
    assert extract_text({"text": "drift detected"}) == "drift detected"


def test_extract_text_combines_grafana_title_and_message():
    out = extract_text({"title": "Data Quality", "message": "20% rows dropped"})
    assert "Data Quality" in out
    assert "20% rows dropped" in out


def test_extract_text_accepts_non_dict_payload():
    assert extract_text("plain string") == "plain string"


def test_extract_text_falls_back_to_json_dump():
    """An unrecognised dict is serialised rather than lost."""
    out = extract_text({"foo": "bar"})
    assert "foo" in out and "bar" in out


def test_to_adaptive_card_has_required_envelope():
    card = to_adaptive_card("hello world", "data-quality")
    attachment = card["attachments"][0]
    assert attachment["contentType"] == "application/vnd.microsoft.card.adaptive"
    body = attachment["content"]["body"]
    # The card leads with a type header, then the message body.
    assert body[0]["weight"] == "Bolder"
    assert body[-1]["text"] == "hello world"
    assert body[-1]["wrap"] is True


def test_to_adaptive_card_header_names_the_alert_type():
    dq = to_adaptive_card("x", "data-quality")["attachments"][0]["content"]["body"][0]
    ml = to_adaptive_card("y", "ml-alerts")["attachments"][0]["content"]["body"][0]
    assert "Data Quality" in dq["text"]
    assert "ML" in ml["text"]


def test_to_adaptive_card_unknown_label_falls_back():
    block = to_adaptive_card("z", "something-else")["attachments"][0]["content"]["body"][0]
    assert block["text"] == "🔔 Alert"
