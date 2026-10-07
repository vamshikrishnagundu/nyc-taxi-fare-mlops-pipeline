from flask import Flask, request, jsonify
import os
import json
import requests
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)

app = Flask(__name__)
app.logger.handlers = logging.getLogger().handlers
app.logger.setLevel(logging.INFO)

# ML alerts fall back to the data-quality webhook until REAL_TEAMS_WEBHOOK_ML is
# set, so the ML channel wiring is a no-op (single channel) until a second
# Power Automate flow/channel is configured.
REAL_WEBHOOK_DATA_QUALITY = os.environ.get("REAL_TEAMS_WEBHOOK")
REAL_WEBHOOK_ML = os.environ.get("REAL_TEAMS_WEBHOOK_ML") or REAL_WEBHOOK_DATA_QUALITY


def extract_text(payload):
    """Pull a readable message out of either our own {"text": ...} alerts
    (Airflow) or Grafana's default alert webhook payload (title/message/alerts)."""
    if not isinstance(payload, dict):
        return str(payload)
    if payload.get("text"):
        return payload["text"]
    if payload.get("message") or payload.get("title"):
        title = payload.get("title", "")
        message = payload.get("message", "")
        return f"{title}\n{message}".strip()
    return json.dumps(payload)


# Per-channel card header so a reader can tell at a glance which kind of alert
# this is before reading the details. Keyed by the forwarder route's label.
ALERT_HEADERS = {
    "data-quality": {"title": "🔴 Data Quality Alert", "color": "Attention"},
    "ml-alerts": {"title": "🟠 ML Alert", "color": "Warning"},
}
DEFAULT_HEADER = {"title": "🔔 Alert", "color": "Accent"}


def to_adaptive_card(text, label="data-quality"):
    """"When a Teams webhook request is received" only accepts Adaptive Card
    attachments, not plain {"text": ...} JSON — wrap the message accordingly.

    The card leads with a bold, colour-coded header naming the alert type
    (Data Quality vs ML), followed by the alert details."""
    header = ALERT_HEADERS.get(label, DEFAULT_HEADER)
    return {
        "attachments": [
            {
                "contentType": "application/vnd.microsoft.card.adaptive",
                "content": {
                    "$schema": "http://adaptivecards.io/schemas/adaptive-card.json",
                    "type": "AdaptiveCard",
                    "version": "1.4",
                    "body": [
                        {
                            "type": "TextBlock",
                            "text": header["title"],
                            "weight": "Bolder",
                            "size": "Large",
                            "color": header["color"],
                            "wrap": True
                        },
                        {
                            "type": "TextBlock",
                            "text": text,
                            "wrap": True,
                            "spacing": "Small"
                        }
                    ]
                }
            }
        ]
    }


def relay(real_webhook, label):
    if not real_webhook:
        return jsonify({"error": f"No webhook configured for {label}"}), 500
    try:
        payload = request.get_json(force=True, silent=True)
        card_payload = to_adaptive_card(extract_text(payload), label)
        headers = {'Content-Type': 'application/json'}
        resp = requests.post(real_webhook, json=card_payload, headers=headers, timeout=10)
        logging.info(f"Forwarded payload to {label} webhook, status={resp.status_code}")
        logging.info(f"Response body: {resp.text[:1000]}")
        return (resp.text, resp.status_code, dict(resp.headers))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/teams', methods=['POST'])
def forward_to_teams_data_quality():
    return relay(REAL_WEBHOOK_DATA_QUALITY, "data-quality")


@app.route('/teams/ml-alerts', methods=['POST'])
def forward_to_teams_ml():
    return relay(REAL_WEBHOOK_ML, "ml-alerts")


@app.route('/send_test', methods=['POST', 'GET'])
def send_test():
    if not REAL_WEBHOOK_DATA_QUALITY:
        return jsonify({"error": "REAL_TEAMS_WEBHOOK not configured"}), 500
    try:
        card_payload = to_adaptive_card(
            "[webhook-forwarder] Test alert: this is a verification message.",
            "data-quality",
        )
        headers = {'Content-Type': 'application/json'}
        resp = requests.post(REAL_WEBHOOK_DATA_QUALITY, json=card_payload, headers=headers, timeout=10)
        logging.info(f"Sent test payload to data-quality webhook, status={resp.status_code}")
        logging.info(f"Test response body: {resp.text[:1000]}")
        return (resp.text, resp.status_code, dict(resp.headers))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8085)
