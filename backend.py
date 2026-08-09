import os
import requests
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from dotenv import load_dotenv
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate

# Load environment variables from .env
load_dotenv()
SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL")

# Initialize Local IBM Granite LLM via Ollama
try:
    llm = Ollama(model="granite-code:8b")
except Exception as e:
    print(f"⚠️ Ollama model initialization warning: {e}")
    llm = None

# Global alert log for Mridul's Streamlit UI
RECENT_ALERTS_LOG = []
MAX_ALERT_LOG_SIZE = 50  # cap so the log doesn't grow unbounded during a long demo

# LLM call tuning
LLM_TIMEOUT_SECONDS = 8
LLM_MAX_RETRIES = 2

_llm_executor = ThreadPoolExecutor(max_workers=2)


def _invoke_chain_with_timeout(chain, inputs: dict):
    """
    Runs chain.invoke() with a timeout so a slow/stuck local Ollama call
    can't block the whole alert dispatch path. Retries a couple of times
    before giving up and letting the caller fall back to a template alert.
    """
    last_error = None
    for attempt in range(1, LLM_MAX_RETRIES + 1):
        future = _llm_executor.submit(chain.invoke, inputs)
        try:
            return future.result(timeout=LLM_TIMEOUT_SECONDS)
        except FutureTimeoutError as e:
            last_error = e
            print(f"⏱️ LLM call timed out (attempt {attempt}/{LLM_MAX_RETRIES})")
        except Exception as e:
            last_error = e
            print(f"⚠️ LLM call failed (attempt {attempt}/{LLM_MAX_RETRIES}): {e}")
    raise last_error if last_error else RuntimeError("LLM call failed with no error captured")


def _append_alert(entry: dict) -> None:
    """Appends an alert to the log and trims it to MAX_ALERT_LOG_SIZE."""
    RECENT_ALERTS_LOG.append(entry)
    if len(RECENT_ALERTS_LOG) > MAX_ALERT_LOG_SIZE:
        del RECENT_ALERTS_LOG[: len(RECENT_ALERTS_LOG) - MAX_ALERT_LOG_SIZE]


def send_slack_notification(message: str) -> bool:
    """Dispatches a text message to the configured Slack Webhook."""
    if not SLACK_WEBHOOK_URL:
        print("❌ Error: SLACK_WEBHOOK_URL not found in environment variables.")
        return False

    payload = {"text": message}
    try:
        response = requests.post(SLACK_WEBHOOK_URL, json=payload, timeout=5)
        if response.status_code == 200:
            print("✅ Slack alert dispatched successfully!")
            return True
        else:
            print(f"❌ Slack API returned error {response.status_code}: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Exception sending Slack notification: {e}")
        return False


def handle_dwell_alert(aisle_name: str, dwell_seconds: int) -> str:
    """
    Triggers LLM 1-sentence urgent notification and posts to Slack + UI feed.
    (Note: Upstream cooldown logic is handled by Pooja's module).
    """
    alert_text = ""
    if llm:
        try:
            prompt = ChatPromptTemplate.from_template(
                "You are AisleIQ, an intelligent store assistant. A customer has been lingering "
                "at the {aisle} aisle for {dwell_time} seconds without picking up an item. "
                "Generate a concise, 1-sentence urgent notification for a floor employee "
                "recommending immediate assistance."
            )
            chain = prompt | llm
            result = _invoke_chain_with_timeout(chain, {"aisle": aisle_name, "dwell_time": dwell_seconds})

            # Extract content string safely
            content_str = str(result.content if hasattr(result, 'content') else result)
            alert_text = f"🚨 AisleIQ Alert: {content_str.strip()}"
        except Exception as e:
            alert_text = f"🚨 AisleIQ Alert: Customer lingering at {aisle_name} for {dwell_seconds}s. Please assist!"
    else:
        alert_text = f"🚨 AisleIQ Alert: Customer lingering at {aisle_name} for {dwell_seconds}s. Please assist!"

    # Log alert for Mridul's Streamlit UI feed (bounded so it can't grow forever)
    _append_alert({
        "aisle": aisle_name,
        "dwell_seconds": dwell_seconds,
        "message": alert_text
    })

    # Dispatch live notification to Slack
    send_slack_notification(alert_text)
    return alert_text


def get_live_alerts():
    """Returns the recent alert log list for Streamlit UI integration."""
    return RECENT_ALERTS_LOG


def summarize_daily_trends(traffic_data: dict) -> str:
    """Generates a plain-English executive summary of daily store traffic and heatmaps."""
    if not llm:
        return "Daily Summary: Store foot traffic operating as normal."
    try:
        prompt = ChatPromptTemplate.from_template(
            "You are AisleIQ's retail analyst. Summarize the following daily traffic and heatmap "
            "metrics into 2 clear sentences for the store manager: {data}"
        )
        chain = prompt | llm
        result = _invoke_chain_with_timeout(chain, {"data": str(traffic_data)})
        return str(result.content if hasattr(result, 'content') else result).strip()
    except Exception as e:
        return f"Daily Summary: Store processed metrics successfully. (Details: {e})"


if __name__ == "__main__":
    print("Testing AisleIQ Backend Engine...")
    test_msg = handle_dwell_alert("Electronics", 52)
    print("Generated Alert:", test_msg)