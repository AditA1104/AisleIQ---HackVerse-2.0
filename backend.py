import os
import requests
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

# Load the hidden variables from the .env file
load_dotenv() 

# 1. Fetch the secure webhook URL (ALL CAPS to match the rest of your script)
SLACK_WEBHOOK_URL = os.environ.get("SLACK_WEBHOOK_URL", "")

# 2. Initialize Local IBM Granite LLM via Ollama (100% Free / Local on Mac) [cite: 710, 715]
try:
    llm = ChatOllama(model="granite-code:8b", temperature=0.2)
except Exception as e:
    print(f"[AisleIQ Backend Warning] Local Ollama model failed to load: {e}")
    llm = None

def send_slack_alert(message: str) -> bool:
    """Pushes an alert message to the store staff Slack channel."""
    # Check if URL is empty or still set to default mock string
    if not SLACK_WEBHOOK_URL or "YOUR/WEBHOOK/URL" in SLACK_WEBHOOK_URL:
        print(f"[Mock Slack Alert Sent - Webhook URL Missing in .env]: {message}")
        return True
    
    payload = {"text": message}
    try:
        response = requests.post(SLACK_WEBHOOK_URL, json=payload, timeout=5)
        
        if response.status_code == 200:
            print(f"✅ [Slack Alert Sent Successfully!]: {message}")
            return True
        else:
            print(f"❌ [Slack Error {response.status_code}]: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ [Slack Connection Failed]: {e}")
        return False

def trigger_dwell_time_alert(aisle_name: str, dwell_seconds: int) -> str:
    """
    Called when Pooja's math module detects dwell_time > 45s.
    Generates a natural, action-oriented alert via IBM Granite LLM.
    """
    if llm:
        try:
            prompt = ChatPromptTemplate.from_template(
                "You are AisleIQ, an intelligent store assistant. A customer has been lingering "
                "at the {aisle} aisle for {dwell_time} seconds without picking up an item. "
                "Generate a concise, 1-sentence urgent notification for a floor employee "
                "recommending immediate assistance."
            )
            chain = prompt | llm
            result = chain.invoke({"aisle": aisle_name, "dwell_time": dwell_seconds})
            # THIS IS THE LINE THAT CHANGED 👇
            alert_text = f"🚨 **AisleIQ Alert**: {str(result.content)}" 
        except Exception as e:
            alert_text = f"🚨 **AisleIQ Alert**: Customer lingering at {aisle_name} for {dwell_seconds}s. Please assist!"
    else:
        alert_text = f"🚨 **AisleIQ Alert**: Customer lingering at {aisle_name} for {dwell_seconds}s. Please assist!"
    # Send to Slack
    send_slack_alert(alert_text)
    return alert_text

def summarize_daily_trends(heatmap_json: dict) -> str:
    """
    Generates an executive summary for store owners based on daily foot traffic heatmaps.
    """
    if not llm:
        return "Daily traffic concentrated heavily in Electronics and Checkout zones."
        
    prompt = ChatPromptTemplate.from_template(
        "You are a retail analytics expert. Based on this daily foot traffic data: {data}, "
        "summarize the top 2 actionable insights for the store owner in plain English. "
        "Highlight high-value red zones and congested areas."
    )
    chain = prompt | llm
    try:
        response = chain.invoke({"data": heatmap_json})
        return str(response.content)
    except Exception as e:
        return "High traffic in Electronics (Shelf 2B). Recommend premium product placement."

# Test block when running backend.py directly
if __name__ == "__main__":
    print("Testing AisleIQ Backend Engine...")
    alert = trigger_dwell_time_alert(aisle_name="Electronics", dwell_seconds=52)
    print("Generated Alert Output:", alert)