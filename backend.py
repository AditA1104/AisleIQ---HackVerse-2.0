import json
import requests

# -------------------------------------------------------------
# 1. INVENTORY CONTEXT (RAG Data)
# Store product specs and stock info for specific aisles
# -------------------------------------------------------------
INVENTORY_DB = {
    "Electronics": [
        {"item": "4K Smart TV 55-inch", "stock": 4, "price": "$450", "common_question": "Does it support HDMI 2.1?"},
        {"item": "Noise-Canceling Headphones", "stock": 12, "price": "$120", "common_question": "Battery life & Bluetooth range"},
        {"item": "USB-C Fast Charger", "stock": 25, "price": "$25", "common_question": "Compatibility with iPhone/Android"}
    ],
    "Snacks": [
        {"item": "Organic Potato Chips", "stock": 50, "price": "$3.50", "common_question": "Gluten-free availability"},
        {"item": "Dark Chocolate Bar 70%", "stock": 30, "price": "$4.00", "common_question": "Vegan ingredients"}
    ]
}

def get_aisle_context(aisle_name):
    """Fetches inventory specs for a given aisle."""
    return INVENTORY_DB.get(aisle_name, [])

# -------------------------------------------------------------
# 2. CONFUSION ALERT GENERATOR
# Translates customer dwell time (>45s) into staff instructions
# -------------------------------------------------------------
def generate_confusion_alert(aisle_name, dwell_time_seconds, customer_id):
    """
    Generates an urgent notification for store staff.
    Uses local Ollama/Granite if running, or a smart fallback template.
    """
    inventory_context = get_aisle_context(aisle_name)
    
    prompt = f"""
    You are an AI assistant for a retail store floor team.
    Customer #{customer_id} has been standing in the '{aisle_name}' aisle for {dwell_time_seconds} seconds without picking up an item.
    Available Inventory: {json.dumps(inventory_context)}

    Write a 2-sentence alert for a staff member's watch telling them where to go and what product info to offer.
    """

    # Attempt to use local Ollama / Granite LLM model
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "granite3-dense", # Change to "llama3" or "mistral" if you use a different local model
                "prompt": prompt,
                "stream": False
            },
            timeout=3
        )
        if response.status_code == 200:
            return response.json().get("response", "").strip()
    except Exception:
        # Fallback if Ollama is not currently active
        pass

    # Backup smart alert if Ollama isn't running
    top_item = inventory_context[0]['item'] if inventory_context else "products"
    return (
        f"🚨 CONFUSION ALERT: Customer #{customer_id} stuck in {aisle_name} (Dwell time: {dwell_time_seconds}s). "
        f"Assistance recommended! Check on stock/specs for {top_item}."
    )

# -------------------------------------------------------------
# 3. HEATMAP MONETIZATION REPORT
# Generates shelf placement pricing recommendations for owners
# -------------------------------------------------------------
def generate_heatmap_monetization_report(aisle_name, dwell_stats):
    """Generates shelf monetization recommendations based on dwell time."""
    total_dwell = dwell_stats.get("total_dwell_time_minutes", 0)
    footfall = dwell_stats.get("footfall_count", 0)

    is_hotspot = total_dwell > 15
    recommendation = (
        "Increase eye-level shelf slotting fee by +20% for featured brand advertising." 
        if is_hotspot else 
        "Rearrange promotional signage and test discount tags to boost engagement."
    )

    report = (
        f"📊 HEATMAP MONETIZATION REPORT ({aisle_name.upper()} AISLE)\n"
        f"• Total Visitors: {footfall}\n"
        f"• Cumulative Dwell Time: {total_dwell} minutes\n"
        f"• Zone Status: {'🔥 HIGH-INTEREST RED ZONE' if is_hotspot else '🔵 LOW-ENGAGEMENT BLUE ZONE'}\n"
        f"• Action Plan: {recommendation}\n"
    )
    return report

# -------------------------------------------------------------
# LOCAL TEST CODE
# -------------------------------------------------------------
if __name__ == "__main__":
    print("--- Testing Confusion Alert ---")
    alert = generate_confusion_alert("Electronics", dwell_time_seconds=52, customer_id=104)
    print(alert)

    print("\n--- Testing Heatmap Monetization Report ---")
    sample_stats = {"footfall_count": 42, "total_dwell_time_minutes": 28}
    report = generate_heatmap_monetization_report("Electronics", sample_stats)
    print(report)