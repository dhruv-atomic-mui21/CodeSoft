import os
import json
import time
import logging
import traceback
from flask import Flask, request, jsonify, render_template
from requests.exceptions import ReadTimeout, ConnectionError
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Flask app config
app = Flask(__name__, template_folder='templates', static_folder='static')
logging.basicConfig(level=logging.INFO)

# Constants
GEMINI_API_KEY = os.getenv('API_KEY')
GEMINI_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
MAX_RETRIES = 3
RETRY_DELAY = 5  # seconds

if not GEMINI_API_KEY:
    raise EnvironmentError("API_KEY not set in environment")

@app.route('/')
def serve_frontend():
    return render_template('recomendation.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.get_json(force=True)
    category = data.get('category')
    preferences = data.get('preferences', [])

    if not category or not preferences:
        return jsonify({"recommendations": [], "error": "Category and preferences are required"}), 400

    prompt = (
        f"I have preferences in the category '{category}': {', '.join(preferences)}.\n"
        f"Please recommend the top 5 items strictly in JSON format like:\n"
        f"[{{\"title\": \"...\", \"description\": \"...\"}}, ...]\nno additional text.\nno need for extra brackets or formatting.\n"
    )

    headers = {
        "Content-Type": "application/json",
        "X-goog-api-key": GEMINI_API_KEY
    }

    payload = {
        "contents": [{"parts": [{"text": prompt}]}]
    }

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.post(GEMINI_ENDPOINT, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            response_json = response.json()

            # Extract raw content
            candidates = response_json.get('candidates', [])
            content_obj = candidates[0].get('content', {}) if candidates else {}
            parts = content_obj.get('parts', [])
            raw_text = parts[0].get('text', '') if parts else ''

            # Try to parse a valid JSON array
            recommendations = parse_recommendation_output(raw_text)
            return jsonify({"recommendations": recommendations})

        except (ReadTimeout, ConnectionError) as e:
            logging.warning(f"[Attempt {attempt}] Timeout or connection error: {e}")
            if attempt == MAX_RETRIES:
                return jsonify({"recommendations": [], "error": "Request to Gemini API timed out."}), 504
            time.sleep(RETRY_DELAY)

        except Exception as e:
            logging.error(f"Exception occurred: {traceback.format_exc()}")
            return jsonify({"recommendations": [], "error": "An internal error occurred."}), 500

import re

def parse_recommendation_output(text):
    """
    Clean and parse the LLM response to extract a valid list of recommendation dicts.
    """

    # Remove Markdown code fences if they exist
    code_block = re.search(r"```(?:json)?\s*(\[\s*{.*?}\s*\])\s*```", text, re.DOTALL)
    if code_block:
        text = code_block.group(1)
    else:
        # Fallback: try to extract JSON-like list
        json_like_match = re.search(r"(\[\s*{.*}\s*\])", text, re.DOTALL)
        if json_like_match:
            text = json_like_match.group(1)

    # Try to parse cleaned JSON array
    try:
        recommendations = json.loads(text.strip())
        if isinstance(recommendations, list):
            return [
                {
                    "title": item.get("title", "Untitled"),
                    "description": item.get("description", "")
                }
                for item in recommendations if isinstance(item, dict)
            ]
    except Exception as e:
        logging.warning(f"Failed to parse clean JSON block: {e}")

    # Fallback: basic line split (worst case)
    recommendations = []
    for line in text.strip().splitlines():
        if ':' in line:
            title, desc = line.split(':', 1)
            recommendations.append({
                "title": title.strip(),
                "description": desc.strip()
            })
    if not recommendations:
        return [{"title": "AI Output", "description": text.strip()}]
    return recommendations


if __name__ == '__main__':
    app.run(debug=True, port=5000)
