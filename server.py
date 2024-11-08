# server.py - Flask backend server using requests
from flask import Flask, request, jsonify
import requests
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

PORT = 3000
COHERE_API_KEY = 'WlAjHTaXW5ElFhvmUNWmcPr97IAiinSuHIqOUOk7'

@app.route('/coherenceapi', methods=['POST'])
def coherence_api():
    try:
        response = requests.post(
            'https://api.cohere.ai/v1/generate',
            json=request.json,
            headers={
                'Content-Type': 'application/json',
                'Authorization': f'Bearer {COHERE_API_KEY}'
            }
        )
        response.raise_for_status()  # Raises an exception for 4xx/5xx responses
        return jsonify(response.json())
    except requests.exceptions.RequestException as e:
        error_message = e.response.json() if e.response else str(e)
        print("Error from Coherence API:", error_message)
        return jsonify({
            'message': 'Failed to fetch data from Coherence API',
            'error': error_message
        }), 500

if __name__ == '__main__':
    app.run(port=PORT)
