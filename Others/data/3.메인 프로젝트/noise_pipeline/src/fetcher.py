import requests
from config.settings import NOISE_API_URL, HEADERS


def fetch_all_noise_data():
    response = requests.post(
        NOISE_API_URL,
        headers=HEADERS,
        timeout=10
    )
    response.raise_for_status()
    return response.json().get("listEnMenu", [])
