import requests

# Base URL of the device on your local network
BASE_URL = "http://30.201.216.198:8080"


def fetch_health_status():
    """Fetch health status from http://192.168.12.88:8080/health"""
    url = f"{BASE_URL}/health"
    #try:
        # Set a timeout (in seconds) so your script doesn't hang indefinitely if the server is down
    response = requests.get(url, timeout=5)

    # Raise an exception if the HTTP request returned an unsuccessful status code (e.g. 404, 500)
    response.raise_for_status()

    # Print or return raw text / JSON response depending on what /health returns
    print(f"[Health Check] Status Code: {response.status_code}")
    print(f"[Health Check] Response Body: {response.text}")
    return response.text

    #except requests.exceptions.RequestException as e:
    #    print(f"Error fetching health endpoint: {e}")
    #    return None


def fetch_nearest_gps():
    """Fetch coordinates from http://192.168.12.88:8080/nearest-gps"""
    url = f"{BASE_URL}/nearest-gps"
    #try:
    response = requests.get(url, timeout=5)
    response.raise_for_status()

        # Parse the JSON response: {"x": 10.0, "y": 50.0}
    gps_data = response.json()

        # Access individual fields from the dictionary
    x_coord = gps_data.get("x")
    y_coord = gps_data.get("y")

    print(f"[Nearest GPS] Parsed Data: {gps_data}")
    print(f"[Nearest GPS] X: {x_coord}, Y: {y_coord}")

    return gps_data

    #except requests.exceptions.RequestException as e:
    #    print(f"Error fetching nearest-gps endpoint: {e}")
    #    return None


if __name__ == "__main__":
    print("--- Checking Server Health ---")
    fetch_health_status()

    print("\n--- Fetching GPS Coordinates ---")
    gps_coordinates = fetch_nearest_gps()
    print(gps_coordinates)