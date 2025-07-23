import pickle
import requests

response = requests.post("http://34.122.51.94:9091/lime", files={"file": open("explain_params.pkl", "rb")}, headers={"token": "54614611"})
print(response.json())