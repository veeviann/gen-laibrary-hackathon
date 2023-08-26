from requests.auth import HTTPBasicAuth
import requests
from dotenv import find_dotenv, load_dotenv
import os

load_dotenv(find_dotenv())

PRIMO_API_KEY = os.environ.get('PRIMO_API_KEY', None)

base_url = "https://api-ap.hosted.exlibrisgroup.com"
collection_id = "6168549700002600"
service_id = "6268549690002600"
extended_url = f"https://api-ap.hosted.exlibrisgroup.com/almaws/v1/electronic/e-collections/{collection_id}/e-services/{service_id}?apikey={PRIMO_API_KEY}"

url = f"https://api-ap.hosted.exlibrisgroup.com/primo/v1/search?vid=65SMU_INST%3ASMU_NUI&tab=Everything&scope=Everything&q=any%2Ccontains%2Cethical%20ai&lang=eng&offset=0&limit=10&sort=rank&pcAvailability=true&getMore=0&conVoc=true&inst=65SMU_INST&skipDelivery=true&disableSplitFacets=true&apikey={PRIMO_API_KEY}"
# headers = {'Accept': 'application/json'}

# files = {'file': open('filename', 'rb')}

response = requests.get(url=extended_url)
print(response)
