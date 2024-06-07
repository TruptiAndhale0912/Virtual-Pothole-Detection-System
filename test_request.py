import requests
import base64

# Load an image and convert it to base64
with open('path_to_image.jpg', 'rb') as img_file:
    img_base64 = base64.b64encode(img_file.read()).decode('utf-8')

# Send a POST request
response = requests.post(
    'http://localhost:5000/analysis',
    data={'imageData': img_base64}
)

print(response.json())
