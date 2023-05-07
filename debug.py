import requests

url = 'http://localhost:5000/get_video_info'
data = {'video_id': 'test_video_id'}

response = requests.post(url, json=data)

print(response.status_code)
print(response.headers)
print(response.text)
