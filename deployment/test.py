import requests

url = "http://1161e797-86d9-48a9-bc66-e132f190c003.eastus.azurecontainer.io/score"

payload="{\"message\": \"{message:There was a problem at line 7\n,details:This may be due to the following reason:\n\n    Key 'Company Name' not found in record.\n\n,correctiveAction:To continue, edit the bot and fix the error. Then, try again.\n\n\nIf you continue to see this message, please contact your system administrator.,code:bot.execution.error}\"}"
headers = {
  'Content-Type': 'application/json'
}

response = requests.request("POST", url, headers=headers, data=payload)
#print(response.content)
print(response.text)