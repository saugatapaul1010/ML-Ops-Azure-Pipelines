import requests
import json

req_sample = {"message": "In Google Chrome -assignment: We found an issue that prevents the bot from running.Search for 'preprocessing' on  https://docs.automationanywhere.com/csh?context=csh-preprocessing-error-messages to find the most common causes and what to do about them."}

# def test_ml_service(scoreurl):
#     assert scoreurl != None
#     headers = {'Content-Type':'application/json'}
#     resp = requests.post(scoreurl, json=json.loads(json.dumps(req_sample)), headers=headers)
#     assert resp.status_code == requests.codes["ok"]
#     assert resp.text != None
#     assert resp.headers.get('content-type') == 'application/json'
#     assert int(resp.headers.get('Content-Length')) > 0

# def test_prediction(scoreurl):
#     assert scoreurl != None
#     headers = {'Content-Type':'application/json'}
#     resp = requests.post(scoreurl, json=json.loads(json.dumps(req_sample)), headers=headers)
#     resp_json = json.loads(resp.text)
#     assert resp_json['output']['predicted_species'] == "1"



def test_ml_service(scoreurl, scorekey):
    assert scoreurl != None

    if scorekey is None:
        headers = {'Content-Type':'application/json'}
    else:
        headers = {'Content-Type':'application/json', 'Authorization':('Bearer ' + scorekey)}

    resp = requests.post(scoreurl, json=json.loads(json.dumps(req_sample)), headers=headers)
    assert resp.status_code == requests.codes["ok"]
    assert resp.text != None
    assert resp.headers.get('content-type') == 'application/json'
    assert int(resp.headers.get('Content-Length')) > 0


def test_prediction(scoreurl, scorekey):
    assert scoreurl != None

    if scorekey is None:
        headers = {'Content-Type':'application/json'}
    else:
        headers = {'Content-Type':'application/json', 'Authorization':('Bearer ' + scorekey)}

    resp = requests.post(scoreurl, json=json.loads(json.dumps(req_sample)), headers=headers)
    resp_json = json.loads(resp.text)
    assert resp_json['output']['predicted_Action_taken_to_solve'] == "1"







