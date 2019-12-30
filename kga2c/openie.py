import json
import requests
import logging


def call_stanford_openie(sentence):
    url = "http://localhost:9000/"
    querystring = {
        "properties": "%7B%22annotators%22%3A%20%22openie%22%7D",
        "pipelineLanguage": "en"}
    try:
        response = requests.request("POST", url, data=sentence, params=querystring)
        response = json.JSONDecoder().decode(response.text)
    except:
        print("OpenIE error")
        logging.debug('OpenIE Error: ' + sentence)
        response = {"sentences": ""}
    return response


def call_stanford_pos(sentence):
    url = "http://localhost:9000/"
    querystring = {
        "properties": "%7B%22annotators%22%3A%20%22openie%22%7D",
        "pipelineLanguage": "en"}
    try:
        response = requests.request("POST", url, data=sentence, params=querystring)
        response = json.JSONDecoder().decode(response.text)
    except:
        print("OpenIE error")
        logging.debug('OpenIE Error: ' + sentence)
        response = {"sentences": ""}
    return response