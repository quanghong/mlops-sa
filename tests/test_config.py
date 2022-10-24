from app import model_prediction
import numpy as np

input_data = {
        "international_plan":1,	
        "voice_mail_plan":2,
        "number_vmail_messages":3,
        "number_customer_service_calls":4,
        "minutes":"1",
        "calls":2,
        "charge":4
}

class NotANumber(Exception):
    def __init__(self, message = "Values entered are not Number"):
        self.message = message 
        super().__init__(self.message)

def form_response(dict_request):
    data = dict_request.values()
    data = [list(map(float,data))]
    return data

def test_model_prediction(data=input_data):
    data = form_response(data)
    res = model_prediction(data)
    assert res != NotANumber().message
