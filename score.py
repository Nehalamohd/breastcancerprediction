import joblib
import json
import numpy as np
def init():
    global model
    model = joblib.load("model.joblib")

def run(data):
    try:
        
        data = json.loads(data)
        inputs = np.array(data["data"])
       
        predictions = model.predict(inputs)
        return json.dumps({"result": predictions.tolist()})
    except Exception as e:
        return json.dumps({"error": str(e)})
    

if __name__== "__main__":
    init()
    test_input=json.dumps({"data":[[17.99,10.38,122.8,1001,0.1184,0.2776,0.3001,0.1471,
                                       0.2419,0.07871,1.095,0.9053,8.589,153.4,0.006399,
                                       0.04904,0.05373,0.01587,0.03003,0.006193,25.38,
                                       17.33,184.6,2019,0.1622,0.6656,0.7119,0.2654,
                                       0.4601,0.1189]]})
    print(run(test_input))