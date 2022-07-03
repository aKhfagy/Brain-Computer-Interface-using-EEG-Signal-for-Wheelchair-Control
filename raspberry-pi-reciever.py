import requests
import time
import MDD10A as HBridge


if __name__ == '__main__':
    try:
        
        predictions = [2, 2, 2, 3, 3, 3, 3, 1, 2, 3, 3, 3, 3, 3, 3, 3, 2, 2, 2]
        
        for prediction in predictions:
            url = "http://192.168.1.7:5000/direct_chair?prediction=" + str(prediction)
            requests.get(url)
            url = "http://192.168.1.7:5000/get_movement"
            response = requests.get(url)
            print(response.json())
            direction = response.json()
            status = direction['Status']
            if status == 'Validating':
                time.sleep(60)
                continue
            
            direction = direction['Direction']
            if prediction == 'right':
                # right
                HBridge.setMotorLeft(1)
                HBridge.setMotorRight(-1)
            elif prediction == 'left':
                # left
                HBridge.setMotorLeft(-1)
                HBridge.setMotorRight(1)
            elif prediction == 'forward':
                # forward
                HBridge.setMotorLeft(1)
                HBridge.setMotorRight(1)
            elif prediction == 'backward':
                # backwards
                HBridge.setMotorLeft(-1)
                HBridge.setMotorRight(-1)
            else:
                # stop
                HBridge.setMotorLeft(0)
                HBridge.setMotorRight(0)

            time.sleep(60)
    except requests.exceptions.ConnectionError:
        print('Connection Failed')

