# Brain Computer Interface using EEG Signal for Wheelchair Control

TUAR data: [https://isip.piconepress.com/projects/tuh_eeg/html/downloads.shtml#c_tuar](https://isip.piconepress.com/projects/tuh_eeg/html/downloads.shtml#c_tuar)

A large electroencephalographic motor imagery dataset for electroencephalographic brain computer interfaces: [https://figshare.com/collections/A_large_electroencephalographic_motor_imagery_dataset_for_electroencephalographic_brain_computer_interfaces/3917698](https://figshare.com/collections/A_large_electroencephalographic_motor_imagery_dataset_for_electroencephalographic_brain_computer_interfaces/3917698)

# User Guide

- Start server on a device
- Record the IP address of the server
- Change IP address present in the URL variable in the mobile app, and raspberry-pi-reciever.py
- raspberry-pi-reciever.py has the code to run on raspberry pi to move the motors and update state for the app, to run it you have to use a raspberry pi 4, with Raspberry pi OS, each reading takes 60 seconds to finish
- server.py has all 2 buttons, one for tracking the EEG signals and the second's for the eye tracker
- The search page in the app is for choosing which model will be used in classification
- The raspberry-pi-reciever.py file is a demo without using live data as we were not able to use a headset to get the EEG data feed
- The app checks the state of the motors every 1 second
- You have to start each part of the app individually
- To view the models and view the accuracies please send an api request to '/view_model?model=[MODEL_NAME]&subject=[SUBJECT]' where you'll replace the subject and model name with the subject and model name you want to view the accuracy for
- You can predict the movement of the wheelchair by using '/predict?model=[MODEL_NAME]&data=[DATA_NAME]'
- You can also make the models and preprocess the data from the API