# MP_Group_70
Repository for Major project of group 70 Team members : Adikar Charvi Sree Teja, Adithya Pillai, Aditi C, Akash K S 

BCI-Controlled Smart Bulb System using SSVEP and ErrP on Raspberry Pi

Overview:
This project enables a Raspberry Pi to control a Philips WiZ smart bulb using EEG signals. It uses two BCI components: SSVEP at 7.5 Hz to toggle the bulb on or off, and ErrP to validate or revert the action. EEG files are sent wirelessly to the Raspberry Pi from a phone or PC.

Working:

Step 1: Upload EEG File
A mobile device or computer sends an EEG .easy file to the Raspberry Pi at:
http://<pi-ip>:5000/upload
The Pi saves the file and automatically starts processing.

Step 2: SSVEP Action
The Pi analyzes the EEG for the 7.5 Hz SSVEP signal. If detected, it toggles the WiZ smart bulb. The bulb turns ON if it was OFF, and OFF if it was ON.

Step 3: ErrP Validation
The same EEG file is then tested for ErrP. If ErrP is detected, meaning the user did not intend the action, the Pi reverts the bulb to its original state. If ErrP is not detected, the SSVEP action remains unchanged.

Step 4: Notifications
The Raspberry Pi can send Firebase Cloud Messaging notifications. Phones subscribed to a topic can receive alerts such as emergency notifications or bulb state confirmation.

Features:

Wireless EEG file upload from mobile or PC

SSVEP-based ON/OFF bulb control

ErrP-based automatic correction of wrong actions

WiZ smart bulb control using local WiFi

Optional Firebase push notifications

Running the System:
Start the Raspberry Pi server by running:
python3 rpi_bci_server.py

Upload an EEG file using:
curl -F "file=@sample.easy" http://<pi-ip>:5000/upload

The Pi analyzes the file, performs SSVEP and ErrP processing, updates bulb state, and sends notifications if enabled.

Requirements:
Install required Python packages:
pip install flask numpy scipy scikit-learn pywizlight joblib xgboost imblearn firebase_admin


