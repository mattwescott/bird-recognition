import json

settings = json.load(file('SETTINGS.json'))

WAV_DIR = settings['WAV_DIR']
MFCC_DIR = settings['MFCC_DIR']
LABEL_DIR = settings['LABEL_DIR']
SUBMISSION_DIR = settings['SUBMISSION_DIR']
OUTPUT_DIR = settings['OUTPUT_DIR']
