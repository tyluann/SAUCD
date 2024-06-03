import os, sys


obj_list = [
    "animal",
    "building",
    "food",
    "furniture",
    "humanBodyFemale",
    "humanBodyMale",
    "humanFaceFemale",
    "humanFaceMale",
    "humanHand",
    "plant",
    "statue",
    "vehicle",
]

field_list = [
    'impulse_1', 'impulse_2', 'impulse_3', 'impulse_4', 
    'lowerResolution_1', 'lowerResolution_2', 'lowerResolution_3', 'lowerResolution_4', 
    'outlyingReconNoise_1', 'outlyingReconNoise_2', 'outlyingReconNoise_3', 'outlyingReconNoise_4', 
    'poissonReconNoise_1', 'poissonReconNoise_2', 'poissonReconNoise_3', 'poissonReconNoise_4', 
    'smoothness_1', 'smoothness_2', 'smoothness_3', 'smoothness_4', 
    'unproportionalScaling_1', 'unproportionalScaling_2', 'unproportionalScaling_3', 'unproportionalScaling_4', 
    'whiteNoise_1', 'whiteNoise_2', 'whiteNoise_3', 'whiteNoise_4'
]

render_list = ['Diff', 'Mid', 'Spec']
gender_list = ['man', 'woman', 'others']

import json

def parse_filename(filename):
    parts = filename[:-5].split('_')
    obj = parts[0]
    render = parts[1]
    user = parts[2]
    date = parts[3]
    time = parts[4]
    if len(parts) == 8:
        gender = ''
    else:
        gender = parts[5]
    nField = int(parts[-3][1:])
    nComp = int(parts[-2][1:])
    valid = True if parts[-1].startswith('T') else False
    error = -1
    if len(parts[-1][1:]) != 0:
        error = float(parts[-1][1:])

    return obj, render, user, date, time, gender, nField, nComp, valid, error

def save_field_list(infile, outfile):
    with open(infile, 'r') as f:
        data = json.load(f)
    with open(outfile, 'w') as f:
        json.dump(list(data.keys()), f)
    return data.keys()

def read_field_list(file="assets/field_list.json"):
    with open(file, 'r') as f:
        field_list = json.load(f)
    return field_list

if __name__ == "__main__":
    infile = "assets/user_study_results/animal_Diff_tyluan_20220727_080943_f28_s84_T.json"
    outfile = "assets/field_list.json"
    save_field_list(infile, outfile)