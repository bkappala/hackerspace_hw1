#
# CS 196 Data Hackerspace
# Assignment 1: Data Parsing and NumPy
# Due September 24th, 2018
#

import json
import csv
import numpy as np

def histogram_times(filename):
    import pandas as pd
    plane_df = pd.read_csv(filename)
    plane_df = plane_df[plane_df['Time'].notnull()]
    output = []
    # for i in range(0, len(plane_df['Time'])):
    #    if(pd.isnull(plane_df.iloc[i]['Time'])):
    #        pass
    #    else:
    #        plane_df.iloc[i]['Hour'] = str(plane_df.iloc[i]['Time']).split(":")[0]

    # plane_df['Hour'] = plane_df['Time'].apply(lambda x: x.split(":")[0])
    splitter = lambda x: x.split(":")[0]
    a = ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16', '17',
         '18', '19', '20', '21', '22', '23']
    plane_df['Hour'] = plane_df['Time'].apply(splitter)
    # plane_df['Hour'] = plane_df['Hour'].apply(converter)
    # print(plane_df.groupby('Hour').size())
    #print(plane_df)
    # print(plane_df.loc[:,'Hour'].value_counts())
    #plane_df['Hour'] = plane_df[plane_df['Hour'].isin(a)]
    valParse = (plane_df[plane_df['Hour'].isin(a)])
    valParse = valParse.loc[:, 'Hour']
    x = list(valParse)
    count = 0
    for i in a:
        count = x.count(i)
        output.append(count)
    return output

def weigh_pokemons(filename, weight):
    with open(filename) as f:
        data = json.load(f)
    output = []
    for i in range(0 , len(data['pokemon'])):
        pokemon_weight = data['pokemon'][i]['weight']
        pokemon_weight = float(pokemon_weight.split(" ")[0])
        if (pokemon_weight == weight):
            output.append(data['pokemon'][i]['name'])
    return output

def single_type_candy_count(filename):
    with open(filename) as f:
        data = json.load(f)
    sumCandies = 0
    for i in range(0 , len(data['pokemon'])):
        if(len(data['pokemon'][i]['type']) == 1) and ('candy_count' in data['pokemon'][i]):
            sumCandies = sumCandies + data['pokemon'][i]['candy_count']
    return sumCandies

def reflections_and_projections(points):
    ref_matrix = np.array([[1,0], [0,-1]], dtype = np.int64)
    print(points)
    output = ref_matrix.dot(points)
    print(output)
    output[1,:] = 2 + output[1,:]
    print(output)
   # rot_matrix = np.array([[np.cos(np.pi/2), np.negative(np.sin(np.pi/2))], [np.sin(np.pi/2), np.cos(np.pi/2)]], dtype = int)
    rot_matrix = np.array([[0, -1], [1, 0]], dtype = int)
    proj_matrix = np.multiply((1/10), np.array([[1, 3], [3,9]], dtype = float))
    output = rot_matrix.dot(output)
    print(output)
    output = proj_matrix.dot(output)
    print(output)
    return output

def normalize(image):
    maxval = np.max(image)
    minval = np.min(image)
    normal = (255/(maxval - minval))*(image - minval)
    return normal

def sigmoid_normalize(image):
    variance = image.var()
    sigmoidNormalized = 255*(1 + np.exp((1/(-variance))*(image - 128)))**(-1)
    return sigmoidNormalized
