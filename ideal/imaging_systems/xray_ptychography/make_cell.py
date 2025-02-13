import numpy as np
import math
from scipy.interpolate import interp1d
from scipy.ndimage import rotate
from scipy.io import loadmat
from skimage.transform import resize
import scipy.io as sio


def parse_matlab_data(energy=7e3):
    file_path = 'scatteringfactors.m'
    with open(file_path, 'r', encoding='latin-1') as file:
        matlab_code = file.read()
    f1 = {}
    f2 = {}

    lines = matlab_code.split('\n')
    elem = None
    
    for line in lines:
        if line.startswith('%%') or not line.strip():
            continue
        if line.startswith('sc'):
            if(elem is not None):
                this_dict[elem] = interp1d(kev_list, val_list, kind='linear', fill_value='extrapolate')(energy)

            _, elem, val = line.strip().split('.')
            kev_list = []
            val_list = []
            if(val == 'f1'):
                this_dict = f1
            if(val == 'f2'):
                this_dict = f2
            
        else:
            kev, val = line.strip().split()
            kev_list.append(float(kev))
            val_list.append(float(val))
    return f1, f2


def make_single_cell(energy,cell_thickness,h2o_thickness,dx = 2.29 / 1e3,sz = 1024):
    cell_size = sz * dx / 2  # um

    #print('Cell size is %f microns' % cell_size)
    fovcell = int(round(cell_size / dx))
    avo_num = 6.022e23
    lambda_val = 1.2398 / energy  # um
    re = 2.818e-9  # classical electron radius (um)
    beta = re / (2 * np.pi) * lambda_val ** 2
    delta = re / (2 * np.pi) * lambda_val ** 2

    if fovcell % 2 == 1:
        fovcell += 1

    if sz % 2 == 1:
        sz += 1

    cell_thickness_map = gen_random_cell()

    resized_cell_thickness_map = np.pad(cell_thickness_map, ((0, fovcell - 256), (0, fovcell - 256)), 'constant')
    cell_thickness_map = resized_cell_thickness_map[:sz, :sz]
    
    cell_thickness_map[cell_thickness_map < np.max(cell_thickness_map) * 0.01] = 0

    h2o_thickness = 0.5
    cell_thickness = 0.5
    h2o_thickness_map = np.ones_like(cell_thickness_map)

    f1, f2 = parse_matlab_data()
    f1['all'] = (50 * f1['h'] + 30 * f1['c'] + 9 * f1['n'] + 10 * f1['o'] + 1 * f1['s']) / (50 + 30 + 9 + 10 + 1)
    f2['all'] = (50 * f2['h'] + 30 * f2['c'] + 9 * f2['n'] + 10 * f2['o'] + 1 * f2['s']) / (50 + 30 + 9 + 10 + 1)
    f1['h2o'] = (2 * f1['h'] + 1 * f1['o']) / 3
    f2['h2o'] = (2 * f2['h'] + 1 * f2['o']) / 3

    cell_N = 50 + 30 + 9 + 10 + 1
    cell_mol_weight = (50 + 30 * 12 + 9 * 14 + 10 * 16 + 32) * 1e-3
    cell_density = 1.35e-3
    cell_na = cell_density * avo_num * cell_N / cell_mol_weight / 1e12

    sampcell_delta = cell_na * (cell_thickness * cell_thickness_map) * (delta * f1['all'])
    sampcell_beta = cell_na * (cell_thickness * cell_thickness_map) * (beta * f2['all'])

    h2o_N = 3
    h2o_mol_weight = (2 * 1 + 1 * 16) * 1e-3
    h2o_density = 1e-3
    h2o_na = h2o_density * avo_num * h2o_N / h2o_mol_weight / 1e12

    h2o_delta = h2o_na * (h2o_thickness * h2o_thickness_map) * (delta * f1['h2o'])
    h2o_beta = h2o_na * (h2o_thickness * h2o_thickness_map) * (beta * f2['h2o'])

    sampcell_amp = np.exp(-2 * np.pi * (sampcell_beta + h2o_beta) / lambda_val)
    sampcell_ph = np.exp(2 * 1j * np.pi * (np.ones_like(sampcell_delta) - (sampcell_delta + h2o_delta)) / lambda_val)

    sampcell = sampcell_amp * sampcell_ph

    return sampcell


def gen_random_cell(sz = 800):
    def pad(array, size):
        padded_array = np.zeros((size, size))
        start = (size - array.shape[0]) // 2
        end = start + array.shape[0]
        padded_array[start:end, start:end] = array
        return padded_array

    def makeSphere(radius, size, x, y, z):
        X, Y, Z = np.meshgrid(np.arange(size),np.arange(size),np.arange(size))
        sphere = np.zeros((size, size, size))
        sphere[(X-x)**2+(Y-y)**2+(Z-z)**2 <= radius**2] = 1 
        return sphere

    def makeStick(length, width, angle, size, x, y):
        stick = np.zeros((size, size))
        half_length = length // 2
        half_width = width // 2
        stick[size // 2 - half_length:size // 2 + half_length, size // 2 - half_width:size // 2 + half_width] = 1
        stick = rotate(stick, angle, reshape=False)
        stick = np.roll(stick, y - size // 2, axis=0)
        stick = np.roll(stick, x - size // 2, axis=1)
        return stick

    def smooth3D(array, sigma):
        from scipy.ndimage import gaussian_filter
        filt = gaussian_filter(array, sigma)
        return filt


    # Load the data
    data = loadmat('vesicle_3D.mat')['img']  # Replace 'variable_name' with the actual variable name in the .mat file
    # Processing
    sz2 = 128
    sz3 = 256

    cell = resize(np.sum(np.abs(data).astype(np.float32), axis=1), (sz,sz))
    cell = cell / np.max(cell)

    cell_max = np.max(cell[cell != 0])

    range_val = 40
    n_items = 10
    coordinates = np.random.randint(sz2 // 2 - range_val, sz2 // 2 + range_val, (n_items, 3))

    radii = np.random.randint(3, 16, n_items)
    lengths = np.random.randint(5, 60, n_items)
    widths = np.random.randint(1, 6, n_items)
    angles = np.random.randint(0, 360, n_items)
    obj = np.zeros((sz2, sz2, sz2))
    sticks = np.zeros((sz2, sz2))

    # Generate random features to superimpose on vesicle structure
    for n in range(len(radii)):
        sphere = makeSphere(radii[n], sz2, coordinates[n, 0], coordinates[n, 1], coordinates[n, 2])
        stick = makeStick(lengths[n], widths[n], angles[n], sz2, coordinates[n, 1], coordinates[n, 2])
        sticks += stick
        obj += sphere

    obj = resize(np.sum(obj, axis=2), (sz3,sz3))
    sticks = resize(sticks, (sz3,sz3))

    obj = obj + smooth3D(sticks * np.max(obj), 0.3)
    obj = np.pad(obj, ((sz - obj.shape[0])//2, (sz - obj.shape[1])//2))

    obj = cell_max*(obj - np.min(obj))/np.ptp(obj)

    cell_thickness_map = cell + obj

    return cell_thickness_map
    
    

    
