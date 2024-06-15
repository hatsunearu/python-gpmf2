from collections import namedtuple
from datetime import datetime, timedelta
from xml.etree import ElementTree as ET

import gpxpy
from . import parse
import numpy as np

class UnsupportedBlock(ValueError):
    pass


AcclBlock = namedtuple("AcclBlock", [
    'us_timestamp',
    'accel_xyz',
    'temperature',
    ])


GPS9Block = namedtuple("GPS9Block", [
    'us_timestamp',
    'gps_times',
    'lat',
    'long',
    'alt',
    'speed_2d',
    'speed_3d',
    'dop',
    'fix'
    ])

GPS5Block = namedtuple("GPS9Block", [
    'us_timestamp',
    'gps_timestamp',
    'lat',
    'long',
    'alt',
    'speed_2d',
    'speed_3d',
    'dop',
    'fix'
    ])



FIX_TYPE = {
    0: "none",
    2: "2d",
    3: "3d"
}

def parse_gps5_strm(strm_dict):
    if not 'GPS5' in strm_dict:
        raise UnsupportedBlock()
    
    gps_data = strm_dict['GPS5'].value.astype('float64') / strm_dict["SCAL"].value.astype('float64')

    latitude, longitude, altitude, speed_2d, speed_3d = gps_data.T

    return GPS5Block(
        us_timestamp=strm_dict['STMP'].value,
        gps_timestamp=strm_dict['GPSU'],
        lat=latitude,
        long=longitude,
        alt=altitude,
        speed_2d=speed_2d,
        speed_3d=speed_3d,
        dop=strm_dict['GPSP'].value.astype('float64') / 100,
        fix=strm_dict['GPSF'].value
    )



def process_gps9_strm(strm_dict):
    if not 'GPS9' in strm_dict:
        raise UnsupportedBlock()
    
    scale = strm_dict['SCAL'].value.astype('float64')
    

    # 5th row is days since 2000, 6th row is seconds since midnight
    gps_times = [ datetime.datetime(year=2000,month=1,day=1,tzinfo=datetime.timezone.utc) 
        + datetime.timedelta(days=int(v[5]), seconds=v[6].astype('float64') / scale[6])
        for v in strm_dict['GPS9'].value ]
    
    # rows 0-5 is lat, long, alt, speed2d, speed3d
    data1 = strm_dict['GPS9'].value[:, 0:5].astype('float64') / scale[0:5]

    return GPS9Block(
        us_timestamp=strm_dict['STMP'].value,
        gps_times=gps_times,
        lat=data1[:, 0],
        long=data1[:, 1],
        alt=data1[:, 2],
        speed_2d=data1[:, 3],
        speed_3d=data1[:, 4],
        dop=strm_dict['GPS9'].value[:, 7].astype('float64') / scale[6],
        fix=strm_dict['GPS9'].value[:, 8].astype('float64')
    )



def parse_accl_strm(strm_dict):

    if not 'ACCL' in strm_dict:
        raise UnsupportedBlock()

    # order is YXZ
    transform_matrix = np.array([
        [0, 1, 0],
        [1, 0, 0],
        [0, 0, 1]
    ]).astype('float64') / float(strm_dict['SCAL'].value)
    
    accel = strm_dict['ACCL'].value @ transform_matrix

    return AcclBlock(
        us_timestamp=strm_dict['STMP'].value,
        accel_xyz=accel,
        temperature=strm_dict['TMPC'].value,
    )

def get_frame_timecounter(cori_blocks, ignore_last_block=False):

    frames_in_block = []
    stmp_raw = []

    # CORI block contains multiple CORI datapoint per frame
    # each CORI block has one STMP
    for block in cori_blocks:
        stmp_block = [item.value for item in block if item.key == 'STMP']
        if len(stmp_block) == 0:
            raise ValueError(f'No STMP block detected in block {block}')
        stmp = stmp_block[0]
        stmp_raw.append(stmp)

        [cori_block] = [item for item in block if item.key == 'CORI']

        frames_in_block.append(cori_block.length.repeat)
    
    prev_stmp = stmp_raw[0]
    frame_timecounter = []
    for fib, stmp in zip(frames_in_block[0:-1], stmp_raw[1:]):
        # make linspace to create frame timecounter interpolated timings for the prev block
        # since the current STMP points to the first frame in the current block,
        # the length of the linspace has to be fib + 1
        # we only take the [0:-1] slice since we need to discard the last one 
        # which would be in the current block
        frame_timecounter.extend(np.linspace(prev_stmp, stmp, fib+1)[0:-1])
        
        prev_stmp = stmp

    # last block is unaccounted for, interpolate 2nd last block's timing
    # could be dangerous if you operate consecutive files at the same time
    # since time can become non monotonic
    if not ignore_last_block:
        delta = stmp_raw[-1] - stmp_raw[-2]
        numsamps = frames_in_block[-2] + 1 # second last block
        dt = delta / numsamps

        frame_timecounter.extend(
            stmp_raw[-1] + np.arange(0, frames_in_block[-1]) * dt
            )

    return frame_timecounter



def convert_strm_dict(klv_strm):
    if klv_strm.key != 'STRM':
        raise ValueError('called convert_strm_dict on a non-STRM KLV')
    
    d = {}
    for klv in klv_strm.value:
        if not klv.key in d:
            d[klv.key] = klv
        else:
            if isinstance(d[klv.key], list):
                d[klv.key].append(klv)
            else:
                d[klv.key] = [d[klv.key], klv]

    return d


def traverse_gpmf(klv_iter):
    for klv in klv_iter:
        if klv.key == 'STRM':
            yield convert_strm_dict(klv)
        elif klv.length.type == '\x00':
            yield from traverse_gpmf(klv.value)