from collections import namedtuple
import datetime
from xml.etree import ElementTree as ET

from abc import ABC, abstractmethod
import numpy as np


class UnsupportedBlock(ValueError):
    pass

# basis change for common ORIOs
ORIOMatrix = {
    'YXZ': [
        [0, 1, 0],
        [1, 0, 0],
        [0, 0, 1]
    ],
    'ZXY': [
        [0, 0, 1],
        [1, 0, 0],
        [0, 1, 0]
    ],
    'XZY': [
        [1, 0, 0],
        [0, 0, 1],
        [0, 1, 0]
    ]
}

# get orin string (ex. zxY) and returns XYZ correction matrix
def orin_matrix_decode(orin_str):
    mat = np.array(ORIOMatrix[orin_str.upper()]).astype(np.float64)
    for i in range(0,2):
        if orin_str[i].islower():
            mat[i] = -mat[i]
    return mat
    


# time series data with camera time as the time basis (floats)
CameraTimeseries = namedtuple('CameraTimeseries', ['time', 'data'])

# time series data with GPS time as the time basis (datetimes)
GpsTimeseries = namedtuple('GpsTimeseries', ['time', 'data'])


# given a list of (timestamp, index) pairs, return an interpolated timestamp of all the reference sample indicies.
# vector specifies from the first pair's index to the last pair's index
def timestamp_index_pair_to_time(pairs, extend=0):
    tt = []
    prev_ts, prev_ind = pairs[0]
    for ts, ind in pairs[1:]:
        # linearly interpolate 
        # P1 = (prev_ts, 0)
        # P2 = (ts, samples in block)
        # samples in block is ind-prev_ind+1
        # discard last sample because that should be included in the next loop
        tt.append(np.linspace(prev_ts, ts, ind-prev_ind+1)[:-1])
        prev_ts, prev_ind = ts, ind
    
    # extension: if there's an extra block at the end
    # the timestamp is ambiguous since there's only one timestamp at the beginning
    # and nothing to constrain the end of the block
    # use the delta-t (sample interval) of the previous block to extrapolate
    # timestamp for the final block
    dt = tt[-1][-1] - tt[-1][-2]
    tt.append(np.linspace(prev_ts, prev_ts + dt * extend, extend+1)[:-1])
    
    return np.concatenate(tt)


class STRMBlockProcessor(ABC):
    def __init__(self):
        self.strm_data = []

    # block name to match STRM stream as relevant to the current processor
    @staticmethod
    @abstractmethod
    def get_blockname():
        pass

    # consume one STRM stream dictionary and record STRM block intermediate output
    # using subclass _process() method
    def process(self, strm_dict: dict):
        if not self.get_blockname() in strm_dict:
            raise UnsupportedBlock()
        processed_blk_data = self._process(strm_dict)
        if processed_blk_data:
            self.strm_data.append(processed_blk_data)
    
    # abstractmethod to define what the block output should be
    @staticmethod
    @abstractmethod
    def _process(strm_dict):
        pass

    # abstractmethod to export the entirity of the data, with appropriate timestamping
    @abstractmethod
    def export(self):
        pass

    # returns number of blocks processed
    def block_count(self):
        return len(self.strm_data)

class GenericTimeseriesBlockProcessor(STRMBlockProcessor):
    @staticmethod
    @abstractmethod
    def get_blockname():
        pass

    @staticmethod
    @abstractmethod
    def get_timeseries_name():
        pass
    
    @staticmethod
    @abstractmethod
    def _process(strm_dict: dict):
        pass
    
    def export(self):
        if self.block_count() == 0:
            return None
        
        data_blocks = []
        # list of tuples: (timestamp, sample index)
        # timestamp is the microsecond time stamp from the first sample in the block that comes from the STRM
        timestamp_index_pairs = []
        ind = 0
        for blk in self.strm_data:
            us_timestamp, data_block = blk
            data_blocks.append(data_block)
            timestamp_index_pairs.append((us_timestamp, ind))
            ind += np.shape(data_block)[0]

        if len(self.strm_data) == 0:
            return {}
        
        tt = timestamp_index_pair_to_time(timestamp_index_pairs, extend=np.shape(data_block)[0])
        timeseries = np.concatenate(data_blocks)
        # timeseries = timeseries[0:len(tt), :]
        # print(f"{timestamp_index_pairs[0]=}")
        # print(f"{timestamp_index_pairs[1]=}")
        # print(f"{timestamp_index_pairs[-1]=}")
        # print(f"{len(tt)-1=}")
        # print(f"{tt[len(tt)-1]=}")
        # print(f"{np.shape(timeseries)}")

        assert(np.shape(timeseries)[0] == len(tt))
        
        return {self.get_timeseries_name(): CameraTimeseries(tt, timeseries)}

AcclBlock = namedtuple("AcclBlock", [
    'us_timestamp',
    'accel_xyz',
    #'temperature',
    ])
class AcclBlockProcessor(GenericTimeseriesBlockProcessor):
    @staticmethod
    def get_blockname():
        return 'ACCL'
    
    @staticmethod
    def get_timeseries_name():
        return 'acceleration'
    
    @staticmethod
    def _process(strm_dict: dict):
        transform_matrix = np.array(
            orin_matrix_decode(strm_dict['ORIN'].value)
        ) / float(strm_dict['SCAL'].value)
        
        accel = strm_dict['ACCL'].value @ transform_matrix

        return AcclBlock(
            us_timestamp=strm_dict['STMP'].value,
            accel_xyz=accel,
            # temperature=strm_dict['TMPC'].value,
        )
    

GravBlock = namedtuple("GravBlock", [
    'us_timestamp',
    'grav_xyz',
    #'temperature',
    ])
class GravBlockProcessor(GenericTimeseriesBlockProcessor):
    @staticmethod
    def get_blockname():
        return 'GRAV'
    
    @staticmethod
    def get_timeseries_name():
        return 'gravity'
    
    @staticmethod
    def _process(strm_dict: dict):
        # grav vector order is XZY, at least on Hero 11
        transform_matrix = np.array(np.array(
            ORIOMatrix['XZY']
        )).astype('float64') / float(strm_dict['SCAL'].value)
        
        grav = strm_dict['GRAV'].value @ transform_matrix

        return GravBlock(
            us_timestamp=strm_dict['STMP'].value,
            grav_xyz=grav
        )
    
GyroBlock = namedtuple("GyroBlock", [
    'us_timestamp',
    'gyro_xyz'
    ])
class GyroBlockProcessor(GenericTimeseriesBlockProcessor):
    @staticmethod
    def get_blockname():
        return 'GYRO'
    
    @staticmethod
    def get_timeseries_name():
        return 'gyroscope'
    
    @staticmethod
    def _process(strm_dict: dict):
        transform_matrix = np.array(np.array(
            orin_matrix_decode(strm_dict['ORIN'].value)
        )).astype('float64') / float(strm_dict['SCAL'].value)
        
        gyro = strm_dict['GYRO'].value @ transform_matrix

        return GyroBlock(
            us_timestamp=strm_dict['STMP'].value,
            gyro_xyz=gyro
        )

CoriBlock = namedtuple("CoriBlock", [
    'us_timestamp',
    'framenumbers'
    ])
class CoriBlockProcessor(GenericTimeseriesBlockProcessor):
    def __init__(self):
        super().__init__()
        self.framecount = 0
    
    @staticmethod
    def get_blockname():
        return 'CORI'
    
    @staticmethod
    def get_timeseries_name():
        return 'frametime'
    
    def _process(self, strm_dict: dict):
        current_frame = self.framecount
        frames_in_block = np.shape(strm_dict['CORI'].value)[0]
        self.framecount += frames_in_block

        return CoriBlock(
            us_timestamp=strm_dict['STMP'].value,
            framenumbers=np.arange(current_frame, current_frame + frames_in_block)
        )

GPS5Block = namedtuple("GPS5Block", [
    'us_timestamp',
    'gps_timestamp',
    'gps_data',
    'dop',
    'fix'
    ])
class Gps5BlockProcessor(STRMBlockProcessor):
    @staticmethod
    def get_blockname():
        return 'GPS5'
    
    @staticmethod
    def _process(strm_dict: dict):
        if len(strm_dict['GPS5'].value) == 0:
            return None
        gps_data = strm_dict['GPS5'].value.astype('float64') / strm_dict["SCAL"].value.astype('float64')

        return GPS5Block(
            us_timestamp=strm_dict['STMP'].value,
            gps_timestamp=strm_dict['GPSU'].value,
            gps_data=gps_data,
            dop=strm_dict['GPSP'].value.astype('float64') / 100,
            fix=strm_dict['GPSF'].value
        )

    def export(self):
        if self.block_count() == 0:
            return None
        
        gps_data_blks = []

        gps_timestamps = []
        dops = []
        fixes = []

        # timestamp associated with gps data indicies
        timestamp_gps_index_pairs = []
        # timestamp associated with once-a-block data indicies (e.g. dop and fix)
        tt_blk = []

        ind = 0
        for gps5_blk in self.strm_data:
            us_timestamp = gps5_blk.us_timestamp

            gps_data_blks.append(gps5_blk.gps_data)

            gps_timestamps.append(gps5_blk.gps_timestamp)
            dops.append(gps5_blk.dop)
            fixes.append(gps5_blk.fix)

            tt_blk.append(us_timestamp)


            timestamp_gps_index_pairs.append((us_timestamp, ind))
            ind += np.shape(gps5_blk.gps_data)[0]
        
        tt_gps_data = timestamp_index_pair_to_time(timestamp_gps_index_pairs)
        gps_data = np.concatenate(gps_data_blks)[0:len(tt_gps_data), :]
        
        return {
            'latlongalt': CameraTimeseries(tt_gps_data, gps_data[:, 0:3]),
            '2dspeed': CameraTimeseries(tt_gps_data, gps_data[:, 3]),
            '3dspeed': CameraTimeseries(tt_gps_data, gps_data[:, 4]),
            'gps_timestamp': CameraTimeseries(tt_blk, gps_timestamps),
            'dop': CameraTimeseries(tt_blk, dops),
            'fix': CameraTimeseries(tt_blk, fixes)
        }

GPS9Block = namedtuple("GPS9Block", [
    'us_timestamp',
    'gps_time',
    'gps_data'
    ])
class Gps9BlockProcessor(STRMBlockProcessor):
    @staticmethod
    def get_blockname():
        return 'GPS9'
    
    @staticmethod
    def _process(strm_dict: dict):
        if len(strm_dict['GPS9'].value) == 0:
            return None
        scale = strm_dict['SCAL'].value.astype('float64')
    
        # 5th row is days since 2000, 6th row is seconds since midnight
        gps_time = [ datetime.datetime(year=2000,month=1,day=1,tzinfo=datetime.timezone.utc) 
            + datetime.timedelta(days=int(v[5]), seconds=v[6].astype('float64') / scale[6])
            for v in strm_dict['GPS9'].value ]
        

        return GPS9Block(
            us_timestamp=strm_dict['STMP'].value,
            gps_time=gps_time,
            # rows 0-4 is lat, long, alt, 2D speed, 3D speed, 
            # rows 5-6 is days since 2000, secs since midnight (ms precision)
            # rows 7-8 is DOP, fix (0, 2D or 3D)
            gps_data=(strm_dict['GPS9'].value[:, 0:9].astype('float64') / scale)[:, (0,1,2,3,4,7,8)]
        )
    def export(self):
        if self.block_count() == 0:
            return None
        
        gps_data_blks = []
        gps_time_blks = []
        tt_us = []
        # gps times that are coincident with camera timestamps
        gps_time_correlation = []

        for gps9_blk in self.strm_data:
            gps_data_blks.append(gps9_blk.gps_data)
            gps_time_blks.append(gps9_blk.gps_time)
            tt_us.append(gps9_blk.us_timestamp)
            gps_time_correlation.append(gps9_blk.gps_time[0])
        
        gps_data = np.concatenate(gps_data_blks)
        gps_times = np.concatenate(gps_time_blks)
        tt_us = np.array(tt_us)
        
        return {
            'latlongalt': GpsTimeseries(gps_times, gps_data[:, 0:3]),
            '2dspeed': GpsTimeseries(gps_times, gps_data[:, 3]),
            '3dspeed': GpsTimeseries(gps_times, gps_data[:, 4]),
            'gps_timestamp': CameraTimeseries(tt_us, gps_time_correlation),
            'dop': GpsTimeseries(gps_times, gps_data[:, 5]),
            'fix': GpsTimeseries(gps_times, gps_data[:, 6])
        }

# STRMs have underlying KLV(s) inside it. 
# this function returns a dictionary for each underlying KLV, with the KLV key being the kay
# and KLV itself being the value.
def convert_strm_dict(klv_strm):
    if klv_strm.key != 'STRM':
        raise ValueError('called convert_strm_dict on a non-STRM KLV')
    
    d = {}
    for klv in klv_strm.value:
        # klv key not already in dictionary; assume klv is a singleton
        if not klv.key in d:
            d[klv.key] = klv
        else:
            # klv is a list already, append
            if isinstance(d[klv.key], list):
                d[klv.key].append(klv)
            # klv was a singleton, but a new one showed up, create a list
            else:
                d[klv.key] = [d[klv.key], klv]

    return d


# generator to extract all STRM blocks and send them through convert_strm_dict and yield STRM-child klv values
def iterate_gpmf(klv_iter):
    for klv in klv_iter:
        if klv.key == 'STRM':
            yield convert_strm_dict(klv)
        elif klv.length.type == '\x00':
            yield from iterate_gpmf(klv.value)


def consume_klv_iter(klv_iter):
    processor_classes = [AcclBlockProcessor, GyroBlockProcessor, GravBlockProcessor, CoriBlockProcessor, Gps5BlockProcessor, Gps9BlockProcessor]
    processors = {pc: pc() for pc in processor_classes}

    for d in iterate_gpmf(klv_iter):
        for p in processors.values():
            try:
                p.process(d)
            except UnsupportedBlock:
                pass
        
    # GPS9 block was found in data, which means GPS5 blocks should be ignored as GPS9 blocks are better
    if processors[Gps9BlockProcessor].block_count() > 0:
        del processors[Gps5BlockProcessor]

    retval = {}
    for p in processors.values():
        export = p.export()
        if export:
            retval.update(export)
        
    
    return retval