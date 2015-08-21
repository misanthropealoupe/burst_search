from multiprocessing import Queue
import numpy as np
#from . import arutil

class DataSource(object):
	def __init__(self, chunk_size, active = True):
		self._chunk_size = chunk_size
		self.queue = Queue()
		self.active = active
	def active(self):
		return self.active

class RtChimeSource(DataSource):
	#NOT DONE
	def __init__(self, stream_variable, chunk_size):
		DataSource(chunk_size)

class FileChimeSource(DataSource):
	def __init__(self, chunk_size, dat_path):
		DataSource(chunk_size)
		self._datfile = open(dat_path,'r')
		self.header = self.get_header() #convenience
		self.pk_dtype = mk_packet_dtype(self.header['nframe'], self.header['nfreq'], self.header['ninput'])
		self.header_dtype = mk_packet_dtype(0,0,0)
		self.tframes = 0

	def get_header(self):
		head_size = self.header_dtype.itemsize
		head_buf = self._datfile.read(head_size)

		self._datfile.seek(0,0)
		return np.fromstring(head_buf, dtype=header_dtype, count=1)[0]

	def pull_chunk(self):
		npk = self._chunk_size
		pksize = self.pk_dtype.itemsize
		read_size = pksize * npk
		buf = self._datfile.read(read_size)
		npkread = len(buf) / pksize

		if buf == '':
			self.active = False
			return

		self.tframes += npkread * pinfo['nframe']
		npb = np.fromstring(buf, dtype=pk_dtype)
		valid = npb['valid']

		if np.logical_and(valid != 0xffffffff, valid != 0).any():
            		#print "Corrupt data, or got confused about offset."
            		return

            	#Do not check for lost packets
            	#if (valid == 0).any():

		self.queue.put(npb)

	def __call__(self):
		self.pull_chunk()

	def __del__(self):
		del self.queue
		f.close()


def mk_packet_dtype(nframe, nfreq, ninput):

    packet_dtype = np.dtype([('valid', 'u4'), ('unused_header', '26b'),
                             ('n_frames', 'u4'), ('n_input', 'u4'), ('n_freq', 'u4'), ('offset_freq', 'u4'),
                             ('seconds', 'u4'), ('micro_seconds', 'u4'), ('seq', '<u4'), ('data', '(%i,%i,%i)u1' % (nframe, nfreq, ninput)) ])

    return packet_dtype