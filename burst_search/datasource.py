from multiprocessing import Queue
import numpy as np

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
	frame_cadence = 2.56e-6
	def __init__(self, desired_cadence, dat_path):
		DataSource(chunk_size)
		self._datfile = open(dat_path,'r')
		self.set_num_packs(desired_cadence)
		#convenience
		self.header = self.get_header()
		self.nframes = self.header['nframe']
		self.nfreq = self.header['nfreq']
		self.pk_dtype = mk_packet_dtype(self.header['nframe'], self.header['nfreq'], self.header['ninput'])
		self.header_dtype = mk_packet_dtype(0,0,0)
		self.tframes = 0

	def get_header(self):
		head_size = self.header_dtype.itemsize
		head_buf = self._datfile.read(head_size)

		self._datfile.seek(0,0)
		return np.fromstring(head_buf, dtype=header_dtype, count=1)[0]

	def pull_chunk(self):
		self.queue.put(self.get_chunk())

	#get a block with a specific length time axis (in 'effective cadence' units)
	def get_intensity_cadence_block(self, num_t):
		npackets = num_t*self.npacks
		nframes = self.nframes
		sumsqr = np.zeros((self.nfreq, num_t),dtype=np.float32)
		for i in xrange(0,num_t):
			raw = get_chunk()['data']
			sumsqr[:,i] = np.sum(np.sum(np.sum(np.square(raw),axis=0),axis=0),axis=1)
		return sumsqr

	def set_num_packs(desired_cadence):
		self.npacks = int(round(desired_cadence/frame_cadence))/self.nframes
		self._chunk_size = self.npacks
		self.eff_cadence = self.npacks*self.nframes*frame_cadence
		return self.eff_cadence


	def get_chunk(self, npk=self_chunk_size):
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
            		return None
            	return npb
            	#Do not check for lost packets
            	#if (valid == 0).any():

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