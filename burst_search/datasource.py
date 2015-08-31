from multiprocessing import Queue
import numpy as np
import random
import sys
from simulate import inject_square_event_chunk, disp_delay

from guppy import hpy
h=hpy()

frame_cadence = 2.56e-6
def mk_packet_dtype(nframe, nfreq, ninput):

	packet_dtype = np.dtype([('valid', 'u4'), ('unused_header', '26b'),
					 ('n_frames', 'u4'), ('n_input', 'u4'), ('n_freq', 'u4'), ('offset_freq', 'u4'),
					 ('seconds', 'u4'), ('micro_seconds', 'u4'), ('seq', '<u4'), ('data', '(%i,%i,%i)u1' % (nframe, nfreq, ninput)) ])

	return packet_dtype

def decode(data):
	cdata = np.zeros(data.shape + (2,), dtype=np.int8)
	cdata[..., 0] = (data / 16).view(np.int8) - 8
	cdata[..., 1] = (data % 16).view(np.int8) - 8
	cdata = cdata.reshape((-1,) + cdata.shape[2:])
	ret = cdata.astype(dtype=np.float32)
	del cdata
	return ret

def dispersion_spread(dm,fmin,fmax):
	return disp_delay(fmin,dm) - disp_delay(fmax,dm)

def get_num_packs(desired_cadence, nframes):
	return int(round(desired_cadence/frame_cadence))/nframes

class DataSource(object):
	def __init__(self, chunk_size, active = True):
		self._chunk_size = chunk_size
		self.queue = Queue()
		self.active = active
	def active(self):
		return self.active

	def __del__(self):
		del self.queue

class RandSource(DataSource):
	def __init__(self, nfreq, chunk_size, same_data=False, sim=False, sim_prob = 0.1, flux=0.1, fmax = 800.0, fmin = 400.0,dm=300.0, cadence=0.001):
		DataSource(chunk_size)
		self._chunk_size = chunk_size
		self._same_data = same_data
		self._nfreq = nfreq
		self._df = (fmax - fmin)/float(nfreq)
		self.t = 0

		self._sim = sim
		self._sim_prob = sim_prob
		self._sim_t0 = 0
		self._event = False
		self._event_dm = dm
		self._fmax = fmax
		self._fmin = fmin
		self._flux = flux
		self._dt = cadence
		if same_data:
			self._dat = np.random.rand(self._nfreq,self._chunk_size).astype(np.float32)

	def get_block(self,useless=0):
		if self._same_data:
			dat = self._dat
		else:
			dat = np.random.rand(self._nfreq,self._chunk_size).astype(np.float32)

		if(self._sim):
			if not self._event:
				if random.random() < self._sim_prob:
					self._event = True
					self._sim_t0 = self.t
			if self._event:
				t0 = self._sim_t0 - self.t
				if abs(t0) >= dispersion_spread(self._event_dm,self._fmin,self._fmax)/self._dt:
					self._event = False

				if self._event:
					print "injecting event"
					inject_square_event_chunk(dat, t0=t0, t_width=2, 
						chunk_length = self._chunk_size,dm=self._event_dm,
						flux=self._flux,df=self._df, fmax = self._fmax, fmin = self._fmin, dt=self._dt)

		self.t += self._chunk_size
		return dat

class RtChimeSource(DataSource):
	#NOT DONE
	def __init__(self, stream_variable, chunk_size):
		DataSource(chunk_size)

class FileChimeSource(DataSource):
	def __init__(self, desired_cadence, dat_path,freq0=800.0):
		DataSource(0)
		self._datfile = open(dat_path,'r')
		#convenience
		self.header_dtype = mk_packet_dtype(0, 0, 0)
		self.header = self.get_header()
		self.pk_dtype = mk_packet_dtype(self.header['n_frames'], 
			self.header['n_freq'], self.header['n_input'])
		self.nframes = self.header['n_frames']
		self.set_num_packs(desired_cadence)
		self._chunk_size = self.npacks
		self.nfreq = self.header['n_freq']
		self.freq0 = freq0
		self.pk_dtype = mk_packet_dtype(self.header['n_frames'], self.header['n_freq'], self.header['n_input'])
		self.header_dtype = mk_packet_dtype(0,0,0)
		self.tframes = 0

	def get_header(self):
		head_size = self.header_dtype.itemsize
		head_buf = self._datfile.read(head_size)

		self._datfile.seek(0,0)
		return np.fromstring(head_buf, dtype=self.header_dtype, count=1)[0]

	def pull_chunk(self):
		self.queue.put(self.get_chunk())

	#get a block with a specific length time axis (in 'effective cadence' units)
	def get_block(self, num_t):
		npackets = num_t*self.npacks
		nframes = self.nframes
		h1 = h.heap()
		sumsqr = np.zeros((self.nfreq, num_t), dtype=np.float32)
		for i in xrange(0,num_t):
			print "chunk"
			chunk = self.get_chunk()
			print h.heap() - h1
			h2 = h.heap()
			print "dat"
			dat = chunk['data']
			print h.heap() - h2
			print "decode"
			h2 = h.heap()
			raw = decode(dat)
			print h.heap() - h2
			del dat
			sumsqr[:,i] = np.sum(np.sum(np.sum(np.square(raw),axis=0),axis=2),axis=1)
			del chunk
			del raw
		print h.heap() - h1
		print sys.getsizeof(sumsqr)
		return sumsqr

	def set_num_packs(self, desired_cadence):
		self.npacks = get_num_packs(desired_cadence,self.nframes)
		self._chunk_size = self.npacks
		self.eff_cadence = self.npacks*self.nframes*frame_cadence
		return self.eff_cadence

	def get_chunk(self, npk=None):
		if npk == None: npk = self._chunk_size
		pksize = self.pk_dtype.itemsize
		read_size = pksize * npk
		buf = self._datfile.read(read_size)
		npkread = len(buf) / pksize

		if buf == '':
			self.active = False
			return

		self.tframes += npkread * self.header['n_frames']
		npb = np.fromstring(buf, dtype=self.pk_dtype)
		del buf
		valid = npb['valid']

		if np.logical_and(valid != 0xffffffff, valid != 0).any():
			#print "Corrupt data, or got confused about offset."
			return None
		return npb

	#def __call__(self):
	#	self.pull_chunk()

	def __del__(self):
		#del self.queue
		self._datfile.close()