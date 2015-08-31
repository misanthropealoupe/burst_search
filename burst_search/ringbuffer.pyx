
import numpy as np
import time

cimport numpy as np
cimport cython

cimport cython.mem.PyMem_Free

np.import_array()

# These must match prototypes in src/ringbuffer.h
DTYPE = np.float32
ctypedef np.float32_t DTYPE_t

CM_DTYPE = np.int64
ctypedef np.int64_t CM_DTYPE_t

#ctypedef extern struct Peak;

# C prototypes.
cdef extern void PyMem_Free(void * ptr)

cdef extern void add_to_ring(DTYPE_t * indata, DTYPE_t * outdata, CM_DTYPE_t *chan_map,
		DTYPE_t *outdata, int * ring_t0, int chunk_size, int ntime, float delta_t, size_t nfreq,
		float freq0, float delta_f, int depth)

cdef extern int burst_get_num_dispersions(size_t nfreq, float freq0,
		float delta_f, int depth)

cdef extern int burst_depth_for_max_dm(float max_dm, float delta_t,
		size_t nfreq, float freq0, float delta_f)

cdef extern void burst_setup_channel_mapping(CM_DTYPE_t *chan_map, size_t nfreq,
		float freq0, float delta_f, int depth)

class DMData(object):
	"""Container for spectra and DM space data.

	"""

	@property
	def spec_data(self):
		return self._spec_data

	@property
	def dm_data(self):
		return self._dm_data

	@property
	def delta_t(self):
		return self._delta_t

	@property
	def freq0(self):
		return self._freq0

	@property
	def delta_f(self):
		return self._delta_f

	@property
	def dm0(self):
		return self._dm0

	@property
	def delta_dm(self):
		return self._delta_dm

	def __init__(self, spec_data, dm_data, delta_t, freq0, delta_f, dm0, delta_dm):
		self._spec_data = spec_data
		self._dm_data = dm_data
		self._delta_t = delta_t
		self._freq0 = freq0
		self._delta_f = delta_f
		self._dm0 = dm0
		self._delta_dm = delta_dm

	def __del__(self):
		del self._spec_data
		del self._dm_data

	@classmethod
	def from_hdf5(cls, group):
		delta_t = group.attrs['delta_t']
		freq0 = group.attrs['freq0']
		delta_f = group.attrs['delta_f']
		dm0 = group.attrs['dm0']
		delta_dm = group.attrs['delta_dm']
		spec_data = group['spec_data'][:]
		dm_data = group['dm_data'][:]
		return cls(spec_data, dm_data, delta_t, freq0, delta_f, dm0, delta_dm)


	def to_hdf5(self, group):
		group.attrs['delta_t'] = self.delta_t
		group.attrs['freq0'] = self.freq0
		group.attrs['delta_f'] = self.delta_f
		group.attrs['dm0'] = self.dm0
		group.attrs['delta_dm'] = self.delta_dm
		group.create_dataset('spec_data', data=self.spec_data)
		group.create_dataset('dm_data', data=self.dm_data)

class RingBuffer(object):
	@property
	def delta_t(self):
		return self._delta_t

	@property
	def nfreq(self):
		return self._nfreq

	@property
	def freq0(self):
		return self._freq0

	@property
	def delta_f(self):
		return self._delta_f

	@property
	def max_dm(self):
		return self._max_dm

	@property
	def ndm(self):
		return self._ndm

	@property
	def depth(self):
		return self._depth

	def __init__(self, chunk_length, buffer_length, delta_t, nfreq, freq0, delta_f, max_dm):
		#h1 = h.heap()
		cdef:
			float cdelta_t = delta_t
			int cnfreq = nfreq
			float cfreq0 = freq0
			float cdelta_f = delta_f
			float cmax_dm = max_dm
			int cchunk_length = chunk_length
			int cbuffer_length = buffer_length

			int depth = burst_depth_for_max_dm(cmax_dm, cdelta_t, cnfreq, cfreq0,
				cdelta_f)

			int cndm =  burst_get_num_dispersions(cnfreq, cfreq0, cdelta_f, depth)

			np.ndarray[ndim=1, dtype=CM_DTYPE_t] chan_map

			np.ndarray[ndim=2, dtype=DTYPE_t] ring_buffer
		chan_map = np.empty(2**depth, dtype=CM_DTYPE)
		ring_buffer = np.zeros(shape=(cndm, buffer_length), dtype=DTYPE)
		self.ring_buffer = ring_buffer
		#print "rb alloc"
		#print h.heap() - h1
		#del h1

		burst_setup_channel_mapping(<CM_DTYPE_t *> chan_map.data, cnfreq, cfreq0,
				cdelta_f, depth)
		self._chan_map = chan_map
		self._delta_t = delta_t
		self._nfreq = nfreq
		self._freq0 = freq0
		self._delta_f = delta_f
		self._max_dm = max_dm
		self._ndm = cndm
		self._depth = depth

		self._chunk_length = chunk_length
		self._buffer_length = buffer_length
		self._ringt0 = 0

	def __call__(self, np.ndarray[ndim=2, dtype=DTYPE_t] data not None):
		assert self._chunk_length == data.shape[1]
		#h1 = h.heap()
		cdef:
			int nfreq = self._nfreq
			int ntime = self._chunk_length

			float delta_t = self.delta_t
			float freq0 = self.freq0
			float delta_f = self.delta_f
			int ndm = self.ndm
			int depth = self.depth
			int chunk_size = self._chunk_length
			int ringt0 = self._ringt0

			np.ndarray[ndim=1, dtype=CM_DTYPE_t] chan_map
			np.ndarray[ndim=2, dtype=DTYPE_t] out
			np.ndarray[ndim=2, dtype=DTYPE_t] ring_buffer = self.ring_buffer

		chan_map = self._chan_map
		out = np.zeros(shape=(ndm, self._chunk_length + ndm), dtype=DTYPE)
		#print "dd setup"
		#print h.heap() - h1
		#ring_buffer = self._ring_buffer
		#rb_dat = self._ring_buffer.data
		#data_pad = np.zeros(shape=(self._nfreq, self._chunk_length + ndm), dtype=DTYPE)
		#data_pad[:,:self._chunk_length] = data[:,:]

		print self.ring_buffer.shape
		#print np.ascontiguousarray(data[:,:]).shape
		print "dedispersing"
		#h1 = h.heap()
		t1 = time.time()
		add_to_ring(
			<DTYPE_t *> data.data,
			<DTYPE_t *> out.data,
			<CM_DTYPE_t *> chan_map.data,
			<DTYPE_t *> ring_buffer.data,
			&ringt0,
			chunk_size,
			ntime,
			delta_t,
			nfreq,
			freq0,
			delta_f,
			depth,
			)
		delt = time.time() - t1
		print "dedisperse of {0}s took {1}s".format(self.delta_t*self._chunk_length, delt)
		print "fraction of real time: {0}".format(self.delta_t*float(self._chunk_length)/(delt))
		#print h.heap() - h1
		#save the state of ring
		self._ringt0 = ringt0
		self.ring_buffer = ring_buffer
		#PyMem_Free(ring_buffer.data)

		#h1 = h.heap()
		#dm_data = np.ascontiguousarray(out[:,:ntime])
		dm_data = np.ascontiguousarray(out[:,:])

		#needs accurate spacing
		spec_data = np.ascontiguousarray(data[:, :ntime])

		dm0 = 0
		delta_dm = (delta_t / 4150.
					/ abs(1. / freq0**2 - 1. / (freq0 + nfreq * delta_f)**2))

		out_cont = DMData(spec_data, dm_data, delta_t, freq0, delta_f, dm0,
						  delta_dm)

		#print "out products"
		#print h.heap() - h1
		return out_cont

