import numpy as np

cimport numpy as np
cimport cython

np.import_array()

# These must match prototypes in src/dedisperse.h
DTYPE = np.float32
ctypedef np.float32_t DTYPE_t

CM_DTYPE = np.int64
ctypedef np.int64_t CM_DTYPE_t

# C prototypes.
cdef extern int burst_get_num_dispersions(size_t nfreq, float freq0,
		float delta_f, int depth)

cdef extern int burst_depth_for_max_dm(float max_dm, float delta_t,
		size_t nfreq, float freq0, float delta_f)

cdef extern int  burst_dm_transform(DTYPE_t *indata1, DTYPE_t *indata2,
		CM_DTYPE_t *chan_map, DTYPE_t *outdata, size_t ntime1, int ntime2,
		float delta_t, size_t nfreq, float freq0, float delta_f, int depth,int jon)

cdef extern void burst_setup_channel_mapping(CM_DTYPE_t *chan_map, size_t nfreq,
		float freq0, float delta_f, int depth)

class Transformer(object):

class RingBuffer(Transformer):
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

	def __init__(self,chunk_size,buffer_size,delta_t, nfreq, freq0, delta_f, max_dm):

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
		ring_buffer = np.empty(shape=(cndm, buffer_length), dtype=DTYPE)


		burst_setup_channel_mapping(<CM_DTYPE_t *> chan_map.data, cnfreq, cfreq0,
				cdelta_f, depth)
		self._ring_buffer = ring_buffer
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

	def __call__(self, np.ndarray[ndim=2, dtype=DTYPE_t] data1 not None):
		cdef int nfreq = self._nfreq

		data2 = np.empty(shape=(nfreq, 0), dtype=DTYPE)

		if data1.shape[0] != data2.shape[0] or data1.shape[0] != self.nfreq:
			msg = ("Input data arrays must have frequency axes with length"
				   " nfreq=%d." % self.nfreq)
			raise ValueError(msg)


		cdef:
			int ntime1 = data1.shape[1]
			int ntime2 = data2.shape[1]

			float delta_t = self.delta_t
			float freq0 = self.freq0
			float delta_f = self.delta_f
			int ndm = self.ndm
			int depth = self.depth

			np.ndarray[ndim=1, dtype=CM_DTYPE_t] chan_map
			np.ndarray[ndim=2, dtype=DTYPE_t] out
				out = np.empty(shape=(ndm, ntime1), dtype=DTYPE)
		chan_map = self._chan_map

		cdef int ntime_out = burst_dm_transform(
			<DTYPE_t *> data1.data,
			<DTYPE_t *> data2.data,
			<CM_DTYPE_t *> chan_map.data,
			<DTYPE_t *> out.data,
			ntime1,
			ntime2,
			delta_t,
			nfreq,
			freq0,
			delta_f,
			depth,
			jon,
			)