import numpy as np

cimport numpy as np
cimport cython


np.import_array()


# These mush match prototypes in src/dedisperse_gbt.h
DTYPE = np.float32
ctypedef np.float32_t DTYPE_t


# C prototypes.
cdef extern size_t find_peak_wrapper(float *data, int nchan, int ndata,
        float *peak_snr, int *peak_channel, int *peak_sample, int *peak_duration)
cdef extern size_t find_peak_wrapper_triangle(float *data, int nchan, int ndata,
        float *peak_snr, int *peak_channel, int *peak_sample, int *peak_duration)


def sievers_find_peak(data, low_dm_exclude=1):

    cdef np.ndarray[ndim=2, dtype=DTYPE_t] dm_data
    cdef int ndm = data.dm_data.shape[0]
    cdef int ntime = data.dm_data.shape[1]

    cdef float peak_snr
    cdef int peak_dm
    cdef int peak_time
    cdef int peak_duration

    dm_data = data.dm_data[low_dm_exclude:,:]
    ndm -= low_dm_exclude

    cdef size_t ret = find_peak_wrapper(
            <DTYPE_t *> dm_data.data,
            ndm,
            ntime,
            &peak_snr,
            &peak_dm,
            &peak_time,
            &peak_duration,
            )

    peak_dm += low_dm_exclude

    return peak_snr, (peak_dm, peak_time), peak_duration

def sievers_find_peak_triangle(data, low_dm_exclude=1):

    #could be quicker with a triangle copy
    #and a triangular matrix
    cdef np.ndarray[ndim=2, dtype=DTYPE_t] dm_data
    cdef int ndm = data.dm_data.shape[0]
    cdef int ntime = data.dm_data.shape[1]

    cdef float peak_snr
    cdef int peak_dm
    cdef int peak_time
    cdef int peak_duration

    dm_data = data.dm_data[low_dm_exclude:,:]
    ndm -= low_dm_exclude

    cdef size_t ret = find_peak_wrapper_triangle(
            <DTYPE_t *> dm_data.data,
            ndm,
            ntime,
            &peak_snr,
            &peak_dm,
            &peak_time,
            &peak_duration,
            )

    peak_dm += low_dm_exclude

    return peak_snr, (peak_dm, peak_time), peak_duration

