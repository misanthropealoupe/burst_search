#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <omp.h>
#include "dedisperse_gbt.h"
#include "dedisperse.h"

#ifndef max
  #define max( a, b ) ( ((a) > (b)) ? (a) : (b) )
#endif

#ifndef min
  #define min( a, b ) ( ((a) < (b)) ? (a) : (b) )
#endif

#define DM0 4148.8
#define NOISE_PERIOD 64
#define SIG_THRESH 30.0
#define THREAD 8
#define OMP_THREADS 8

extern typedef struct Peak;

/*--------------------------------------------------------------------------------*/
void copy_in_data(Data *dat, float *indata, int ndata)
{
  int npad=dat->ndata-ndata1;
  memset(dat->raw_data[0],0,dat->raw_nchan*dat->ndata*sizeof(float));
  //printf("npad is %d\n",npad);
  assert(ndata2>=npad);
  
  for (int i=0;i<dat->raw_nchan;i++) {
    for (int j=0;j<ndata1;j++) {

      //this line changes depending on memory ordering of input data
      //dat->raw_data[i*dat->ndata+j]=indata1[i*ndata1+j];      
#ifdef BURST_DM_NOTRANSPOSE
      dat->raw_data[i][j]=indata1[i*ndata1+j];      
#else
      dat->raw_data[i][j]=indata1[j*dat->raw_nchan+i];
#endif
    }
  }
}

Data *put_data_into_burst_struct(float *indata, size_t ntime, size_t nfreq, size_t *chan_map, int depth)
{
  
  Data *dat=(Data *)calloc(1,sizeof(Data));
  dat->raw_nchan=nfreq;
  int nchan=get_nchan_from_depth(depth);
  dat->nchan=nchan;
  //int nextra=get_burst_nextra(ntime2,depth);
  dat->ndata=ntime;
  dat->raw_data=matrix(dat->raw_nchan,dat->ndata);
  dat->chan_map=chan_map;
  dat->data=matrix(dat->nchan,dat->ndata);
  copy_in_data(dat,indata,ntime,indata2,ntime2);
  
  return dat;
}

void remap_data(Data *dat)
{					
  assert(dat->chan_map);
  memset(dat->data[0],0,sizeof(dat->data[0][0])*dat->nchan*dat->ndata);  
  for (int i=0;i<dat->raw_nchan;i++) {
    int ii=dat->chan_map[i];
    for (int j=0;j<dat->ndata;j++)
      dat->data[ii][j]+=dat->raw_data[i][j];
  }
}
/*--------------------------------------------------------------------------------*/

void make_rect_mat(float** mat, float* data, int rows, int cols){
	for(int i = 0; i < rows; i++)
		mat[i] = data + i*cols;
}
void make_triangular_mat(float** mat, float* data, int rows, int offset, int delta){
	mat[0] = data;
	for(int i = 1; i < rows; i++){
		mat[i] = &mat[i - 1] + offset + (i - 1)*delta;
	}
}

/* returns data starting at a ringt0 - chunk_size after one call
*   Note that ringt0 corresponds to the value of ringt0 after the call
*   to update_ring_buffer.
*/
void add_to_ring(float* indata, float* outdata, int* chan_map, float* ring_buffer_data,
	int ringt0, int chunk_size, int ring_length, float delta_t, size_t nfreq, float freq0, float delta_f, int depth){

	Data *dat=put_data_into_burst_struct(indata,chunk_size,nfreq,chan_map,depth);
	remap_data(dat);
	float** ring_buffer = malloc(sizeof(float*)*dat->nchan);
	make_rect_mat(ring_buffer,ring_buffer_data,dat->nchan,ring_length);

	//allocate the triangular matrix for output
	nfreq = nchan;
	float* tmp = malloc((nfreq*chunk_size + (nfreq*(nfreq - 1))/2)*sizeof(float));
	float** tmp_mat = malloc(nchan*sizeof(float*));
	make_triangular_mat(tmp_mat,tmp,chunk_size,1);
	dedisperse_lagged(dat->data,tmp,nchan,chunk_size);
	update_ring_buffer(tmp,ring_buffer,dat->nchan,chunk_size,ring_length,&ringt0);
	
	float ** out_mat = malloc(sizeof(float*)*dat->nchan);
	make_rect_mat(out_mat,outdata,dat->nchan,chunk_size);
	for(int i = 0; i < dat->nchan; i++)
		memcpy(out_mat[i],ring_buffer[i] + (ringt0 - chunk_length),(ringt0 - i));
	free(tmp);
}


/*
 * Incremental dedispersion is implemented as two steps:
 * dedisperse_lagged() -> udpate_ring_buffer().
 *
 * @inin is an (nchan, ndat) array
 * @outout is a non-rectangular array: outout[j] points to a buffer of length ndat+j (or larger)
 *
 * WARNING!! Caller must overallocate the 'inin' buffer so that inin[j] has length >= ndat+nchan-1
 * (i.e. inin must be large enough to store the outout array)
 *
 * In current implementation, nchan must be a power of 2, but there is no constraint on ndat.
 */

void dedisperse_lagged(float **inin, float **outout, int nchan, int ndat)
{
    assert(nchan >= 2);
    assert(ndat >= 1);
    
    // detects underallocation, in the common case where inin was allocated with matrix()
    assert(inin[1] - inin[0] >= nchan + ndat - 1);
    assert(outout[1] - outout[0] >= nchan + ndat - 1);
    
    int npass = get_npass(nchan);
    assert(nchan == (1 << npass));   // currently require nchan to be a power of two

    int bs = nchan;
    float **in = inin;
    float **out = outout;

    for (int i = 0; i < npass; i++) {    
  for (int j = 0; j < nchan; j += bs)
      dedisperse_kernel_lagged(in+j, out+j, bs, ndat + j/bs, j/bs);

  float **tmp=in;
  in = out;
  out = tmp;
  bs /= 2;
    } 

    // non-rectangular copy
    for (int j = 0; j < nchan; j++)
  memcpy(out[j], in[j], (ndat+j) * sizeof(float));
}

/*
 * Incremental dedispersion is implemented as two steps:
 * dedisperse_lagged() -> udpate_ring_buffer().
 *
 *   @chunk: This should be the output array from dedisperse_lagged() above.
 *
 *   @nchunk: Should be the same as the @npad argument to dedisperse_lagged().
 *
 *   @ring_buffer: an array of shape (nchan, nring), where nring >= nchan+nchunk.
 *
 *   @ring_t0: this keeps track of the current position in the ring buffer, and
 *       is automatically updated by this routine.  (It can be initialized to zero 
 *       before the first call.)
 *
 * When update_ring_buffer() returns, the following elements of the ring buffer
 * are "valid", i.e. all samples which contribute at the given DM have been summed.
 *
 *     ring_buffer[0][ring_t0-nring:ring_t0]
 *     ring_buffer[1][ring_t0-nring:ring_t0-1]
 *         ...
 *     ring_buffer[nchan][ring_t0-nring:ring_t0-nchan+1]
 *
 * where the index ranges are written in Python notation and are understood to be
 * "wrapped" in the ring buffer.  Invalid elements are in a state of partial summation 
 * and shouldn't be used for anything yet.
 */

void update_ring_buffer(float **chunk, float **ring_buffer, int nchan, int nchunk, int nring, int *ring_t0)
{
    assert(nring >= nchunk + nchan - 1);

    int t0 = *ring_t0;

    for (int j = 0; j < nchan; j++) {
#if 0
  // A straightforward implementation using "%" operator turned out to be too slow...
  for (int i = 0; i < j; i++)
      ring_buffer[j][t1+i] += chunk[j][i];   // note += here
  for (int i = j; i < nchunk+j; i++)
      ring_buffer[j][(t0-j+i+nring) % nring] = chunk[j][i];    // note = here
#else
  // ... so I ended up with the following "ugly but fast" implementation instead

  // Update starts at this position in ring buffer
  int t1 = (t0-j+nring) % nring;

  // The logical index range [t1:t1+j] is stored as [t1:t1+n1] and [0:j-n1]   (this defines n1)
  // t2 = position in the ring buffer at the end of this index range
  int n1, t2;
  if (t1+j < nring) {
      n1 = j;
      t2 = t1 + j;
  }
  else {   // wraparound case
      n1 = nring - t1;
      t2 = t1 + j - nring;
  }

  // The logical index range [t2:t2+nchunk] is stored as [t2:t2+n2] and [0:nchunk-n2]
  int n2 = (t2+nchunk < nring) ? nchunk : (nring - t2);

  for (int i = 0; i < n1; i++)
      ring_buffer[j][t1+i] += chunk[j][i];    // note += here
  for (int i = n1; i < j; i++)
      ring_buffer[j][i-n1] += chunk[j][i];    // note += here
  for (int i = 0; i < n2; i++)
      ring_buffer[j][t2+i] = chunk[j][j+i];   // note = here
  for (int i = n2; i < nchunk; i++)
      ring_buffer[j][i-n2] = chunk[j][j+i];   // note = here 
#endif
    }

    *ring_t0 = (t0 + nchunk) % nring;
}

/*--------------------------------------------------------------------------------*/