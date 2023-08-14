# Copyright 2024 Crown in Right of Canada
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import numcodecs
import warnings
import numpy as np

# Define helper functions for quantization and dequantization
try:
    # If numba is available, use it to create compiled versions of the functions.  The best numba function
    # has a C/Fortran-like structure with loops, so it performs poorly in raw Python, but the best vector
    # function does not gain much from numba
    import numba
    numba.config.THREADING_LAYER = 'threadsafe'
    
    # Numba note: fastmath=True causes LLVM to assume that nans don't exist, so this will require some
    # care when quantizing actual nan values (missing/out-of-bounds data)
    @numba.jit([numba.int32[:,:,:](numba.types.Array(numba.types.float32,3,'C',readonly=True),#numba.float32[:,:,:],
                                   numba.int64,numba.float32[:],
                                   numba.float32[:]),
                # Disable float64 compatibility for now because of fixed-type nan check
                #numba.int32[:,:,:](numba.float64[:,:,:],
                #                   numba.int64,numba.float64[:],
                #                   numba.float64[:])
                ],
           nopython=True,fastmath=True)#,parallel=True)
    def quantizer(buf,nbits,plane_min,plane_max):
        '''Encode buffer through linear quantization, using per-layer minima
        and maxima; each 2D plane of the buffer is an independent stream.  Internal processing 
        happens at float32 precision, so this encoder is only meaningful for nbits <= 23'''
        
        Nplanes = buf.shape[0]
        Ni = buf.shape[1]
        Nj = buf.shape[2]

        buftype = type(buf[0,0,0])

        # Maximum quantized level (0 -- MAX_LEVEL-1 inclusive, giving 2**nbits values)
        MAX_LEVEL = numba.int32(2**nbits - 1)
        # Sigil value for NaNs
        NAN_SIGIL = numba.int32(MAX_LEVEL+1)
        MAX_LEVELf = buftype(MAX_LEVEL)

        quantized_int = np.empty((Nplanes,Ni,Nj),dtype=np.int32)
        for ii in numba.prange(Nplanes):
            plane_delta = plane_max[ii] - plane_min[ii]
            if (plane_delta <= 0): 
                plane_delta = 1
            plane_scale = MAX_LEVELf / plane_delta
            for jj in numba.prange(Ni):
                for kk in range(Nj):
                    # with fastmath=True, np.isnan is assumed to always be false.  However, we really really do
                    # want to check for nans, so we go via the int32 representation of a float:
                    #if (np.isnan(buf[ii,jj,kk])):
                    if ((buf.view(np.int32)[ii,jj,kk] & 0x7f80_0000) == 0x7f80_0000):
                        # Catches ±inf and nans
                        quantized_int[ii,jj,kk] = NAN_SIGIL
                    else:
                        quantized_int[ii,jj,kk] = np.rint(plane_scale * (buf[ii,jj,kk] - plane_min[ii]))
                    

        return quantized_int

    @numba.jit(numba.float32[:,:,:](numba.int32[:,:,:],numba.int64,numba.float32[:],numba.float32[:]),
        nopython=True,fastmath=True)#,parallel=True)
    def dequantizer(buf,nbits,plane_min,plane_max):
        '''Takes quantized integer values and re-scale them to their float32 equivalents'''
        Nplanes = buf.shape[0]
        Ni = buf.shape[1]
        Nj = buf.shape[2]

        MAX_LEVEL = numba.int32(2**nbits - 1)
        NAN_SIGIL = numba.int32(MAX_LEVEL+1)

        buf_out = np.empty((Nplanes,Ni,Nj),dtype=np.float32)
        for ii in numba.prange(Nplanes):
            delta = np.float32((plane_max[ii]-plane_min[ii])/MAX_LEVEL)
            for jj in numba.prange(Ni):
                for kk in range(Nj):
                    if (buf[ii,jj,kk] == NAN_SIGIL):
                        buf_out[ii,jj,kk] = np.nan
                    else:
                        buf_out[ii,jj,kk] = buf[ii,jj,kk]*delta + plane_min[ii]

        return buf_out
except ImportError:
    # numba isn't available, so define vector functions as a fallback
    def quantizer(buf,nbits,plane_min,plane_max):
        '''Encode buffer through quantization and linear prediction, using per-layer minima
        and maxima; each 2D plane of the buffer is an independent stream.  Internal processing 
        happens at float32 precision, so this encoder is only meaningful for nbits <= 23'''
        
        Nplanes = buf.shape[0]
        Ni = buf.shape[1]
        Nj = buf.shape[2]
        
        # Maximum quantized level (0 -- MAX_LEVEL-1 inclusive, giving 2**nbits values)
        MAX_LEVEL = np.int32(2**nbits - 1)
        # Sigil value for NaNs
        NAN_SIGIL = np.int32(MAX_LEVEL+1)
        MAX_LEVELf = float(MAX_LEVEL)
        
        # quantized_int = jnp.empty((Nplanes,Ni,Nj),dtype=np.int32)
        plane_delta = np.where((plane_max - plane_min) > 0, plane_max - plane_min, 1)                
        
        # # Quantize the array per-plane
        plane_scale = MAX_LEVELf / plane_delta
        # quantized_f = np.round(MAX_LEVEL * ((buf - plane_min[:,np.newaxis,np.newaxis])/(plane_delta[:,np.newaxis,np.newaxis])))
        quantized_f = np.rint(plane_scale[:,None,None] * (buf - plane_min[:,None,None]))
        # # Mark any NANs by the sigil value 
        np.nan_to_num(quantized_f,copy=False,nan=NAN_SIGIL)
        quantized_int = quantized_f.astype(np.int32)
        
        return quantized_int 

    def dequantizer(buf,nbits,plane_min,plane_max):
        '''Takes quantized integer values and re-scale them to their float32 equivalents'''
        Nplanes = buf.shape[0]
        Ni = buf.shape[1]
        Nj = buf.shape[2]

        MAX_LEVEL = numba.int32(2**nbits - 1)
        NAN_SIGIL = numba.int32(MAX_LEVEL+1)

        delta = ((plane_max - plane_min)/MAX_LEVEL).astype(np.float32)
        buf_out = buf*delta[:,None,None] + plane_min[:,None,None]
        buf_out[buf == NAN_SIGIL] = np.nan

        return buf_out

@numba.vectorize([numba.uint32(numba.int32)],nopython=True)
def negabinary(binary):
    '''Encode a signed 32-bit value (or array thereof) into base negative two, 
    following https://en.wikipedia.org/wiki/Negative_base#Shortcut_calculation'''
    Schroeppel2 = np.uint32(0xAAAAAAAA)
    bu32 = np.uint32(binary)
    return ((bu32 + Schroeppel2) ^ Schroeppel2)

@numba.vectorize([numba.int32(numba.uint32)],nopython=True)
def binanegary(negabinary):
    '''Convert a 32-bit value from base negative two to two's complement (signed) form'''
    Schroeppel2 = np.uint32(0xAAAAAAAA)
    bu32 = negabinary
    return ((bu32 ^ Schroeppel2) - Schroeppel2)

@numba.jit([numba.int32[:,:,:](numba.types.Array(numba.types.int32, 3, 'A', aligned=True, readonly=True))],nopython=True,nogil=True)
def lorenzo2d(A):
    '''Perform Loernzo encoding (lexical prediction based on S/W/SW values) on a 2D array'''
    out = np.zeros_like(A)
    for k in range(0,A.shape[0]):
        for i in range(1,A.shape[2]):
            out[k,0,i] = A[k,0,i] - A[k,0,i-1]
        for j in numba.prange(1,A.shape[1]):
            out[k,j,0] = A[k,j,0] - A[k,j-1,0]
            for i in range(1,A.shape[2]):
                out[k,j,i] = A[k,j,i] - A[k,j-1,i] - A[k,j,i-1] + A[k,j-1,i-1]
        out[k,0,0] = A[k,0,0]
    return out

@numba.jit([numba.int32[:,:,:](numba.int32[:,:,:])],nopython=True,nogil=True)
def unlorenzo2d(A):
    '''Invert Lorenzo encoding on a 2D array'''
    out = np.zeros_like(A)
    Nk = A.shape[0]
    Nj = A.shape[1]
    Ni = A.shape[2]
    for kk in range(Nk):
        # Initialize first row of output
        for ii in range(Ni):
            out[kk,0,ii] = A[kk,0,ii]
        # Cumulative sum along first axis
        for ii in numba.prange(Ni):
            for jj in range(1,Nj):
                out[kk,jj,ii] = A[kk,jj,ii] + out[kk,jj-1,ii]
    
        # # Cumulative sum along second axis
        for jj in numba.prange(A.shape[1]):
            for ii in range(1,A.shape[2]):
                out[kk,jj,ii] = out[kk,jj,ii] + out[kk,jj,ii-1]

    return out


class LayerQuantizer(numcodecs.abc.Codec):
    '''LayerQuantizer: dynamic plane-based quantization and linear prediction

    The LayerQuantizer compressor is inspired by the compression of the .fstd files, where a floating-point
    field undergoes a level-based quantization.  Per level (everything but the last two dimensions), the
    encoder records the field minimum and maximum, then it quantizes the field using `nbits` bits.  Nans
    are enoded as (2**nbits + 1).  The resulting integer stream is passed to Blosc for compression.

    By default, the layer minima and maxima are based on their true values, per encoded chunk.  Optionally,
    this can be replaced by fstd-compatible quantization, which expands the quantized range to the next
    largest power of 2.  This prevents data loss when re-encoding these values, whether by encoding an already-
    compressed fstd file or by encoding a subset of a once-encoded layer.

    This quantizer also applies a lossless predictor to the quantized values, to further reduce on-disk storage
    costs beyond what is naively provided by the entropy coder.  The fstd format uses Lorenzo encoding 
    (doi:10.1111/1467-8659.00681), which is also used here.  Direct Lorenzo encoding does not result in meaningful
    compression, but combining the Lorenzo-encoded value with a base-negative-two representation does, since it
    packs small-magnitude values (positive or negative) into small-mangnitude unsigned values with long runs of
    binary zero.'''

    codec_id = 'layerquantizer0.3b'
    def __init__(self,nbits=16,transform='Lorenzo',blosc_cname='zstd',blosc_clevel=5,in_id=codec_id,pow2_range=False):
        super().__init__()
        assert(in_id == self.codec_id)
        self.nbits = nbits
        self.blosc_cname = blosc_cname # In testing, zstd was better than lz4
        self.blosc_clevel = blosc_clevel
        self.bloscer = numcodecs.Blosc(cname=blosc_cname,clevel=blosc_clevel,shuffle=1)
        self.pow2_range = pow2_range
        self.transform = transform
    def encode(self,ibuf):
        '''Encode buffer through layer quantization; each 2D plane of the buffer is quantized
        independently.  This encoder is only valid for float32, so nbits is only meaningful
        for nbits <= 24; any larger value will result in no quantization'''
        assert(ibuf.dtype == np.float32)
        # Create a view of the input buffer so that shape modifications are non-destructive to the
        # input array
        buf = ibuf.view()

        if (self.nbits > 24):
            # Trivial encoding, just apply blosc to the field
            return self.bloscer.encode(buf)

        # Preserve a record of the input buffer shape
        input_shape = buf.shape

        # Reshape the buffer to (nplanes, ni, nj) format, since the quantization is effectively 3D
        buf.shape = ((-1,) + tuple(buf.shape[-2:]))
        nplanes = buf.shape[0]

        # Look for the array minimum and maximum by plane
        with warnings.catch_warnings():
            # Suppress a warning message if an entire plane is nans
            warnings.filterwarnings('ignore', r'All-NaN (slice|axis) encountered')
            plane_min = np.nanmin(buf,axis=(1,2))
            plane_max = np.nanmax(buf,axis=(1,2))
        # If an entire plane is nan (can happen with chunking), set the min and max to both be 0
        np.nan_to_num(plane_min,copy=False,nan=0)
        np.nan_to_num(plane_max,copy=False,nan=0)

        # Get the per-plane dynamic range
        plane_delta = plane_max - plane_min

        # Fix up the dynamic range, assigning a minimum range to any plane that had 0 dynamic range
        plane_adjdelta = plane_delta.copy()
        # If min=max=0, then set max=1
        plane_adjdelta[(plane_delta == 0) & (plane_min == 0)] = 1
        # Otherwise, set delta=max(|min|,2*min)
        plane_adjdelta[(plane_delta == 0) & (plane_min != 0)] = np.maximum(np.abs(plane_min[(plane_delta == 0) & (plane_min != 0)]),
                                                                           2*plane_min[(plane_delta == 0) & (plane_min != 0)])        

        plane_max[plane_delta == 0] = (plane_min[plane_delta == 0] + plane_adjdelta[plane_delta == 0]).astype(np.float32)
        plane_delta = plane_adjdelta

        if (self.pow2_range):
            # Force range to be a power of 2, for compatibility with fstd quantization.
            # Currently, we know that the field spans [min, max], inclusive of boundaries,
            # but now we want to find new_max such that the field spans [min, new_max) 
            # (note _not_ inclusivve of top boundary) and new_amax - min is a power of 2.

            # The net effect is to ensure that (delta)/2**nbits is also a power of 2,
            # so quantization levels are spaced evenly.  Additionally, re-quantizing
            # a subset of this stream will not change the field values, since the 
            # quantization delta will either remain the same (if the included range is
            # large enough) or fall by a precise power of 2 (if the included range
            # shrinks sufficiently)

            # However, one side effect is that the field will not use the full range of
            # quantized values.  This effect is most notable for fields like 'cloud fraction'
            # which have a natural range of [0,1] inclusive.  This quantization forces
            # the effective range to [0,2), losing one effective bit.

            # The current fstd code also adjusts the minimum to truncate the law few bits
            # of the mantissa in order to accomplish all of the quantization with integer
            # (fixed-point) math, but this shouldn't be necessary here.  If we're
            # re-encoding an fstd file, we'll already see the truncated minimum.

            # New range value, inclusive of minimum but exclusive of maximum.  The log2
            # would cause a problem if plane_delta were 0, but those cases have been
            # already corrected above
            plane_delta = 2**(1 + np.floor(np.log2(plane_delta)))

            # Use this range to adjust the plane maximum.  plane_max is still
            # notionally inclusive, so it must be adjusted by (2^N-1)/2^N to
            # representt the maximum quantizable value
            plane_max = (plane_min + (plane_delta)*(2**self.nbits - 1)/(2**self.nbits)).astype(np.float32)

        quantized_int = quantizer(buf,self.nbits,plane_min,plane_max)

        if (self.transform == 'Lorenzo'):
            quantized_int = negabinary(lorenzo2d(quantized_int))

        # Create the output buffer
        outbuf = np.empty((3 + 2*nplanes + buf.size),dtype=np.int32)

        # Encode chunk shape
        outbuf[0] = nplanes
        outbuf[1:3] = buf.shape[1:]

        # Encode plane maxima
        outbuf[3:(3+nplanes)] = plane_min.view(np.int32)
        outbuf[(3+nplanes):(3+2*nplanes)] = plane_max.view(np.int32)

        # Encode the quantized buffer values
        outbuf[(3+2*nplanes):] = quantized_int.ravel()

        # Output stream format:
        # [nplanes, nx, ny, mins[nplanes], max[nplanes], bitstream
        return self.bloscer.encode(outbuf)
        
        # return outbuf
    def decode(self,buf,out=None):
        '''Decode the encoded input bytestream'''
        intstream = np.frombuffer(self.bloscer.decode(buf),dtype=np.int32)
        # Get chunk size
        nplanes = intstream[0]
        nx = intstream[1]
        ny = intstream[2]

        if (out is not None):
            assert(out.size == nplanes*nx*ny)
            out.shape = (nplanes,nx,ny)
        else:
            out = np.empty((nplanes,nx,ny),dtype=np.float32)

        # Retrieve plane extrema
        plane_min = intstream[3:(3+nplanes)].view(np.float32)
        plane_max = intstream[(3+nplanes):(3+2*nplanes)].view(np.float32)
        plane_delta = plane_max - plane_min

        # Get a 3D view of the integer stream for re-scaling
        int_field = intstream[(3+2*nplanes):].view()
        int_field.shape = (nplanes,nx,ny)

        if (self.transform == 'Lorenzo'):
            int_field = unlorenzo2d(binanegary(int_field.view(np.uint32)))

        # Maximum quantized level (0 -- MAX_LEVEL inclusive)
        MAX_LEVEL = 2**self.nbits - 1
        # Sigil value for NaNs
        NAN_SIGIL = MAX_LEVEL+1

        out[:] = plane_min[:,np.newaxis,np.newaxis] + np.float32(((1/MAX_LEVEL)*(plane_delta[:,np.newaxis,np.newaxis])*int_field))
        # Re-assign any NANs

        NAN_SIGIL = 2**self.nbits + 1
        out[int_field == NAN_SIGIL] = np.nan
        
        return out
    def get_config(self):
        config = {'id' : self.codec_id, 
                'nbits' : self.nbits,
                'blosc_cname' : self.blosc_cname,
               'blosc_clevel' : self.blosc_clevel,
               'transform' : self.transform}
        if (self.pow2_range):
            config['pow2_range'] = True
        return config
    @classmethod
    def from_config(cls,config):
        return cls(**config)
    
numcodecs.registry.register_codec(LayerQuantizer)
