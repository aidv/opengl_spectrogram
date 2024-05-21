typedef struct
{
	unsigned int channels;
	int PREPAD, POSPAD, POS1, POS2;
	unsigned int fftLen, minus1fftLen, NFFTDIV2, halfLen, PADDEDLEN;
	float *mInput, *mCQTBuf, *delayLineInterleaved;
	unsigned int *mBitRev;
	float *mSineTab;
	float *x_fft;
	float *fftBuf;
	float *window;
	float *real, *imag;
	float *mag;
	// Shared variable between all FFT length and modes and channel config
	unsigned int mInputPos;
	// Constant Q
	float *poles, *gainPoles;
	float *bufferMovedRe, *bufferMovedIm;
	float *x_zeroReal, *x_zeroImag;
	float *filIIRBufReal;
	float *filIIRBufImag;
	float *q_fft_frameReal;
	float *q_fft_frameImag;
} ConstantQTransform;
void PhaseRadarInit(ConstantQTransform *cqt, unsigned int channels, unsigned int frameLen);
void PhaseRadarProcessSamples(ConstantQTransform *cqt, float *x, unsigned int inSampleCount, char stftCQT, char draw);
void PhaseRadarFree(ConstantQTransform *cqt);