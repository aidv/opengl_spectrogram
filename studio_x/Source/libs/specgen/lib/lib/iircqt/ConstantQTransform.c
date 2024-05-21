#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include "ConstantQTransform.h"
#ifndef min
#define min(a,b) (((a)<(b))?(a):(b))
#endif
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#ifndef M_2PI
#define M_2PI (M_PI * 2.0)
#endif
unsigned int LLIntegerLog2(unsigned int v)
{
	unsigned int i = 0;
	while (v > 1)
	{
		++i;
		v >>= 1;
	}
	return i;
}
unsigned LLRevBits(unsigned int x, unsigned int bits)
{
	unsigned int y = 0;
	while (bits--)
	{
		y = (y + y) + (x & 1);
		x >>= 1;
	}
	return y;
}
void LLbitReversalTbl(unsigned *dst, int fftLen)
{
	unsigned int bits = LLIntegerLog2(fftLen);
	for (int i = 0; i < fftLen; ++i)
		dst[i] = LLRevBits(i, bits);
}
void LLsinHalfTblFloat(float *dst, int fftLen)
{
	const double twopi_over_n = 6.283185307179586476925286766559 / fftLen;
	for (int i = 0; i < fftLen; ++i)
		dst[i] = (float)sin(twopi_over_n * i);
}
void LLdiscreteHartleyFloat(float *A, const int nPoints, const float *sinTab)
{
	int i, j, n, n2, theta_inc, nptDiv2;
	float alpha, beta;
	// FHT - stage 1 and 2 (2 and 4 points)
	for (i = 0; i < nPoints; i += 4)
	{
		const float	x0 = A[i];
		const float	x1 = A[i + 1];
		const float	x2 = A[i + 2];
		const float	x3 = A[i + 3];
		const float	y0 = x0 + x1;
		const float	y1 = x0 - x1;
		const float	y2 = x2 + x3;
		const float	y3 = x2 - x3;
		A[i] = y0 + y2;
		A[i + 2] = y0 - y2;
		A[i + 1] = y1 + y3;
		A[i + 3] = y1 - y3;
	}
	// FHT - stage 3 (8 points)
	for (i = 0; i < nPoints; i += 8)
	{
		alpha = A[i];
		beta = A[i + 4];
		A[i] = alpha + beta;
		A[i + 4] = alpha - beta;
		alpha = A[i + 2];
		beta = A[i + 6];
		A[i + 2] = alpha + beta;
		A[i + 6] = alpha - beta;
		alpha = A[i + 1];
		const float beta1 = 0.70710678118654752440084436210485f*(A[i + 5] + A[i + 7]);
		const float beta2 = 0.70710678118654752440084436210485f*(A[i + 5] - A[i + 7]);
		A[i + 1] = alpha + beta1;
		A[i + 5] = alpha - beta1;
		alpha = A[i + 3];
		A[i + 3] = alpha + beta2;
		A[i + 7] = alpha - beta2;
	}
	n = 16;
	n2 = 8;
	theta_inc = nPoints >> 4;
	nptDiv2 = nPoints >> 2;
	while (n <= nPoints)
	{
		for (i = 0; i < nPoints; i += n)
		{
			int theta = theta_inc;
			const int n4 = n2 >> 1;
			alpha = A[i];
			beta = A[i + n2];
			A[i] = alpha + beta;
			A[i + n2] = alpha - beta;
			alpha = A[i + n4];
			beta = A[i + n2 + n4];
			A[i + n4] = alpha + beta;
			A[i + n2 + n4] = alpha - beta;
			for (j = 1; j < n4; j++)
			{
				float	sinval = sinTab[theta];
				float	cosval = sinTab[theta + nptDiv2];
				float	alpha1 = A[i + j];
				float	alpha2 = A[i - j + n2];
				float	beta1 = A[i + j + n2] * cosval + A[i - j + n] * sinval;
				float	beta2 = A[i + j + n2] * sinval - A[i - j + n] * cosval;
				theta += theta_inc;
				A[i + j] = alpha1 + beta1;
				A[i + j + n2] = alpha1 - beta1;
				A[i - j + n2] = alpha2 + beta2;
				A[i - j + n] = alpha2 - beta2;
			}
		}
		n <<= 1;
		n2 <<= 1;
		theta_inc >>= 1;
	}
}
void linspace(double *x, int n, double a, double b, int len)
{
	int i;
	double d = (b - a) / (double)(n - 1);
	for (i = 0; i < len; i++)
		x[i] = a + i * d;
}
double diff_of_products(double a, double b, double c, double d)
{
	double w = d * c;
	double e = fma(-d, c, w);
	double f = fma(a, b, -w);
	return f + e;
}
void solve_quadratic_roots(double a, double b, double c, double *x0, double *x1)
{
	double q = -0.5 * (b + copysign(sqrt(diff_of_products(b, b, 4.0*a, c)), b));
	*x0 = q / a;
	*x1 = c / q;
}
void calculate_poles(double *thetas, double Q, int NFFT, double slope, float *poles)
{
	int i;
	const double db = 3.0;
	// Hyperbola : (y-max_frac_pi*pi)(y-x) = -softness, that saturates the tau to a maximum value
	const double softness = 0.005;
	const double max_frac_pi = 0.6;
	double rr[2];
	double tau;
	const double w = 1.0 / pow(10.0, db / 20.0);
	for (i = 0; i < (NFFT >> 1); i++)
	{
		tau = Q * (M_2PI * M_PI) / NFFT / thetas[i];
		// The hyperbola solutions are calculated
		solve_quadratic_roots(1.0, -(tau + max_frac_pi * M_PI), M_PI * tau * max_frac_pi - softness, &rr[0], &rr[1]);
		// the smallest solution is the desired value
		if (rr[0] < rr[1])
			tau = rr[0];
		else
			tau = rr[1];
		// The pole is calculated for the saturated Q
		solve_quadratic_roots(2.0 * w - (1.0 + cos(tau)), -(4.0 * w * cos(tau) - 2.0 * (1.0 + cos(tau))), 2.0 * w - (1.0 + cos(tau)), &rr[0], &rr[1]);
		if (fabs(rr[0]) < 1.0)
			poles[i] = rr[0];
		else
			poles[i] = rr[1];
		poles[NFFT - i - 1] = poles[i];
	}
}
void LLraisedCosTblFloat(float *dst, unsigned int n)
{
	const double twopi_over_n = 6.283185307179586476925286766559 / n;
	for (unsigned int i = 0; i < n; ++i)
		dst[i] = (float)(0.5 * (1.0 - cos(twopi_over_n * (i + 0.5))));
}
void PhaseRadarInit(ConstantQTransform *cqt, unsigned int channels, unsigned int frameLen)
{
	memset(cqt, 0, sizeof(ConstantQTransform));
	int i;
	cqt->fftLen = frameLen;
	cqt->minus1fftLen = (cqt->fftLen - 1);
	cqt->NFFTDIV2 = (cqt->fftLen >> 1);
	cqt->halfLen = ((cqt->fftLen >> 1) + 1);
	// number of points of pre and post padding used to set initial conditions
	cqt->PREPAD = 10;
	cqt->POSPAD = 100;
	cqt->POS1 = (cqt->fftLen - cqt->PREPAD);
	cqt->POS2 = ((cqt->fftLen - (cqt->fftLen - cqt->PREPAD + 1)) + 1);
	cqt->PADDEDLEN = ((cqt->fftLen >> 1) + cqt->PREPAD + cqt->POSPAD);
	cqt->channels = channels;
	cqt->mCQTBuf = (float*)malloc(channels * cqt->fftLen * sizeof(float));
	cqt->mInput = (float*)malloc(channels * cqt->fftLen * sizeof(float));
	memset(cqt->mInput, 0, channels * cqt->fftLen * sizeof(float));
	cqt->delayLineInterleaved = (float*)malloc(channels * cqt->fftLen * sizeof(float));
	cqt->mBitRev = (unsigned int*)malloc(cqt->fftLen * sizeof(unsigned int));
	cqt->mSineTab = (float*)malloc(cqt->fftLen * sizeof(float));
	cqt->x_fft = (float*)malloc(cqt->fftLen * sizeof(float));
	cqt->fftBuf = (float*)malloc(cqt->fftLen * sizeof(float));
	cqt->window = (float*)malloc(cqt->fftLen * sizeof(float));
	cqt->real = (float*)malloc(cqt->fftLen * sizeof(float));
	cqt->imag = (float*)malloc(cqt->fftLen * sizeof(float));
	cqt->mag = (float*)malloc(channels * cqt->halfLen * sizeof(float));
	cqt->poles = (float*)malloc(cqt->PADDEDLEN * sizeof(float));
	cqt->gainPoles = (float*)malloc(cqt->halfLen * sizeof(float));
	cqt->bufferMovedRe = (float*)malloc(cqt->PADDEDLEN * sizeof(float));
	cqt->bufferMovedIm = (float*)malloc(cqt->PADDEDLEN * sizeof(float));
	cqt->x_zeroReal = (float*)malloc((cqt->PADDEDLEN - 1) * sizeof(float));
	cqt->x_zeroImag = (float*)malloc((cqt->PADDEDLEN - 1) * sizeof(float));
	cqt->filIIRBufReal = (float*)malloc(cqt->PADDEDLEN * sizeof(float));
	cqt->filIIRBufImag = (float*)malloc(cqt->PADDEDLEN * sizeof(float));
	cqt->q_fft_frameReal = (float*)malloc(cqt->PADDEDLEN * sizeof(float));
	cqt->q_fft_frameImag = (float*)malloc(cqt->PADDEDLEN * sizeof(float));
	LLraisedCosTblFloat(cqt->window, cqt->fftLen);
	LLbitReversalTbl(cqt->mBitRev, cqt->fftLen);
	LLsinHalfTblFloat(cqt->mSineTab, cqt->fftLen);
	cqt->mInputPos = 0;
	// Initialization
	double Q = 32.0;
	// frequency bins grid(linear in this case) - pre and pos padding is added
	double slope = M_PI / cqt->fftLen;
	double *thetas = (double*)malloc(cqt->NFFTDIV2 * sizeof(double));
	linspace(thetas, cqt->NFFTDIV2 + cqt->PREPAD + cqt->POSPAD, -cqt->PREPAD, cqt->NFFTDIV2 + cqt->POSPAD - 1, cqt->NFFTDIV2);
	for (i = 0; i < cqt->NFFTDIV2; i++)
		thetas[i] = fabs(slope * thetas[i]);
	thetas[cqt->PREPAD] = DBL_EPSILON; // zero digital frequency
	// poles of the IIR LTV Q FFT transform for the parameters above
	float *polesCal = (float*)malloc(cqt->fftLen * sizeof(float));
	calculate_poles(thetas, Q, cqt->fftLen, slope, polesCal);
	free(thetas);
	memcpy(cqt->poles, (polesCal + cqt->POS1), cqt->POS2 * sizeof(float));
	memcpy(cqt->poles + cqt->POS2, polesCal, (cqt->NFFTDIV2 + cqt->POSPAD) * sizeof(float));
	free(polesCal);
	for (i = 0; i < cqt->halfLen; i++)
		cqt->gainPoles[i] = (float)(1.0 / (2.0 / (1.0 - cqt->poles[i + cqt->PREPAD])));
}
void PhaseRadarProcessSamples(ConstantQTransform *cqt, float *x, unsigned int inSampleCount, char stftCQT, char draw)
{
	unsigned int ch, i, symIdx;
	for (ch = 0; ch < cqt->channels; ch++)
	{
		for (i = 0; i < inSampleCount; i++)
			cqt->mInput[ch * cqt->fftLen + ((cqt->mInputPos + i) & cqt->minus1fftLen)] = x[ch + i * cqt->channels];
	}
	cqt->mInputPos = (cqt->mInputPos + inSampleCount) & cqt->minus1fftLen;
	if (!draw)
		return;
	float scale = 1.0 / (float)(cqt->fftLen);
	if (stftCQT)
	{
		for (ch = 0; ch < cqt->channels; ch++)
		{
			// Currently support only one channel
			for (i = 0; i < cqt->fftLen; ++i)
				cqt->mCQTBuf[i] = cqt->mInput[ch * cqt->fftLen + ((i + cqt->mInputPos) & cqt->minus1fftLen)];
			// Copy overlapping buffer to FFT buffer with fftshift
			memcpy(cqt->x_fft, cqt->mCQTBuf + cqt->NFFTDIV2, (cqt->fftLen - cqt->NFFTDIV2 + 1) * sizeof(float));
			memcpy(cqt->x_fft + cqt->NFFTDIV2, cqt->mCQTBuf, cqt->NFFTDIV2 * sizeof(float));
			for (i = 0; i < cqt->fftLen; ++i)
				cqt->fftBuf[i] = cqt->x_fft[cqt->mBitRev[i]];
			// Do FFT
			LLdiscreteHartleyFloat(cqt->fftBuf, cqt->fftLen, cqt->mSineTab);
			// Spectral modify
			for (i = 1; i < cqt->fftLen; i++)
			{
				symIdx = cqt->fftLen - i;
				cqt->real[i] = cqt->fftBuf[i] + cqt->fftBuf[symIdx];
				cqt->imag[i] = cqt->fftBuf[i] - cqt->fftBuf[symIdx];
			}
			cqt->real[0] = cqt->fftBuf[0] * 2.0f;
			cqt->imag[0] = 0.0f;
			// Arrange spectrum
			memcpy(cqt->bufferMovedRe, cqt->real + cqt->POS1, cqt->POS2 * sizeof(float));
			memcpy(cqt->bufferMovedIm, cqt->imag + cqt->POS1, cqt->POS2 * sizeof(float));
			memcpy(cqt->bufferMovedRe + cqt->POS2, cqt->real, (cqt->NFFTDIV2 + cqt->POSPAD) * sizeof(float));
			memcpy(cqt->bufferMovedIm + cqt->POS2, cqt->imag, (cqt->NFFTDIV2 + cqt->POSPAD) * sizeof(float));
			for (i = 0; i < cqt->PADDEDLEN - 1; i++)
			{
				cqt->x_zeroReal[i] = cqt->bufferMovedRe[i + 1] + cqt->bufferMovedRe[i];
				cqt->x_zeroImag[i] = cqt->bufferMovedIm[i + 1] + cqt->bufferMovedIm[i];
			}
			cqt->filIIRBufReal[0] = cqt->bufferMovedRe[0];
			cqt->filIIRBufImag[0] = cqt->bufferMovedIm[0];
			for (i = 1; i < cqt->PADDEDLEN; i++)
			{
				cqt->filIIRBufReal[i] = cqt->x_zeroReal[i - 1] + cqt->poles[i] * cqt->filIIRBufReal[i - 1];
				cqt->filIIRBufImag[i] = cqt->x_zeroImag[i - 1] + cqt->poles[i] * cqt->filIIRBufImag[i - 1];
			}
			for (i = 0; i < cqt->PADDEDLEN - 1; i++)
			{
				cqt->filIIRBufReal[i] += cqt->filIIRBufReal[i + 1];
				cqt->filIIRBufImag[i] += cqt->filIIRBufImag[i + 1];
			}
			cqt->q_fft_frameReal[cqt->PADDEDLEN - 1] = cqt->filIIRBufReal[cqt->PADDEDLEN - 1];
			cqt->q_fft_frameImag[cqt->PADDEDLEN - 1] = cqt->filIIRBufImag[cqt->PADDEDLEN - 1];
			for (i = cqt->PADDEDLEN - 1; i-- > 0; )
			{
				cqt->q_fft_frameReal[i] = cqt->filIIRBufReal[i] + cqt->poles[i] * cqt->q_fft_frameReal[i + 1];
				cqt->q_fft_frameImag[i] = cqt->filIIRBufImag[i] + cqt->poles[i] * cqt->q_fft_frameImag[i + 1];
			}
			for (i = 0; i < cqt->halfLen; i++)
			{
				cqt->mag[ch * cqt->halfLen + i] = hypotf(cqt->q_fft_frameReal[i + cqt->PREPAD], cqt->q_fft_frameImag[i + cqt->PREPAD]) * scale * cqt->gainPoles[i];
			}
		}
	}
	else
	{
		for (ch = 0; ch < cqt->channels; ch++)
		{
			for (i = 0; i < cqt->fftLen; ++i)
				cqt->fftBuf[cqt->mBitRev[i]] = cqt->mInput[ch * cqt->fftLen + ((i + cqt->mInputPos) & cqt->minus1fftLen)] * cqt->window[i];
			// Do FFT
			LLdiscreteHartleyFloat(cqt->fftBuf, cqt->fftLen, cqt->mSineTab);
			cqt->mag[ch * cqt->halfLen + 0] = fabsf(cqt->fftBuf[0]) * scale * 2.0f;
			for (i = 1; i < cqt->halfLen; i++)
			{
				symIdx = cqt->fftLen - i;
				cqt->real[i] = cqt->fftBuf[i] + cqt->fftBuf[symIdx];
				cqt->imag[i] = cqt->fftBuf[i] - cqt->fftBuf[symIdx];
				cqt->mag[ch * cqt->halfLen + i] = hypotf(cqt->real[i], cqt->imag[i]) * scale * 2.0f;
			}
		}
	}
}
void PhaseRadarFree(ConstantQTransform *cqt)
{
	free(cqt->mInput);
	free(cqt->mCQTBuf);
	free(cqt->delayLineInterleaved);
	free(cqt->mBitRev);
	free(cqt->mSineTab);
	free(cqt->x_fft);
	free(cqt->fftBuf);
	free(cqt->window);
	free(cqt->real);
	free(cqt->imag);
	free(cqt->mag);
	free(cqt->poles);
	free(cqt->gainPoles);
	free(cqt->bufferMovedRe);
	free(cqt->bufferMovedIm);
	free(cqt->x_zeroReal);
	free(cqt->x_zeroImag);
	free(cqt->filIIRBufReal);
	free(cqt->filIIRBufImag);
	free(cqt->q_fft_frameReal);
	free(cqt->q_fft_frameImag);
}