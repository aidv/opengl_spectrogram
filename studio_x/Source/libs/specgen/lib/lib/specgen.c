/*
**	Generate a spectrogram as a PNG file from a given sound file.
*/

/*
**	Todo:
**      - Make magnitude to colour mapper allow abitrary scaling (ie cmdline
**        arg).
**      - Better cmdline arg parsing and flexibility.
*/

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <limits.h>
#include <assert.h>
//#include <vld.h>
//  Windows
#ifdef _WIN32
#include <Windows.h>
////////////////////////////////////////////////////////////////////
// Performance timer
double get_wall_time()
{
	LARGE_INTEGER time, freq;
	if (!QueryPerformanceFrequency(&freq))
		return 0;
	if (!QueryPerformanceCounter(&time))
		return 0;
	return (double)time.QuadPart / freq.QuadPart;
}
double get_cpu_time()
{
	FILETIME a, b, c, d;
	if (GetProcessTimes(GetCurrentProcess(), &a, &b, &c, &d) != 0)
		return (double)(d.dwLowDateTime | ((unsigned long long)d.dwHighDateTime << 32)) * 0.0000001;
	else
		return 0;
}
#else
#include <time.h>
#include <sys/time.h>
double get_wall_time()
{
	struct timeval time;
	if (gettimeofday(&time, NULL))
		return 0;
	return (double)time.tv_sec + (double)time.tv_usec * .000001;
}
double get_cpu_time()
{
	return (double)clock() / CLOCKS_PER_SEC;
}
#endif

#define STB_SPRINTF_IMPLEMENTATION
#include "../stb_sprintf.h"
char *createStringVAList(const char *fmt, ...)
{
	va_list args;
	va_start(args, fmt);
	size_t bufsz = stbsp_vsnprintf(NULL, 0, fmt, args);
	char *filenameNew1 = (char *)malloc(bufsz + 1);
	stbsp_vsnprintf(filenameNew1, bufsz + 1, fmt, args);
	va_end(args);
	return filenameNew1;
}
void bgra2rbga(int *buffer, int width, int height)
{
	for (int i = 0; i < width * height; i++) {
		buffer[i] = (buffer[i] & 0xFF000000) |         // ______AA
			((buffer[i] & 0x00FF0000) >> 16) | // RR______
			(buffer[i] & 0x0000FF00) |         // __GG____
			((buffer[i] & 0x000000FF) << 16);  // ____BB__
	}
}
void rbga2bgra(int *buffer, int width, int height)
{
	for (int i = 0; i < width * height; i++) {
		buffer[i] = (buffer[i] & 0xFF000000) |         // ______AA
			((buffer[i] & 0x00FF0000) >> 16) | // BB______
			(buffer[i] & 0x0000FF00) |         // __GG____
			((buffer[i] & 0x000000FF) << 16);  // ____RR__
	}
}
#define ARRAY_LEN(x)		((int) (sizeof (x) / sizeof (x [0])))


#ifndef M_PI
#define M_PI 3.141592653589793f
#endif
#define MAX(x, y)		((x) > (y) ? (x) : (y))
#include <stdio.h>
#include <stdint.h>

typedef struct
{
	int left, top, width, height;
} rect;
/* The greatest number of linear ticks seems to occurs from 0-14000 (15 ticks).
** The greatest number of log ticks occurs 10-99999 or 11-100000 (35 ticks).
** Search for "worst case" for the commentary below that says why it is 35.
*/
#define DISTMAX 64
typedef struct
{
	double value[64];  /* 35 or more */
	double distance[64];
	/* The digit that changes from label to label.
	** This ensures that a range from 999 to 1001 prints 999.5 and 1000.5
	** instead of 999 1000 1000 1000 1001.
	*/
	int decimal_places_to_print;
} TICKS;

/* Decide where to put ticks and numbers on an axis.
**
** Graph-labelling convention is that the least significant digit that changes
** from one label to the next should change by 1, 2 or 5, so we step by the
** largest suitable value of 10^n * {1, 2 or 5} that gives us the required
** number of divisions / numeric labels.
*/

/* The old code used to make 6 to 14 divisions and number every other tick.
** What we now mean by "division" is one of teh gaps between numbered segments
** so we ask for a minimum of 3 to give the same effect as the old minimum of
** 6 half-divisions.
** This results in the same axis labelling for all maximum values
** from 0 to 12000 in steps of 1000 and gives sensible results from 13000 on,
** to a maximum of 7 divisions and 8 labels from 0 to 14000.
**/
#define TARGET_DIVISIONS 14

/* A tolerance to use in floating point < > <= >= comparisons so that
** imprecision doesn't prevent us from printing an initial or final label
** if it should fall exactly on min or max but doesn't due to FP problems.
** For example, for 0-24000, the calculations might give 23999.9999999999.
*/
#define DELTA (1e-10)

static int calculate_log_ticks(double min_freq, double max_freq, double distance, TICKS *ticks, int tk);

/* log_scale is pseudo-boolean:
** 0 means use a linear scale,
** 1 means use a log scale and
** 2 is an internal value used when calling back from calculate_log_ticks() to
**   label the range with linear numbering but logarithmic spacing.
*/
double hz2mel(double hz)
{
	return 2595 * log10(1.0 + (hz / 700));
}
double mel2hz(double mel)
{
	return 700 * (pow(10, (mel / 2595.0)) - 1.0);
}
#include <float.h>
double wrightOmegaq(double X)
{
	double EXP1 = exp(1);
	double EXP2 = exp(2);
	double LN2 = log(2);
	double OMEGA = 0.5671432904097838;
	double ONE_THIRD = 1.0 / 3.0;
	double W;
	// Special values
	if (X > pow(2.0, 59.0))
		W = X; // W self-saturates: X > 2^59 (abs(Y) > 2^54 too)
	else if (X == 0.0)
		W = OMEGA; // Omega constant
	else if (X == 1.0)
		W = 1;
	else if (X == 1.0 + EXP1)
		W = EXP1;
	else
	{
		if (X < log(DBL_EPSILON * DBL_MIN) - LN2)
			W = 0.0; // Z -> -Inf
		else
		{
			// W used in order retain datatype
			if (X <= -2.0)
			{
				// Region 3: series about -Inf
				double x = exp(X);
				W = x * (1.0 - x * (1.0 - x * (36.0 - x * (64.0 - 125.0 * x)) / 24.0));
				// Series is exact, X < -exp(2)
				if (X < -EXP2)
					return W;
			}
			else if (X > M_PI + 1)
			{
				// Region 7: log series about Z = Inf
				double x = log(X);
				double lzi = x / X;
				W = X - x + lzi * (1.0 + lzi * (0.5 * x - 1.0 + lzi * ((ONE_THIRD * x - 1.5) * x + 1)));
			}
			else
			{
				// Region 4: series about Z = 1
				double x = X - 1.0;
				W = 1.0 + x * (1.0 / 2.0 + x * (1.0 / 16.0 - x * (1.0 / 192.0 + x * (1.0 / 3072.0 - (13.0 / 61440.0) * x))));
			}
			// Residual
			double r = X - (W + log(W));
			if (fabs(r) > DBL_EPSILON)
			{
				// FSC-type iteration, N = 3, (Fritsch, Shafer, & Crowley, 1973)
				double w1 = 1.0 + W;
				double w2 = w1 + 2 * ONE_THIRD * r;
				W = W * (1.0 + r * (w1 * w2 - 0.5 * r) / (w1 * (w1 * w2 - r)));
				// Test residual
				r = X - (W + log(W));
				// Second iterative improvement via FSC method, if needed
				if (fabs(r) > DBL_EPSILON)
				{
					w1 = 1.0 + W;
					w2 = w1 + 2 * ONE_THIRD * r;
					W = W * (1.0 + r * (w1 * w2 - 0.5 * r) / (w1 * (w1 * w2 - r)));
				}
			}
		}
	}
	return W;
}
double linLgTransform(double x, double a)
{
	return x * a + log(x) * (1.0 - a);
}
double linExpTransform(double x, double a)
{
	return -(wrightOmegaq(-log(-(a - 1.0) / a) - x / (a - 1)) * (a - 1.0)) / a;
}
const double linLogRatio = 0.0001; // 0.00001
double lgTransform(double x)
{
	//return log(x);
	return hz2mel(x);
	//return linLgTransform(x, linLogRatio);
}
double expTransform(double x)
{
	//return exp(x);
	return mel2hz(x);
	//return linExpTransform(x, linLogRatio);
}
static int calculate_ticks(double min_freq, double max_freq, double distance, int log_scale, TICKS *ticks, int tk)
{
	double step;
	double range = max_freq - min_freq;
	int k;
	double value;
	if (log_scale == 1)
		return calculate_log_ticks(min_freq, max_freq, distance, ticks, tk);
	step = pow(10.0, floor(log10(max_freq)));
	do
	{
		if (range / (step * 5) >= TARGET_DIVISIONS)
		{
			step *= 5;
			break;
		};
		if (range / (step * 2) >= TARGET_DIVISIONS)
		{
			step *= 2;
			break;
		};
		if (range / step >= TARGET_DIVISIONS)
			break;
		step /= 10;
	} while (1);
	ticks->decimal_places_to_print = lrint(-floor(log10(step)));
	if (ticks->decimal_places_to_print < 0)
		ticks->decimal_places_to_print = 0;
	k = 0;
	value = ceil(min_freq / step) * step;

#define add_tick(val) do \
	{	if (val >= min_freq - DELTA && val < max_freq + DELTA) \
		{	ticks->value [k] = val ; \
			ticks->distance [k] = distance * \
				(log_scale == 2 \
					? /*log*/ (lgTransform(val) - lgTransform(min_freq)) / (lgTransform(max_freq) - lgTransform(min_freq)) \
					: /*lin*/ (val - min_freq) / range) ; \
			k++ ; \
			} ; \
		} while (0)

	add_tick(value - step / 2);

	while (value <= max_freq + DELTA)
	{
		if (k >= (DISTMAX - 1))
			break;
		add_tick(value);
		add_tick(value + step / 2);

		value += step;
	}
	return k;
}
static int add_log_ticks(double min_freq, double max_freq, double distance, TICKS *ticks, int k, double start_value)
{
	double value;

	for (value = start_value; value <= max_freq + DELTA; value *= 10.0)
	{
		if (value < min_freq - DELTA) continue;
		ticks->value[k] = value;
		ticks->distance[k] = distance * (lgTransform(value) - lgTransform(min_freq)) / (lgTransform(max_freq) - lgTransform(min_freq));
		k++;
	}
	return k;
}
static int calculate_log_ticks(double min_freq, double max_freq, double distance, TICKS *ticks, int tk)
{
	int k = 0;
	double underpinning;
	if (max_freq / min_freq < 10.0)
		return calculate_ticks(min_freq, max_freq, distance, 2, ticks, tk);
	if (max_freq / min_freq > 1000000)
	{
		printf("Error: Frequency range is too large for logarithmic scale.\n");
		exit(1);
	};
	underpinning = pow(10.0, floor(log10(min_freq)));
	k = add_log_ticks(min_freq, max_freq, distance, ticks, k, underpinning);
	if (k >= TARGET_DIVISIONS + 1)
	{
		k = add_log_ticks(min_freq, max_freq, distance, ticks, k, underpinning * 2.0);
		k = add_log_ticks(min_freq, max_freq, distance, ticks, k, underpinning * 5.0);
	}
	else
	{
		int i;
		for (i = 2; i <= tk; i++)
			k = add_log_ticks(min_freq, max_freq, distance, ticks, k, underpinning * (1.0 * i));
	}
	return k;
}
static int count_ticks(double min_freq, double max_freq, double distance, int log_scale, int tk);
static int clog_ticks(double min_freq, double max_freq, double distance, int k, double start_value)
{
	double value;

	for (value = start_value; value <= max_freq + DELTA; value *= 10.0)
	{
		if (value < min_freq - DELTA) continue;
		k++;
	}
	return k;
}
static int count_log_ticks(double min_freq, double max_freq, double distance, int tk)
{
	int k = 0;
	double underpinning;
	if (max_freq / min_freq < 10.0)
		return count_ticks(min_freq, max_freq, distance, 2, tk);
	if (max_freq / min_freq > 1000000)
	{
		printf("Error: Frequency range is too large for logarithmic scale.\n");
		exit(1);
	};
	underpinning = pow(10.0, floor(log10(min_freq)));
	k = clog_ticks(min_freq, max_freq, distance, k, underpinning);
	if (k >= TARGET_DIVISIONS + 1)
	{
		k = clog_ticks(min_freq, max_freq, distance, k, underpinning * 2.0);
		k = clog_ticks(min_freq, max_freq, distance, k, underpinning * 5.0);
	}
	else
	{
		int i;
		for (i = 2; i <= tk; i++)
			k = clog_ticks(min_freq, max_freq, distance, k, underpinning * (1.0 * i));
	}
	return k;
}
static int count_ticks(double min_freq, double max_freq, double distance, int log_scale, int tk)
{
	double step;
	double range = max_freq - min_freq;
	int k;
	double value;
	if (log_scale == 1)
		return count_log_ticks(min_freq, max_freq, distance, tk);
	step = pow(10.0, floor(log10(max_freq)));
	do
	{
		if (range / (step * 5) >= TARGET_DIVISIONS)
		{
			step *= 5;
			break;
		};
		if (range / (step * 2) >= TARGET_DIVISIONS)
		{
			step *= 2;
			break;
		};
		if (range / step >= TARGET_DIVISIONS)
			break;
		step /= 10;
	} while (1);
	k = 0;
	value = ceil(min_freq / step) * step;

#define atick(val) do \
	{	if (val >= min_freq - DELTA && val < max_freq + DELTA) \
		{\
			k++ ; \
			} ; \
		} while (0)

	atick(value - step / 2);

	while (value <= max_freq + DELTA)
	{
		if (k >= (DISTMAX - 1))
			break;
		atick(value);
		atick(value + step / 2);

		value += step;
	}
	return k;
}
static int render_spect_border(double height, double min_freq, double max_freq, char log_freq, double ticksCoordinate[64], double ticksLabel[64])
{
	TICKS ticks;
	ticks.decimal_places_to_print = 0;
	memset(ticksCoordinate, 0, 64 * sizeof(double));
	memset(ticksLabel, 0, 64 * sizeof(double));
	int tk = 50;
	int tick_count = count_ticks(min_freq, max_freq, height, log_freq, tk);
	while (tick_count > 64)
	{
		tk--;
		tick_count = count_ticks(min_freq, max_freq, height, log_freq, tk);
	}
	tick_count = calculate_ticks(min_freq, max_freq, height, log_freq, &ticks, tk);
	for (int k = 0; k < tick_count; k++)
	{
		ticksCoordinate[k] = height - ticks.distance[k];
		ticksLabel[k] = ticks.value[k];
	}
	return tick_count;
}
typedef struct
{
	// Image
	float *image;
	unsigned int halfLen, frameCount, allocatedFrame;
} imageBuf;
void initImageBuf(imageBuf *img, unsigned int halfLen)
{
	img->halfLen = halfLen;
	img->frameCount = 0;
	img->allocatedFrame = 128;
	img->image = (float *)malloc(halfLen * img->allocatedFrame * sizeof(float));
	memset(img->image, 0, halfLen * img->allocatedFrame * sizeof(float));
}
void freeImageBuf(imageBuf *img)
{
	free(img->image);
}
void writeImageBufIntoMemory(imageBuf *img, float *mag)
{
	if (img->frameCount >= img->allocatedFrame)
	{
		img->allocatedFrame += 128;
		img->image = (float *)realloc(img->image, img->halfLen * img->allocatedFrame * sizeof(float));
	}
	memcpy(img->image + img->frameCount * img->halfLen, mag, img->halfLen * sizeof(float));
	img->frameCount++;
}
unsigned long upper_power_of_two(unsigned long v)
{
	v--;
	v |= v >> 1;
	v |= v >> 2;
	v |= v >> 4;
	v |= v >> 8;
	v |= v >> 16;
	v++;
	return v;
}
typedef struct
{
	int width, height, channels, samplerate;
	int log_freq;
	float *timeAxis;
	double min_time, max_time;
	double min_freq, max_freq;
	unsigned int frameShift;
	imageBuf *img;
	double spec_floor_db;
} RENDER;
/* Helper function:
** Map the index for an output pixel in a column to an index into the
** FFT result representing the same frequency.
** magindex is from 0 to maglen-1, representing min_freq to max_freq Hz.
** Return values from are from 0 to speclen representing frequencies from
** 0 to the Nyquist frequency.
** The result is a floating point number as it may fall between elements,
** allowing the caller to interpolate onto the input array.
*/
static double magindex_to_specindex(double speclenMapping, double recipmaglen, int magindex, double min_freq, double max_freq, int samplerate, char log_freq)
{
	double freq; /* The frequency that this output value represents */

	if (!log_freq)
		freq = min_freq + (max_freq - min_freq) * (double)magindex * recipmaglen;
	else
	{
		// Log scale
		double low2 = lgTransform(min_freq);
		double high2 = lgTransform(max_freq);
		double lgVect = low2 + (high2 - low2) * (double)magindex * recipmaglen;
		freq = expTransform(lgVect);
		// Mel scale
		/*double low = hz2mel(min_freq);
		double high = hz2mel(max_freq);
		double melVect = low + (high - low) * (double)magindex * recipmaglen;
		freq = mel2hz(melVect);*/
	}
	return freq * speclenMapping;
}
// Map values from the spectrogram onto an array of magnitudes, the values
// for display. Reads spec[0..speclen], writes mag[0..maglen-1]
static inline void interp_spec(float *mag, int maglen, const float *spec, int speclen, const RENDER *render, int samplerate)
{
	int k;

	/* Map each output coordinate to where it depends on in the input array.
	** If there are more input values than output values, we need to average
	** a range of inputs.
	** If there are more output values than input values we do linear
	** interpolation between the two inputs values that a reverse-mapped
	** output value's coordinate falls between.
	**
	** spec points to an array with elements [0..speclen] inclusive
	** representing frequencies from 0 to samplerate/2 Hz. Map these to the
	** scale values min_freq to max_freq so that the bottom and top pixels
	** in the output represent the energy in the sound at min_ and max_freq Hz.
	*/
	double speclenMapping = speclen / (samplerate / 2.0);
	double recipmaglen = 1.0 / maglen;
	for (k = 0; k < maglen; k++)
	{	/* Average the pixels in the range it comes from */
		double this = magindex_to_specindex(speclenMapping, recipmaglen, k, render->min_freq, render->max_freq, samplerate, render->log_freq);
		double next = magindex_to_specindex(speclenMapping, recipmaglen, k + 1, render->min_freq, render->max_freq, samplerate, render->log_freq);

		/* Range check: can happen if --max-freq > samplerate / 2 */
		if (this > speclen)
		{
			mag[k] = 0.0;
			return;
		}

		if (next > this + 1)
		{	/* The output indices are more sparse than the input indices
			** so average the range of input indices that map to this output,
			** making sure not to exceed the input array (0..speclen inclusive)
			*/
			/* Take a proportional part of the first sample */
			double count = 1.0 - (this - floor(this));
			double sum = spec[(int)this] * count;

			while ((this += 1.0) < next && (int)this <= speclen)
			{
				sum += spec[(int)this];
				count += 1.0;
			}
			/* and part of the last one */
			if ((int)next <= speclen)
			{
				sum += spec[(int)next] * (next - floor(next));
				count += next - floor(next);
			}

			mag[k] = sum / count;
		}
		else
			/* The output indices are more densely packed than the input indices
			** so interpolate between input values to generate more output values.
			*/
			/* Take a weighted average of the nearest values */
			mag[k] = spec[(int)this] * (1.0 - (this - floor(this))) + spec[(int)this + 1] * (this - floor(this));
	}
}
#define DR_WAV_IMPLEMENTATION
#include "../dr_wav.h"
void writeWavTest(char *filenameNew, float *sndBuf, unsigned int ch, unsigned int fs, unsigned long long totalPCMFrameCount)
{
	drwav pWav;
	drwav_data_format format;
	format.container = drwav_container_riff;
	format.format = DR_WAVE_FORMAT_IEEE_FLOAT;
	format.channels = ch;
	format.sampleRate = fs;
	format.bitsPerSample = 32;
	unsigned int fail = drwav_init_file_write(&pWav, filenameNew, &format, 0);
	drwav_uint64 framesWritten = drwav_write_pcm_frames(&pWav, totalPCMFrameCount, sndBuf);
	drwav_uninit(&pWav);
}
void interleaveChannel(float *chan_buffers, unsigned int num_channels, float *buffer, size_t num_frames)
{
	size_t i, samples = num_frames * num_channels;
	for (i = 0; i < samples; i++)
		buffer[i] = chan_buffers[num_frames * (i % num_channels) + (i / num_channels)];
}
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../stb_image_write.h"
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "../stb_image_resize.h"
#include "iircqt/ConstantQTransform.h"
static void render_to_surface(float *mag_spec, const RENDER *render, double *data, float *resized, int stride, int width, int height, int ch, unsigned int halfLen, unsigned int paddednewWidth, double g, int nThreads)
{
	int w;
	if (width < render->img[ch].frameCount)
	{
		int ret = stbir_resize_float_preset4Spectrogram(render->img[ch].image, halfLen, render->img[ch].frameCount, halfLen * sizeof(float), resized, halfLen, paddednewWidth, halfLen * sizeof(float), 1, g, nThreads);
#pragma omp parallel for default(none) num_threads(nThreads) private(w)
		for (w = 0; w < width; w++)
		{
			interp_spec(&mag_spec[w * height], height, &resized[w * halfLen], halfLen, render, render->samplerate);
			for (int h = 0; h < height; h++)
			{
				int x, y;
				if (mag_spec[w * height + h] < DBL_EPSILON)
					mag_spec[w * height + h] = DBL_EPSILON;
				float val = 20.0f * log10f(mag_spec[w * height + h]);
				y = height - 1 - h;
				data[y * width + w] = val;
			}
		}
	}
	else
	{
#pragma omp parallel for default(none) num_threads(nThreads) private(w)
		for (w = 0; w < render->img[ch].frameCount; w++)
		{
			interp_spec(&resized[w * height], height, &render->img[ch].image[w * halfLen], halfLen, render, render->samplerate);
		}
		int ret = stbir_resize_float_preset4Spectrogram(resized, height, render->img[ch].frameCount, height * sizeof(float), mag_spec, height, paddednewWidth, height * sizeof(float), 1, g, nThreads);
#pragma omp parallel for default(none) num_threads(nThreads) private(w)
		for (w = 0; w < width; w++)
		{
			for (int h = 0; h < height; h++)
			{
				int x, y;
				if (mag_spec[w * height + h] < DBL_EPSILON)
					mag_spec[w * height + h] = DBL_EPSILON;
				float val = 20.0f * log10f(mag_spec[w * height + h]);
				y = height - 1 - h;
				data[y * width + w] = val;
			}
		}
	}
}
void *getSpectrogram(char *filename, int fftLen, int userStart, int userEnd, double freqStart, double freqEnd,
	int nThreads, char dbg, int imgWidth, int imgHeight, int channels, int rulerOnly)
{
	RENDER render = { 0 };
	render.log_freq = 1; // Log plot
	render.min_freq = freqStart;
	render.max_freq = freqEnd;
	render.width = imgWidth;
	render.height = imgHeight;
	unsigned int i;
	drwav pWav;
	// Now load from dr_wav
	if (!drwav_init_file(&pWav, filename, 0))
		return 0;
	render.samplerate = pWav.sampleRate;
	double lowerBound = ((double)render.samplerate / (double)fftLen) * 0.5;
	if (render.log_freq)
	{
		if (render.min_freq < lowerBound)
			render.min_freq = lowerBound;
		if (render.max_freq > render.samplerate / 2.0)
			render.max_freq = render.samplerate / 2.0;
	}
	size_t whProd = render.height * render.width;
	if (channels != pWav.channels)
		return 0;
	char *memBuf = (char *)malloc((1 + 64 * 2 + pWav.channels * whProd) * sizeof(double));
	if (render.width < 1)
	{
		printf("Error : 'width' parameter must be >= %d\n", 1);
		return 0;
	}
	if (render.height < 1)
	{
		printf("Error : 'height' parameter must be >= %d\n", 1);
		return 0;
	}
	double *ticksCoordinate = ((double *)memBuf) + 1;
	double *ticksLabel = ticksCoordinate + 64;
	*((double *)memBuf) = render_spect_border(render.height, render.min_freq, render.max_freq, render.log_freq, ticksCoordinate, ticksLabel);
	if (rulerOnly)
		return memBuf;
	unsigned int frameCentre = fftLen >> 1;
	if (userStart > userEnd)
		userStart = userEnd;
	if (userStart > pWav.totalPCMFrameCount)
		userStart = 0;
	drwav_uint64 end = min(userEnd, pWav.totalPCMFrameCount);
	drwav_uint64 duration = end - userStart;
	double pixelpersmps = (double)render.width / (double)duration;
	const double pixelsPerSecond = pixelpersmps * render.samplerate;
	const double tstep = 1.0 / pixelsPerSecond;
	const double samplesPerPixel = render.samplerate * tstep;
	unsigned int frameShift = (unsigned int)ceil(samplesPerPixel);
	if (frameShift > fftLen / 2)
		frameShift = fftLen / 2;
	if (frameShift > duration)
		frameShift = upper_power_of_two(duration);
	render.min_time = userStart;
	render.max_time = end;
	render.frameShift = frameShift;
	int stride = 4 * render.width;
	double *spectro = ticksLabel + 64;
	drwav_uint64 numReads;
	unsigned int seek;
	unsigned int runBlankSamples;
	float *pPCMFrames_drwav = (float *)malloc((size_t)(frameShift * pWav.channels * sizeof(float)));
	if (pPCMFrames_drwav == NULL)
	{
		printf("  [dr_wav] Out of memory");
		return 0;
	}
	render.channels = pWav.channels;
	ConstantQTransform transformer;
	PhaseRadarInit(&transformer, pWav.channels, fftLen);
	float *wavDebuggingBuffer = (float *)malloc(pWav.channels * fftLen * sizeof(float));
	render.img = (imageBuf *)malloc(pWav.channels * sizeof(imageBuf));
	for (unsigned int ch = 0; ch < transformer.channels; ch++)
		initImageBuf(&render.img[ch], transformer.halfLen);
	//for (i = 0; i < transformer.halfLen; i++)
	//	printf("%1.8f,", i / (float)transformer.halfLen * (pWav.sampleRate / 2.0f));
	if (userStart >= fftLen)
	{
		seek = userStart - fftLen;
		runBlankSamples = 0;
	}
	else
	{
		seek = 0;
		runBlankSamples = fftLen - userStart;
	}
	drwav_seek_to_pcm_frame(&pWav, seek);
	unsigned cnt = seek;
	const char useCQT = 0;
	// If starting position is smaller than frame size
	for (i = 0; i < frameShift * pWav.channels; i++)
		pPCMFrames_drwav[i] = 0.0f;
	if (userStart < fftLen)
	{
		if (dbg)
			printf("Seeking back samples: ");
		drwav_uint64 pcmFrameCount_drwav = 0;
		for (drwav_uint64 iPCMFrame = 0; iPCMFrame < runBlankSamples; iPCMFrame += frameShift)
		{
			if ((runBlankSamples - iPCMFrame) < frameShift)
				pcmFrameCount_drwav = drwav_read_pcm_frames_f32(&pWav, frameShift - (runBlankSamples - iPCMFrame), pPCMFrames_drwav + (runBlankSamples - iPCMFrame) * pWav.channels);
			PhaseRadarProcessSamples(&transformer, pPCMFrames_drwav, frameShift, useCQT, 0);
			if (dbg)
				printf("-%d, ", runBlankSamples - iPCMFrame);
		}
		cnt = pcmFrameCount_drwav;
		numReads = ceilf((float)(end - cnt) / (float)frameShift) * frameShift + frameCentre;
		if (dbg)
			printf("\n");
	}
	else
		numReads = (end - userStart) + fftLen + frameCentre;
	unsigned int estimatedFrames = ceil((double)numReads / frameShift);
	render.timeAxis = (float *)malloc(estimatedFrames * sizeof(float));
	unsigned int correspondingPosition = userStart;
	unsigned int idx = 0;
	char draw;
	for (drwav_uint64 iPCMFrame = 0; iPCMFrame < numReads; iPCMFrame += frameShift)
	{
		drwav_uint64 pcmFrameCount_drwav = drwav_read_pcm_frames_f32(&pWav, frameShift, pPCMFrames_drwav);
		if (pcmFrameCount_drwav != frameShift) // The end of file
		{
			for (i = 0; i < (frameShift - pcmFrameCount_drwav) * pWav.channels; i++)
				pPCMFrames_drwav[pcmFrameCount_drwav * pWav.channels + i] = 0.0f;
		}
		if (cnt < (userStart + frameCentre - frameShift))
		{
			draw = 0;
			cnt += frameShift;
		}
		else if (cnt == (userStart + frameCentre - frameShift))
		{
			cnt += frameShift;
			draw = 1;
			if (dbg)
				printf("Draw now! Actual position(Corresponding to frame centre): %d\n", correspondingPosition);
			//printf("Reading waveform location: %d\n", seek + iPCMFrame);
		}
		else
		{
			correspondingPosition += frameShift;
			draw = 1;
			if (dbg)
				printf("Draw now! Actual position(Corresponding to frame centre): %d\n", correspondingPosition);
			//printf("Reading waveform location: %d\n", seek + iPCMFrame);
		}
		if (iPCMFrame >= (numReads - frameShift * 2))
		{
			if (iPCMFrame >= (numReads - frameShift * 1))
			{
				draw = 3;
				float normalizeDistBetweenFrame1ToUserEnd = (float)(end - (correspondingPosition - frameShift)) / (float)frameShift;
				float normalizeDistBetweenUserEndToFrame2 = 1.0f - normalizeDistBetweenFrame1ToUserEnd;
				if (dbg)
					printf("Last -0 frame\n");
			}
			else
			{
				draw = 2;
				if (dbg)
					printf("Last -1 frame\n");
			}
		}
		PhaseRadarProcessSamples(&transformer, pPCMFrames_drwav, frameShift, useCQT, draw);
		if (draw)
		{
			if (dbg == 2)
			{
				// copy to temporary buffer and FHT
				for (unsigned int ch = 0; ch < transformer.channels; ch++)
					for (i = 0; i < fftLen; ++i)
						wavDebuggingBuffer[ch * fftLen + i] = transformer.mInput[ch * fftLen + ((i + transformer.mInputPos) & transformer.minus1fftLen)];
				char *filename = createStringVAList("out%d.wav", idx++);
				interleaveChannel(wavDebuggingBuffer, transformer.channels, transformer.delayLineInterleaved, fftLen);
				writeWavTest(filename, transformer.delayLineInterleaved, transformer.channels, pWav.sampleRate, fftLen);
				free(filename);
			}
			render.timeAxis[render.img[0].frameCount] = correspondingPosition;
			for (unsigned int ch = 0; ch < transformer.channels; ch++)
				writeImageBufIntoMemory(&render.img[ch], transformer.mag + transformer.halfLen * ch);
			if (correspondingPosition > end)
				break;
		}
		//printf("");
	}
	drwav_uninit(&pWav);
	free(pPCMFrames_drwav);
	PhaseRadarFree(&transformer);
	free(wavDebuggingBuffer);
	// Do this sanity check here, as soon as max_freq has its default value
	if (render.min_freq >= render.max_freq)
	{
		printf("Error : freqStart (%g) must be less than max_freq (%g)\n", render.min_freq, render.max_freq);
		return 0;
	}
	unsigned int halfLen = render.img[0].halfLen;
	double a = (render.max_time - render.timeAxis[render.img[0].frameCount - 2]) / (double)render.frameShift;
	double b = (render.timeAxis[render.img[0].frameCount - 1] - render.max_time) / (double)render.frameShift;
	double act = render.img[0].frameCount + a;
	double rat1 = (double)render.width / (double)render.img[0].frameCount;
	double rat2 = (double)render.width - rat1;
	double c = rat2 * b + render.width * a;
	double newWidth = render.width * render.width / c;
	unsigned int paddednewWidth = floor(newWidth);
	double k = paddednewWidth / (double)render.img[0].frameCount;
	double d = paddednewWidth - k;
	double rightCorner = d * b + paddednewWidth * a;
	double g = rightCorner / render.width;

	double virtualLen = paddednewWidth / g;
	double k2 = virtualLen / (double)render.img[0].frameCount;
	double d2 = virtualLen - k2;
	double recoveredCorner = d2 * b + virtualLen * a;
	float *resized, *mag_spec;
	if (render.width < render.img[0].frameCount)
	{
		resized = (float *)malloc(paddednewWidth * halfLen * sizeof(float));
		mag_spec = malloc(render.width * render.height * sizeof(float));
		memset(mag_spec, 0, render.width * render.height * sizeof(float));
	}
	else
	{
		resized = (float *)malloc(render.img[0].frameCount * render.height * sizeof(float));
		memset(resized, 0, render.img[0].frameCount * render.height * sizeof(float));
		mag_spec = malloc(paddednewWidth * render.height * sizeof(float));
	}
	for (unsigned int ch = 0; ch < transformer.channels; ch++)
	{
		render_to_surface(mag_spec, &render, spectro + ch * whProd, resized, stride, render.width, render.height, ch, halfLen, paddednewWidth, g, nThreads);
		freeImageBuf(&render.img[ch]);
	}
	free(resized);
	free(mag_spec);

	free(render.timeAxis);
	free(render.img);
	return memBuf;
}
__declspec(dllexport) void freeInternal(void *ptr)
{
	free(ptr);
}
__declspec(dllexport) double *spectrogramFFIConfirmMemorySize(int imgWidth, int imgHeight, int channels)
{
	/*fprintf(stderr, "imgWidth = %d\n", imgWidth);
	fprintf(stderr, "imgHeight = %d\n", imgHeight);
	fprintf(stderr, "Audio channels = %d\n", channels);*/
	size_t whProd = imgHeight * imgWidth;
	double *scalar = (double *)malloc(sizeof(double));
	*scalar = 1 + 64 * 2 + channels * whProd;
	return scalar;
}
__declspec(dllexport) double *spectrogramFFIReturnDouble(char *filename, int fftLen, int userStart, int userEnd, double freqStart, double freqEnd,
	int nThreads, int dbg, int imgWidth, int imgHeight, int channels, int rulerOnly)
{
	/*fprintf(stderr, "filename = %s\n", filename);
	fprintf(stderr, "fftLen = %d\n", fftLen);
	fprintf(stderr, "userStart = %d\n", userStart);
	fprintf(stderr, "userEnd = %d\n", userEnd);
	fprintf(stderr, "freqStart = %1.14lf\n", freqStart);
	fprintf(stderr, "freqEnd = %1.14lf\n", freqEnd);
	fprintf(stderr, "nThreads = %d\n", nThreads);
	fprintf(stderr, "dbg = %d\n", dbg);
	fprintf(stderr, "imgWidth = %d\n", imgWidth);
	fprintf(stderr, "imgHeight = %d\n", imgHeight);
	fprintf(stderr, "Audio channels = %d\n", channels);*/
	void *buf = getSpectrogram(filename, fftLen, userStart, userEnd, freqStart, freqEnd, nThreads, dbg, 
		imgWidth, imgHeight, channels, rulerOnly);
	return (double*)buf;
}
__declspec(dllexport) double *wavFileInfo(char *filename)
{
	drwav pWav = { 0 };
	// Now load from dr_wav
	if (!drwav_init_file(&pWav, filename, 0))
		return 0;
	double *buf = (double *)malloc(4 * sizeof(double));
	buf[0] = pWav.totalPCMFrameCount;
	buf[1] = pWav.channels;
	buf[2] = pWav.sampleRate;
	buf[3] = pWav.bitsPerSample;
	drwav_uninit(&pWav);
	return buf;
}
__declspec(dllexport) double *waveformFFIConfirmMemorySize(int userStart, int userEnd, int channel, int canvasWidth)
{
	double *scalar = (double*)malloc(sizeof(double));
	drwav_uint64 duration = userEnd - userStart;
	double samplesInPixelWith = (double)duration / (double)canvasWidth;
	unsigned int decimation = ceil(samplesInPixelWith / 4);
	if (decimation < 1)
		decimation = 1;
	if (decimation == 1)
	{
		*scalar = duration * channel;
	}
	else
	{
		unsigned int numberOfSmallerList = (unsigned int)(ceil((double)duration / (double)decimation));
		drwav_uint64 lngLen = numberOfSmallerList * decimation;
		unsigned int padding = lngLen - duration;
		unsigned int paddingSide = (unsigned int)(ceil((double)padding / 2.0));
		drwav_uint64 decimatedLen = numberOfSmallerList * 2;
		*scalar = decimatedLen * channel;
	}
	return scalar;
}
void *getWaveform(char *filename, int userStart, int userEnd, int canvasWidth)
{
	unsigned int i;
	drwav pWav;
	// Now load from dr_wav
	if (!drwav_init_file(&pWav, filename, 0))
		return 0;
	if (userStart < 0)
		userStart = 0;
	if (userEnd > pWav.totalPCMFrameCount)
		userEnd = pWav.totalPCMFrameCount;
	drwav_seek_to_pcm_frame(&pWav, userStart);
	drwav_uint64 duration = userEnd - userStart;
	double samplesInPixelWith = (double)duration / (double)canvasWidth;
	unsigned int decimation = ceil(samplesInPixelWith / 4);
	if (decimation < 1)
		decimation = 1;
	float *pPCMFrames_drwav = (float *)malloc((size_t)(duration * pWav.channels * sizeof(float)));
	drwav_uint64 pcmFrameCount_drwav = drwav_read_pcm_frames_f32(&pWav, duration, pPCMFrames_drwav);
	drwav_uninit(&pWav);
	if (decimation == 1)
	{
		double *pPCMFrames_drwav_ = (double*)malloc((size_t)(duration * pWav.channels * sizeof(double)));
		double **pPCMFrames_drwav2 = (double**)malloc(pWav.channels * sizeof(double*));
		for (i = 0; i < pWav.channels; i++)
			pPCMFrames_drwav2[i] = pPCMFrames_drwav_ + i * duration;
		for (i = 0; i < duration * pWav.channels; i++)
			pPCMFrames_drwav2[i % pWav.channels][i / pWav.channels] = pPCMFrames_drwav[i];
		free(pPCMFrames_drwav2);
		free(pPCMFrames_drwav);
		return pPCMFrames_drwav_;
	}
	else
	{
		unsigned int numberOfSmallerList = (unsigned int)(ceil((double)duration / (double)decimation));
		drwav_uint64 lngLen = numberOfSmallerList * decimation;
		unsigned int padding = lngLen - duration;
		unsigned int paddingSide = (unsigned int)(ceil((double)padding / 2.0));
		drwav_uint64 decimatedLen = numberOfSmallerList * 2;
		float *pPCMFrames_drwav_ = (float*)malloc((size_t)((paddingSide + duration + paddingSide) * pWav.channels * sizeof(float)));
		float **pPCMFrames_drwav2 = (float**)malloc(pWav.channels * sizeof(float*));
		for (i = 0; i < pWav.channels; i++)
			pPCMFrames_drwav2[i] = pPCMFrames_drwav_ + i * duration;
		for (unsigned int ch = 0; ch < pWav.channels; ch++)
		{
			for (i = 0; i < paddingSide; i++)
			{
				pPCMFrames_drwav2[ch][i] = 0.0;
				pPCMFrames_drwav2[ch][paddingSide + duration + i] = 0.0;
			}
		}
		for (i = 0; i < duration * pWav.channels; i++)
			pPCMFrames_drwav2[i % pWav.channels][paddingSide + i / pWav.channels] = pPCMFrames_drwav[i];
		free(pPCMFrames_drwav);
		double *pPCMFrames_drwav3 = (double*)malloc((size_t)(decimatedLen * pWav.channels * sizeof(double)));
		double **finalPointsToPlot = (double**)malloc(pWav.channels * sizeof(double*));
		for (i = 0; i < pWav.channels; i++)
			finalPointsToPlot[i] = pPCMFrames_drwav3 + i * decimatedLen;
		for (unsigned int ii = 0; ii < numberOfSmallerList; ii++)
		{
			i = ii * decimation;
			for (unsigned int ch = 0; ch < pWav.channels; ch++)
			{
				float mmin = pPCMFrames_drwav2[ch][1];
				float mmax = pPCMFrames_drwav2[ch][1];
				for (unsigned int j = 1; j < decimation; j++)
				{
					if (pPCMFrames_drwav2[ch][i + j] < mmin)
						mmin = pPCMFrames_drwav2[ch][i + j];
					if (pPCMFrames_drwav2[ch][i + j] > mmax)
						mmax = pPCMFrames_drwav2[ch][i + j];
				}
				finalPointsToPlot[ch][ii] = mmin;
				finalPointsToPlot[ch][ii + numberOfSmallerList] = mmax;
			}
		}
		free(pPCMFrames_drwav_);
		free(pPCMFrames_drwav2);
		free(finalPointsToPlot);
		return pPCMFrames_drwav3;
	}
}
__declspec(dllexport) double *getWaveformFFIReturnDouble(char *filename, int userStart, int userEnd, int canvasWidth)
{
	void *buf = getWaveform(filename, userStart, userEnd, canvasWidth);
	return (double*)buf;
}
unsigned int getTimeFrames(float fs, int fftLen, int userStart, int userEnd, int imgWidth)
{
	RENDER render = { 0 };
	render.log_freq = 1; // Log plot
	render.width = imgWidth;
	unsigned int i;
	unsigned int frameCentre = fftLen >> 1;
	if (userStart > userEnd)
		userStart = userEnd;
	drwav_uint64 end = userEnd;
	drwav_uint64 duration = userEnd - userStart;
	render.samplerate = fs;
	double pixelpersmps = (double)render.width / (double)duration;
	const double pixelsPerSecond = pixelpersmps * render.samplerate;
	const double tstep = 1.0 / pixelsPerSecond;
	const double samplesPerPixel = render.samplerate * tstep;
	unsigned int frameShift = (unsigned int)ceil(samplesPerPixel);
	if (frameShift > fftLen / 2)
		frameShift = fftLen / 2;
	if (frameShift > duration)
		frameShift = upper_power_of_two(duration);
	render.min_time = userStart;
	render.max_time = userEnd;
	render.frameShift = frameShift;
	drwav_uint64 numReads;
	unsigned int seek;
	unsigned int runBlankSamples;
	if (userStart >= fftLen)
	{
		seek = userStart - fftLen;
		runBlankSamples = 0;
	}
	else
	{
		seek = 0;
		runBlankSamples = fftLen - userStart;
	}
	unsigned cnt = seek;
	if (userStart < fftLen)
	{
		drwav_uint64 pcmFrameCount_drwav = 0;
		for (drwav_uint64 iPCMFrame = 0; iPCMFrame < runBlankSamples; iPCMFrame += frameShift)
		{
			if ((runBlankSamples - iPCMFrame) < frameShift)
				pcmFrameCount_drwav = frameShift - (runBlankSamples - iPCMFrame);
		}
		cnt = pcmFrameCount_drwav;
		numReads = ceilf((float)(end - cnt) / (float)frameShift) * frameShift + frameCentre;
	}
	else
		numReads = (end - userStart) + fftLen + frameCentre;
	unsigned int cntBk = cnt;
	unsigned int correspondingPosition = userStart;
	unsigned int idx = 0;
	char draw;
	unsigned int totalFrames = 0;
	for (drwav_uint64 iPCMFrame = 0; iPCMFrame < numReads; iPCMFrame += frameShift)
	{
		if (cnt < (userStart + frameCentre - frameShift))
		{
			draw = 0;
			cnt += frameShift;
		}
		else if (cnt == (userStart + frameCentre - frameShift))
		{
			cnt += frameShift;
			draw = 1;
			//printf("Reading waveform location: %d\n", seek + iPCMFrame);
		}
		else
		{
			correspondingPosition += frameShift;
			draw = 1;
			//printf("Reading waveform location: %d\n", seek + iPCMFrame);
		}
		if (iPCMFrame >= (numReads - frameShift * 2))
		{
			if (iPCMFrame >= (numReads - frameShift * 1))
			{
				draw = 3;
				float normalizeDistBetweenFrame1ToUserEnd = (float)(end - (correspondingPosition - frameShift)) / (float)frameShift;
				float normalizeDistBetweenUserEndToFrame2 = 1.0f - normalizeDistBetweenFrame1ToUserEnd;
			}
			else
			{
				draw = 2;
			}
		}
		if (draw)
		{
			totalFrames++;
			if (correspondingPosition > end)
				break;
		}
	}
	return totalFrames;
}
float *getAxis(float fs, int fftLen, int userStart, int userEnd, double freqStart, double freqEnd, char dbg, int imgWidth, int imgHeight, int totalFrames)
{
	RENDER render = { 0 };
	render.log_freq = 1; // Log plot
	render.min_freq = freqStart;
	render.max_freq = freqEnd;
	render.width = imgWidth;
	render.height = imgHeight;
	unsigned int i;
	render.samplerate = fs;
	double lowerBound = ((double)render.samplerate / (double)fftLen) * 0.5;
	if (render.log_freq)
	{
		if (render.min_freq < lowerBound)
			render.min_freq = lowerBound;
		if (render.max_freq > render.samplerate / 2.0)
			render.max_freq = render.samplerate / 2.0;
	}
	unsigned int frameCentre = fftLen >> 1;
	if (userStart > userEnd)
		userStart = userEnd;
	drwav_uint64 end = userEnd;
	drwav_uint64 duration = userEnd - userStart;
	double pixelpersmps = (double)render.width / (double)duration;
	const double pixelsPerSecond = pixelpersmps * render.samplerate;
	const double tstep = 1.0 / pixelsPerSecond;
	const double samplesPerPixel = render.samplerate * tstep;
	unsigned int frameShift = (unsigned int)ceil(samplesPerPixel);
	if (frameShift > fftLen / 2)
		frameShift = fftLen / 2;
	if (frameShift > duration)
		frameShift = upper_power_of_two(duration);
	render.min_time = userStart;
	render.max_time = userEnd;
	render.frameShift = frameShift;
	drwav_uint64 numReads;
	unsigned int seek;
	unsigned int runBlankSamples;
	//for (i = 0; i < transformer.halfLen; i++)
	//	printf("%1.8f,", i / (float)transformer.halfLen * (pWav.sampleRate / 2.0f));
	if (userStart >= fftLen)
	{
		seek = userStart - fftLen;
		runBlankSamples = 0;
	}
	else
	{
		seek = 0;
		runBlankSamples = fftLen - userStart;
	}
	unsigned cnt = seek;
	if (userStart < fftLen)
	{
		if (dbg)
			printf("Seeking back samples: ");
		drwav_uint64 pcmFrameCount_drwav = 0;
		for (drwav_uint64 iPCMFrame = 0; iPCMFrame < runBlankSamples; iPCMFrame += frameShift)
		{
			if ((runBlankSamples - iPCMFrame) < frameShift)
				pcmFrameCount_drwav = frameShift - (runBlankSamples - iPCMFrame);
			if (dbg)
				printf("-%d, ", runBlankSamples - iPCMFrame);
		}
		cnt = pcmFrameCount_drwav;
		numReads = ceilf((float)(end - cnt) / (float)frameShift) * frameShift + frameCentre;
		if (dbg)
			printf("\n");
	}
	else
		numReads = (end - userStart) + fftLen + frameCentre;
	unsigned int cntBk = cnt;
	unsigned int correspondingPosition = userStart;
	unsigned int idx = 0;
	char draw;
	render.timeAxis = (float *)malloc(totalFrames * sizeof(float));
	idx = 0;
	unsigned int frameCount = 0;
	for (drwav_uint64 iPCMFrame = 0; iPCMFrame < numReads; iPCMFrame += frameShift)
	{
		if (cnt < (userStart + frameCentre - frameShift))
		{
			draw = 0;
			cnt += frameShift;
		}
		else if (cnt == (userStart + frameCentre - frameShift))
		{
			cnt += frameShift;
			draw = 1;
			if (dbg)
				printf("Draw now! Actual position(Corresponding to frame centre): %d\n", correspondingPosition);
			//printf("Reading waveform location: %d\n", seek + iPCMFrame);
		}
		else
		{
			correspondingPosition += frameShift;
			draw = 1;
			if (dbg)
				printf("Draw now! Actual position(Corresponding to frame centre): %d\n", correspondingPosition);
			//printf("Reading waveform location: %d\n", seek + iPCMFrame);
		}
		if (iPCMFrame >= (numReads - frameShift * 2))
		{
			if (iPCMFrame >= (numReads - frameShift * 1))
			{
				draw = 3;
				float normalizeDistBetweenFrame1ToUserEnd = (float)(end - (correspondingPosition - frameShift)) / (float)frameShift;
				float normalizeDistBetweenUserEndToFrame2 = 1.0f - normalizeDistBetweenFrame1ToUserEnd;
				if (dbg)
					printf("Last -0 frame\n");
			}
			else
			{
				draw = 2;
				if (dbg)
					printf("Last -1 frame\n");
			}
		}
		if (draw)
		{
			render.timeAxis[frameCount++] = correspondingPosition;
			if (correspondingPosition > end)
				break;
		}
		//printf("");
	}
	for (i = 0; i < frameCount; i++)
		render.timeAxis[i] = (render.timeAxis[i] - render.timeAxis[0]) / (render.max_time - render.timeAxis[0]);
	// Do this sanity check here, as soon as max_freq has its default value
	if (render.min_freq >= render.max_freq)
	{
		printf("Error : freqStart (%g) must be less than max_freq (%g)\n", render.min_freq, render.max_freq);
		return 0;
	}
	return render.timeAxis;
}
float *getSpecPixel(char *filename, int fftLen, int userStart, int userEnd, double freqStart, double freqEnd,
	int nThreads, char dbg, int imgWidth, int imgHeight, int channels)
{
	RENDER render = { 0 };
	render.log_freq = 1; // Log plot
	render.min_freq = freqStart;
	render.max_freq = freqEnd;
	render.width = imgWidth;
	render.height = imgHeight;
	unsigned int i;
	drwav pWav;
	// Now load from dr_wav
	if (!drwav_init_file(&pWav, filename, 0))
		return 0;
	render.samplerate = pWav.sampleRate;
	double lowerBound = ((double)render.samplerate / (double)fftLen) * 0.5;
	if (render.log_freq)
	{
		if (render.min_freq < lowerBound)
			render.min_freq = lowerBound;
		if (render.max_freq > render.samplerate / 2.0)
			render.max_freq = render.samplerate / 2.0;
	}
	if (channels != pWav.channels)
		return 0;
	unsigned int frameCentre = fftLen >> 1;
	if (userStart > userEnd)
		userStart = userEnd;
	if (userStart > pWav.totalPCMFrameCount)
		userStart = 0;
	drwav_uint64 end = min(userEnd, pWav.totalPCMFrameCount);
	drwav_uint64 duration = end - userStart;
	double pixelpersmps = (double)render.width / (double)duration;
	const double pixelsPerSecond = pixelpersmps * render.samplerate;
	const double tstep = 1.0 / pixelsPerSecond;
	const double samplesPerPixel = render.samplerate * tstep;
	unsigned int frameShift = (unsigned int)ceil(samplesPerPixel);
	if (frameShift > fftLen / 2)
		frameShift = fftLen / 2;
	if (frameShift > duration)
		frameShift = upper_power_of_two(duration);
	render.min_time = userStart;
	render.max_time = end;
	render.frameShift = frameShift;
	int stride = 4 * render.width;
	drwav_uint64 numReads;
	unsigned int seek;
	unsigned int runBlankSamples;
	float *pPCMFrames_drwav = (float *)malloc((size_t)(frameShift * pWav.channels * sizeof(float)));
	if (pPCMFrames_drwav == NULL)
	{
		printf("  [dr_wav] Out of memory");
		return 0;
	}
	render.channels = pWav.channels;
	ConstantQTransform transformer;
	PhaseRadarInit(&transformer, pWav.channels, fftLen);
	float *wavDebuggingBuffer = (float *)malloc(pWav.channels * fftLen * sizeof(float));
	render.img = (imageBuf *)malloc(sizeof(imageBuf));
	initImageBuf(render.img, transformer.halfLen * transformer.channels);
	//for (i = 0; i < transformer.halfLen; i++)
	//	printf("%1.8f,", i / (float)transformer.halfLen * (pWav.sampleRate / 2.0f));
	if (userStart >= fftLen)
	{
		seek = userStart - fftLen;
		runBlankSamples = 0;
	}
	else
	{
		seek = 0;
		runBlankSamples = fftLen - userStart;
	}
	drwav_seek_to_pcm_frame(&pWav, seek);
	unsigned cnt = seek;
	const char useCQT = 0;
	// If starting position is smaller than frame size
	for (i = 0; i < frameShift * pWav.channels; i++)
		pPCMFrames_drwav[i] = 0.0f;
	if (userStart < fftLen)
	{
		if (dbg)
			printf("Seeking back samples: ");
		drwav_uint64 pcmFrameCount_drwav = 0;
		for (drwav_uint64 iPCMFrame = 0; iPCMFrame < runBlankSamples; iPCMFrame += frameShift)
		{
			if ((runBlankSamples - iPCMFrame) < frameShift)
				pcmFrameCount_drwav = drwav_read_pcm_frames_f32(&pWav, frameShift - (runBlankSamples - iPCMFrame), pPCMFrames_drwav + (runBlankSamples - iPCMFrame) * pWav.channels);
			PhaseRadarProcessSamples(&transformer, pPCMFrames_drwav, frameShift, useCQT, 0);
			if (dbg)
				printf("-%d, ", runBlankSamples - iPCMFrame);
		}
		cnt = pcmFrameCount_drwav;
		numReads = ceilf((float)(end - cnt) / (float)frameShift) * frameShift + frameCentre;
		if (dbg)
			printf("\n");
	}
	else
		numReads = (end - userStart) + fftLen + frameCentre;
	unsigned int cntBk = cnt;
	unsigned int correspondingPosition = userStart;
	unsigned int idx = 0;
	char draw;
	unsigned int totalFrames = 0;
	for (drwav_uint64 iPCMFrame = 0; iPCMFrame < numReads; iPCMFrame += frameShift)
	{
		if (cnt < (userStart + frameCentre - frameShift))
		{
			draw = 0;
			cnt += frameShift;
		}
		else if (cnt == (userStart + frameCentre - frameShift))
		{
			cnt += frameShift;
			draw = 1;
			if (dbg)
				printf("Draw now! Actual position(Corresponding to frame centre): %d\n", correspondingPosition);
			//printf("Reading waveform location: %d\n", seek + iPCMFrame);
		}
		else
		{
			correspondingPosition += frameShift;
			draw = 1;
			if (dbg)
				printf("Draw now! Actual position(Corresponding to frame centre): %d\n", correspondingPosition);
			//printf("Reading waveform location: %d\n", seek + iPCMFrame);
		}
		if (iPCMFrame >= (numReads - frameShift * 2))
		{
			if (iPCMFrame >= (numReads - frameShift * 1))
			{
				draw = 3;
				float normalizeDistBetweenFrame1ToUserEnd = (float)(end - (correspondingPosition - frameShift)) / (float)frameShift;
				float normalizeDistBetweenUserEndToFrame2 = 1.0f - normalizeDistBetweenFrame1ToUserEnd;
				if (dbg)
					printf("Last -0 frame\n");
			}
			else
			{
				draw = 2;
				if (dbg)
					printf("Last -1 frame\n");
			}
		}
		if (draw)
		{
			totalFrames++;
			if (correspondingPosition > end)
				break;
		}
	}
	cnt = cntBk;
	correspondingPosition = userStart;
	idx = 0;
	for (drwav_uint64 iPCMFrame = 0; iPCMFrame < numReads; iPCMFrame += frameShift)
	{
		drwav_uint64 pcmFrameCount_drwav = drwav_read_pcm_frames_f32(&pWav, frameShift, pPCMFrames_drwav);
		if (pcmFrameCount_drwav != frameShift) // The end of file
		{
			for (i = 0; i < (frameShift - pcmFrameCount_drwav) * pWav.channels; i++)
				pPCMFrames_drwav[pcmFrameCount_drwav * pWav.channels + i] = 0.0f;
		}
		if (cnt < (userStart + frameCentre - frameShift))
		{
			draw = 0;
			cnt += frameShift;
		}
		else if (cnt == (userStart + frameCentre - frameShift))
		{
			cnt += frameShift;
			draw = 1;
			if (dbg)
				printf("Draw now! Actual position(Corresponding to frame centre): %d\n", correspondingPosition);
			//printf("Reading waveform location: %d\n", seek + iPCMFrame);
		}
		else
		{
			correspondingPosition += frameShift;
			draw = 1;
			if (dbg)
				printf("Draw now! Actual position(Corresponding to frame centre): %d\n", correspondingPosition);
			//printf("Reading waveform location: %d\n", seek + iPCMFrame);
		}
		if (iPCMFrame >= (numReads - frameShift * 2))
		{
			if (iPCMFrame >= (numReads - frameShift * 1))
			{
				draw = 3;
				float normalizeDistBetweenFrame1ToUserEnd = (float)(end - (correspondingPosition - frameShift)) / (float)frameShift;
				float normalizeDistBetweenUserEndToFrame2 = 1.0f - normalizeDistBetweenFrame1ToUserEnd;
				if (dbg)
					printf("Last -0 frame\n");
			}
			else
			{
				draw = 2;
				if (dbg)
					printf("Last -1 frame\n");
			}
		}
		PhaseRadarProcessSamples(&transformer, pPCMFrames_drwav, frameShift, useCQT, draw);
		if (draw)
		{
			if (dbg == 2)
			{
				// copy to temporary buffer and FHT
				for (unsigned int ch = 0; ch < transformer.channels; ch++)
					for (i = 0; i < fftLen; ++i)
						wavDebuggingBuffer[ch * fftLen + i] = transformer.mInput[ch * fftLen + ((i + transformer.mInputPos) & transformer.minus1fftLen)];
				char *filename = createStringVAList("out%d.wav", idx++);
				interleaveChannel(wavDebuggingBuffer, transformer.channels, transformer.delayLineInterleaved, fftLen);
				writeWavTest(filename, transformer.delayLineInterleaved, transformer.channels, pWav.sampleRate, fftLen);
				free(filename);
			}
			writeImageBufIntoMemory(render.img, transformer.mag);
			if (correspondingPosition > end)
				break;
		}
		//printf("");
	}
	drwav_uninit(&pWav);
	free(pPCMFrames_drwav);
	PhaseRadarFree(&transformer);
	free(wavDebuggingBuffer);
	// Do this sanity check here, as soon as max_freq has its default value
	if (render.min_freq >= render.max_freq)
	{
		printf("Error : freqStart (%g) must be less than max_freq (%g)\n", render.min_freq, render.max_freq);
		return 0;
	}
	float *imgDat = render.img->image;
	free(render.img);
	return imgDat;
}
#ifndef _WINDLL
int main()
{
	char *filename = "test (2).wav";
	int fftLen = 4096;
	int halfLen = fftLen / 2 + 1;
	int start = 27336;
	int end = 272336;
	int imgWidth = 1200;
	int wavCh = 2;
	float fs = 44100.0f;
	unsigned int totalFrames = getTimeFrames(fs, fftLen, start, end, imgWidth);
	float *axis = getAxis(fs, fftLen, start, end, 1, 22050, 0, imgWidth, 381, totalFrames);
	for (int i = 0; i < totalFrames; i++)
		printf("%1.8f\n", axis[i]);
	size_t memSize = halfLen * totalFrames * wavCh;
	float *buf2 = getSpecPixel(filename, fftLen, start, end, 1, 22050, 1, 0, imgWidth, 381, wavCh);
	free(axis);
	free(buf2);
	return 0;
}
/*int main()
{
	char *filename = "test (2).wav";
	double *totalMemorySize = spectrogramFFIConfirmMemorySize(5, 381, 2);
	free(totalMemorySize);
	double timeTotal = 0;
	int trials = 10;
	int instances = 24;
	for (int trial = 0; trial < trials; trial++)
	{
		double bb = get_wall_time();
		int numThreads = 8;
		double eta[24];
		int instance;
#pragma omp parallel for default(none) num_threads(numThreads) private(instance)
		for (instance = 0; instance < instances; instance++)
		{
			int tid = omp_get_thread_num();
			double cc = get_wall_time();
			double *buf2 = getSpecPixel(filename, 4096, 371, 272336, 1, 22050, 1, 0, 5, 381, 2, 0);
			free(buf2);
			eta[instance] = get_wall_time() - cc;
		}
		for (instance = 0; instance < instances; instance++)
			printf("Instance %d ETA = %1.14lf\n", instance + 1, eta[instance] * 1000.0);
		double aa = get_wall_time();
		timeTotal += (aa - bb);
	}
	printf("%1.14lf\n", timeTotal / trials * 1000.0);
	system("pause");
	return 0;
}*/
/*int main()
{
	char *filename = "test.wav";
	unsigned int fftLen = 4096;
	drwav_uint64 userStart = 30051;
	drwav_uint64 userEnd = 90052;
	int nThreads = 8;
	char dbg = 0;
	int ch = 2;
	//double *info = wavFileInfo(filename);
	double ticksCoordinate[64], ticksLabel[64];
	int numTicks;
	double *totalMemorySize = spectrogramFFIConfirmMemorySize(1800, 210, ch);
	double *wavMemorySize = waveformFFIConfirmMemorySize(0, 7847888, 2, 300);
	double *buf3 = getWaveformFFIReturnDouble("test.wav", 0, 7847888, 300);
	FILE *fp2 = fopen("bb.dat", "wb");
	fwrite(buf3, sizeof(double), (size_t)(*wavMemorySize), fp2);
	fclose(fp2);
	free(wavMemorySize);
	free(buf3);
	fclose(fp2);
	//double *buf2 = spectrogramFFIReturnDouble(filename, fftLen, userStart, userEnd, 20.0, 20000.0, 8, dbg, 1920, 1080, ch);
	double *buf2 = spectrogramFFIReturnDouble(filename, 4096, 0, 8512435, 1, 22000, 8, 0, 1800, 210, 2, 0);
	double *buf1 = spectrogramFFIReturnDouble(filename, 4096, 0, 8512435, 1, 22000, 8, 0, 1800, 210, 2, 1);
	for (int i = 0; i < 1 + 64 * 2; i++)
	{
		printf("%1.14lf %1.14lf\n", buf2[i], buf1[i]);
	}
	FILE *fp = fopen("aa.dat", "wb");
	fwrite(buf2 + 1 + 64 * 2, sizeof(double), 1800 * 210 * 2, fp);
	fclose(fp);
	free(totalMemorySize);
	free(buf2);
	free(buf1);
	return 0;
}*/
#endif