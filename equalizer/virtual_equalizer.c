/*-
 * Copyright (c) 2019 Google LLC, written by Richard Kralovic <riso@google.com>
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR AND CONTRIBUTORS ``AS IS'' AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE AUTHOR OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
 * OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 */

#include <err.h>
#include <errno.h>
#include <fcntl.h>
#include <getopt.h>
#include <math.h>
#include <pthread.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sysexits.h>
#include <sys/ioctl.h>
#include <sys/socket.h>
#include <sys/soundcard.h>
#include <sys/types.h>
#include <sys/un.h>
#include <unistd.h>
#include <complex.h>
#include <alloca.h>
#include <stdio.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#define SWAP_COMPLEX(a, b) do { double complex temp = (a); (a) = (b); (b) = temp; } while(0)
#define IS_POWER_OF_TWO(n) ((n) > 0 && ((n) & ((n) - 1)) == 0)
#define ERROR_RETURN(msg) do { fprintf(stderr, "%s\n", msg); return; } while(0)
#define SMALL_FFT_THRESHOLD 1024
#define USE_THREAD_PARALLELISM_THRESHOLD 8192

static void *aligned_malloc(size_t size, size_t alignment) {
	void *ptr = NULL;
	if (posix_memalign(&ptr, alignment, size) != 0) return NULL;
	return ptr;
}
#define aligned_free(ptr) free(ptr)

static double *cos_table = NULL;
static double *sin_table = NULL;
static int max_table_size = 0;
static pthread_mutex_t twiddle_mutex = PTHREAD_MUTEX_INITIALIZER;
static int cpu_has_sse3 = 0;
static int cpu_has_avx = 0;
static int cpu_features_detected = 0;
static double complex *fft_buffer = NULL;
static int fft_buffer_size = 0;
static pthread_mutex_t buffer_mutex = PTHREAD_MUTEX_INITIALIZER;

#include "../virtual_utils.h"
#include "../virtual_oss.h"

struct Equalizer {
	double	rate;
	int	block_size;
	int	do_normalize;

	/* (block_size * 2) elements, time domain */
	double *fftw_time;

	/* (block_size * 2) elements, half-complex, freq domain */
	double *fftw_freq;

	// fftw_plan forward;
	// fftw_plan inverse;
};

static int be_silent = 0;

static void detect_cpu_features() {
	if (cpu_features_detected) return;
	
	size_t len = sizeof(int);
	sysctlbyname("hw.instruction_sse3", &cpu_has_sse3, &len, NULL, 0);
	sysctlbyname("hw.instruction_avx", &cpu_has_avx, &len, NULL, 0);
	cpu_features_detected = 1;
}

static void init_twiddle_factors(int n) {
	if (cos_table != NULL && n <= max_table_size) return;
	
	pthread_mutex_lock(&twiddle_mutex);
	
	if (cos_table != NULL && n <= max_table_size) {
		pthread_mutex_unlock(&twiddle_mutex);
		return;
	}
	
	if (cos_table) {
		free(cos_table);
		free(sin_table);
	}
	
	int new_size = 1;
	while (new_size < n) new_size <<= 1;
	
	cos_table = (double*)aligned_malloc(new_size * sizeof(double), 64);
	sin_table = (double*)aligned_malloc(new_size * sizeof(double), 64);
	
	if (!cos_table || !sin_table) {
		fprintf(stderr, "Failed to allocate twiddle factor tables\n");
		pthread_mutex_unlock(&twiddle_mutex);
		exit(1);
	}
	
	#pragma omp parallel for if(new_size > 1024)
	for (int i = 0; i < new_size; i++) {
		double angle = -2.0 * M_PI * i / new_size;
		cos_table[i] = cos(angle);
		sin_table[i] = sin(angle);
	}
	
	max_table_size = new_size;
	pthread_mutex_unlock(&twiddle_mutex);
}

static double complex* get_fft_buffer(int n, int use_stack) {
	if (use_stack) return (double complex*)alloca(n * sizeof(double complex));
	
	pthread_mutex_lock(&buffer_mutex);
	
	if (fft_buffer == NULL || n > fft_buffer_size) {
		if (fft_buffer) aligned_free(fft_buffer);
		
		fft_buffer = (double complex*)aligned_malloc(n * sizeof(double complex), 64);
		if (!fft_buffer) {
			pthread_mutex_unlock(&buffer_mutex);
			return NULL;
		}
		
		fft_buffer_size = n;
	}
	
	double complex *buffer = fft_buffer;
	pthread_mutex_unlock(&buffer_mutex);
	
	return buffer;
}

static void bit_reversal_permutation(double complex *X, int n) {
	int log2n = 0;
	int temp = n;
	while (temp >>= 1) log2n++;
	
	for (int i = 1, j = 0; i < n; i++) {
		int bit = n >> 1;
		
		while (j >= bit) {
			j -= bit;
			bit >>= 1;
		}
		j += bit;
		
		if (i < j) SWAP_COMPLEX(X[i], X[j]);
	}
}

static void butterfly_operations(double complex *X, int n, int len, int is_inverse) {
	int half_len = len >> 1;
	int table_step = n / len;
	double sign = is_inverse ? 1.0 : -1.0;
	
	#pragma omp parallel for if(n >= USE_THREAD_PARALLELISM_THRESHOLD)
	for (int i = 0; i < n; i += len) {
		for (int j = 0; j < half_len; j++) {
			int idx = j * table_step;
			double wr = cos_table[idx];
			double wi = sign * sin_table[idx];
			
			int pos1 = i + j;
			int pos2 = pos1 + half_len;
			
			double complex u = X[pos1];
			double complex v = X[pos2] * (wr + I * wi);
			
			X[pos1] = u + v;
			X[pos2] = u - v;
		}
	}
}

static void custom_fft_core(double *data, double *result, int n, int is_inverse) {
	if (!IS_POWER_OF_TWO(n)) ERROR_RETURN("FFT length must be a power of two");
	
	init_twiddle_factors(n);
	if (!cpu_features_detected) detect_cpu_features();
	
	int use_stack = (n <= SMALL_FFT_THRESHOLD);
	double complex *X = get_fft_buffer(n, use_stack);
	if (!X) ERROR_RETURN("Failed to get FFT buffer");
	
	if (!is_inverse) {
		for (int i = 0; i < n; i++) X[i] = data[i];
	} else {
		X[0] = data[0];
		X[n/2] = data[n/2];
		
		for (int i = 1; i < n/2; i++) {
			X[i] = data[i] + I * data[n-i];
			X[n-i] = data[i] - I * data[n-i];
		}
	}
	
	bit_reversal_permutation(X, n);
	
	for (int len = 2; len <= n; len <<= 1) {
		butterfly_operations(X, n, len, is_inverse);
	}
	
	if (!is_inverse) {
		result[0] = creal(X[0]);
		result[n/2] = creal(X[n/2]);
		
		for (int i = 1; i < n/2; i++) {
			result[i] = creal(X[i]);
			result[n-i] = cimag(X[i]);
		}
	} else {
		double scale = 1.0 / n;
		
		for (int i = 0; i < n; i++) {
			result[i] = creal(X[i]) * scale;
		}
	}
}

static void custom_fft_r2hc(double *data, double *result, int n) {
	custom_fft_core(data, result, n, 0);
}

static void custom_ifft_hc2r(double *data, double *result, int n) {
	custom_fft_core(data, result, n, 1);
}

static void cleanup_fft(void) {
	pthread_mutex_lock(&twiddle_mutex);
	if (cos_table) {
		aligned_free(cos_table);
		cos_table = NULL;
	}
	if (sin_table) {
		aligned_free(sin_table);
		sin_table = NULL;
	}
	max_table_size = 0;
	pthread_mutex_unlock(&twiddle_mutex);
	
	pthread_mutex_lock(&buffer_mutex);
	if (fft_buffer) {
		aligned_free(fft_buffer);
		fft_buffer = NULL;
	}
	fft_buffer_size = 0;
	pthread_mutex_unlock(&buffer_mutex);
}

static void
message(const char *fmt,...)
{
	va_list list;

	if (be_silent)
		return;
	va_start(list, fmt);
	vfprintf(stderr, fmt, list);
	va_end(list);
}

/*
 * Masking window value for -1 < x < 1.
 *
 * Window must be symmetric, thus, this function is queried for x >= 0
 * only. Currently a Hann window.
 */
static double
equalizer_get_window(double x)
{
	return (0.5 + 0.5 * cos(M_PI * x));
}

static int
equalizer_load_freq_amps(struct Equalizer *e, const char *config)
{
	double prev_f = 0.0;
	double prev_amp = 1.0;
	double next_f = 0.0;
	double next_amp = 1.0;
	int i;

	if (strncasecmp(config, "normalize", 4) == 0) {
		while (*config != 0) {
			if (*config == '\n') {
				config++;
				break;
			}
			config++;
		}
		e->do_normalize = 1;
	} else {
		e->do_normalize = 0;
	}

	for (i = 0; i <= (e->block_size / 2); ++i) {
		const double f = (i * e->rate) / e->block_size;

		while (f >= next_f) {
			prev_f = next_f;
			prev_amp = next_amp;

			if (*config == 0) {
				next_f = e->rate;
				next_amp = prev_amp;
			} else {
				int len;

				if (sscanf(config, "%lf %lf %n", &next_f, &next_amp, &len) == 2) {
					config += len;
					if (next_f < prev_f) {
						message("Parse error: Nonincreasing sequence of frequencies.\n");
						return (0);
					}
				} else {
					message("Parse error.\n");
					return (0);
				}
			}
			if (prev_f == 0.0)
				prev_amp = next_amp;
		}
		e->fftw_freq[i] = ((f - prev_f) / (next_f - prev_f)) * (next_amp - prev_amp) + prev_amp;
	}
	return (1);
}

static void
equalizer_init(struct Equalizer *e, int rate, int block_size)
{
	size_t buffer_size;

	e->rate = rate;
	e->block_size = block_size;

	buffer_size = sizeof(double) * e->block_size;

	e->fftw_time = (double *)malloc(buffer_size);
	e->fftw_freq = (double *)malloc(buffer_size);

	// e->forward = fftw_plan_r2r_1d(block_size, e->fftw_time, e->fftw_freq,
	//     FFTW_R2HC, FFTW_MEASURE);
	// e->inverse = fftw_plan_r2r_1d(block_size, e->fftw_freq, e->fftw_time,
	//     FFTW_HC2R, FFTW_MEASURE);
	// No need to create FFTW plans, as we're using our own FFT functions.
}

static int
equalizer_load(struct Equalizer *eq, const char *config)
{
	int retval = 0;
	int N = eq->block_size;
	int buffer_size = sizeof(double) * N;
	int i;

	memset(eq->fftw_freq, 0, buffer_size);

	message("\n\nReloading amplification specifications:\n%s\n", config);

	if (!equalizer_load_freq_amps(eq, config))
		goto end;

	double *requested_freq = (double *)malloc(buffer_size);

	memcpy(requested_freq, eq->fftw_freq, buffer_size);

	// fftw_execute(eq->inverse);
	custom_ifft_hc2r(eq->fftw_freq, eq->fftw_time, eq->block_size);


	/* Multiply by symmetric window and shift */
	for (i = 0; i < (N / 2); ++i) {
		double weight = equalizer_get_window(i / (double)(N / 2)) / N;

		eq->fftw_time[N / 2 + i] = eq->fftw_time[i] * weight;
	}
	for (i = (N / 2 - 1); i > 0; --i) {
		eq->fftw_time[i] = eq->fftw_time[N - i];
	}
	eq->fftw_time[0] = 0;

	// fftw_execute(eq->forward);
	custom_fft_r2hc(eq->fftw_time, eq->fftw_freq, eq->block_size);

	for (i = 0; i < N; ++i) {
		eq->fftw_freq[i] /= (double)N;
	}

	/* Debug output */
	for (i = 0; i <= (N / 2); ++i) {
		double f = (eq->rate / N) * i;
		double a = sqrt(pow(eq->fftw_freq[i], 2.0) +
		    ((i > 0 && i < N / 2) ? pow(eq->fftw_freq[N - i], 2.0) : 0));

		a *= N;
		double r = requested_freq[i];

		message("%3.1lf Hz: requested %2.2lf, got %2.7lf (log10 = %.2lf), %3.7lfdb\n",
		    f, r, a, log(a) / log(10), (log(a / r) / log(10.0)) * 10.0);
	}

	/* Normalize FIR filter, if any */
	if (eq->do_normalize) {
		double sum = 0;

		for (i = 0; i < N; ++i)
			sum += fabs(eq->fftw_time[i]);
		if (sum != 0.0) {
			for (i = 0; i < N; ++i)
				eq->fftw_time[i] /= sum;
		}
	}
	for (i = 0; i < N; ++i) {
		message("%.3lf ms: %.10lf\n", 1000.0 * i / eq->rate, eq->fftw_time[i]);
	}

	/* End of debug */

	retval = 1;

	free(requested_freq);
end:
	return (retval);
}

static void
equalizer_done(struct Equalizer *eq)
{

	// fftw_destroy_plan(eq->forward);
	// fftw_destroy_plan(eq->inverse);
	// No cleanup needed for our custom FFT.
	free(eq->fftw_time);
	free(eq->fftw_freq);
}

static struct option equalizer_opts[] = {
	{"device", required_argument, NULL, 'd'},
	{"part", required_argument, NULL, 'p'},
	{"channels", required_argument, NULL, 'c'},
	{"what", required_argument, NULL, 'w'},
	{"off", no_argument, NULL, 'o'},
	{"quiet", no_argument, NULL, 'q'},
	{"file", no_argument, NULL, 'f'},
	{"help", no_argument, NULL, 'h'},
};

static void
usage()
{
	message("Usage: virtual_equalizer -d /dev/vdsp.ctl \n"
	    "\t -d, --device [control device]\n"
	    "\t -w, --what [rx_dev,tx_dev,rx_loop,tx_loop, default tx_dev]\n"
	    "\t -p, --part [part number, default 0]\n"
	    "\t -c, --channels [channels, default -1]\n"
	    "\t -f, --file [read input from file, default standard input]\n"
	    "\t -o, --off [disable equalizer]\n"
	    "\t -q, --quiet\n"
	    "\t -h, --help\n");
	exit(EX_USAGE);
}

int
equalizer_main(int argc, char **argv)
{
	struct virtual_oss_fir_filter fir = {};
	struct virtual_oss_io_info info = {};

	struct Equalizer e;

	char buffer[65536];
	unsigned cmd_fir_set = VIRTUAL_OSS_SET_TX_DEV_FIR_FILTER;
	unsigned cmd_fir_get = VIRTUAL_OSS_GET_TX_DEV_FIR_FILTER;
	unsigned cmd_info = VIRTUAL_OSS_GET_DEV_INFO;
	const char *dsp = NULL;
	int rate;
	int channels = -1;
	int part = 0;
	int opt;
	int len;
	int offset;
	int disable = 0;
	int f = STDIN_FILENO;

	while ((opt = getopt_long(argc, argv, "d:c:f:op:w:qh",
	    equalizer_opts, NULL)) != -1) {
		switch (opt) {
		case 'd':
			dsp = optarg;
			break;
		case 'c':
			channels = atoi(optarg);
			if (channels == 0) {
				message("Wrong number of channels\n");
				usage();
			}
			break;
		case 'p':
			part = atoi(optarg);
			if (part < 0) {
				message("Invalid part number\n");
				usage();
			}
			break;
		case 'w':
			if (strcmp(optarg, "rx_dev") == 0) {
				cmd_fir_set = VIRTUAL_OSS_SET_RX_DEV_FIR_FILTER;
				cmd_fir_get = VIRTUAL_OSS_GET_RX_DEV_FIR_FILTER;
				cmd_info = VIRTUAL_OSS_GET_DEV_INFO;
			} else if (strcmp(optarg, "tx_dev") == 0) {
				cmd_fir_set = VIRTUAL_OSS_SET_TX_DEV_FIR_FILTER;
				cmd_fir_get = VIRTUAL_OSS_GET_TX_DEV_FIR_FILTER;
				cmd_info = VIRTUAL_OSS_GET_DEV_INFO;
			} else if (strcmp(optarg, "rx_loop") == 0) {
				cmd_fir_set = VIRTUAL_OSS_SET_RX_LOOP_FIR_FILTER;
				cmd_fir_get = VIRTUAL_OSS_GET_RX_LOOP_FIR_FILTER;
				cmd_info = VIRTUAL_OSS_GET_LOOP_INFO;
			} else if (strcmp(optarg, "tx_loop") == 0) {
				cmd_fir_set = VIRTUAL_OSS_SET_TX_LOOP_FIR_FILTER;
				cmd_fir_get = VIRTUAL_OSS_GET_TX_LOOP_FIR_FILTER;
				cmd_info = VIRTUAL_OSS_GET_LOOP_INFO;
			} else {
				message("Bad -w argument not recognized\n");
				usage();
			}
			break;
		case 'f':
			if (f != STDIN_FILENO) {
				message("Can only specify one file\n");
				usage();
			}
			f = open(optarg, O_RDONLY);
			if (f < 0) {
				message("Cannot open specified file\n");
				usage();
			}
			break;
		case 'o':
			disable = 1;
			break;
		case 'q':
			be_silent = 1;
			break;
		default:
			usage();
		}
	}

	fir.number = part;
	info.number = part;

	int fd = open(dsp, O_RDWR);

	if (fd < 0) {
		message("Cannot open DSP device\n");
		return (EX_SOFTWARE);
	}
	if (ioctl(fd, VIRTUAL_OSS_GET_SAMPLE_RATE, &rate) < 0) {
		message("Cannot get sample rate\n");
		return (EX_SOFTWARE);
	}
	if (ioctl(fd, cmd_fir_get, &fir) < 0) {
		message("Cannot get current FIR filter\n");
		return (EX_SOFTWARE);
	}
	if (disable) {
	  	for (fir.channel = 0; fir.channel != channels; fir.channel++) {
			if (ioctl(fd, cmd_fir_set, &fir) < 0) {
				if (fir.channel == 0) {
					message("Cannot disable FIR filter\n");
					return (EX_SOFTWARE);
				}
				break;
			}
		}
		return (0);
	}
	equalizer_init(&e, rate, fir.filter_size);
	equalizer_load(&e, "");

	if (f == STDIN_FILENO) {
		if (ioctl(fd, cmd_info, &info) < 0) {
			message("Cannot read part information\n");
			return (EX_SOFTWARE);
		}
		message("Please enter EQ layout for %s, <freq> <gain>:\n", info.name);
	}
	offset = 0;
	while (1) {
		if (offset == (int)(sizeof(buffer) - 1)) {
			message("Too much input data\n");
			return (EX_SOFTWARE);
		}
		len = read(f, buffer + offset, sizeof(buffer) - 1 - offset);
		if (len <= 0)
			break;
		offset += len;
	}
	buffer[offset] = 0;
	close(f);

	if (f == STDIN_FILENO)
		message("Loading new EQ layout\n");

	if (equalizer_load(&e, buffer) == 0) {
		message("Invalid equalizer data\n");
		return (EX_SOFTWARE);
	}
	fir.filter_data = e.fftw_time;

	for (fir.channel = 0; fir.channel != channels; fir.channel++) {
		if (ioctl(fd, cmd_fir_set, &fir) < 0) {
			if (fir.channel == 0)
				message("Cannot set FIR filter on channel\n");
			break;
		}
	}

	close(fd);
	equalizer_done(&e);

	return (0);
}
