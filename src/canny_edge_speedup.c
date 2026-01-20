/*******************************************************************************
 * --------------------------------------------
 *(c) 2001 University of South Florida, Tampa
 * Use, or copying without permission prohibited.
 * PERMISSION TO USE
 * In transmitting this software, permission to use for research and
 * educational purposes is hereby granted.  This software may be copied for
 * archival and backup purposes only.  This software may not be transmitted
 * to a third party without prior permission of the copyright holder. This
 * permission may be granted only by Mike Heath or Prof. Sudeep Sarkar of
 * University of South Florida (sarkar@csee.usf.edu). Acknowledgment as
 * appropriate is respectfully requested.
 *
 *  Heath, M., Sarkar, S., Sanocki, T., and Bowyer, K. Comparison of edge
 *    detectors: a methodology and initial study, Computer Vision and Image
 *    Understanding 69 (1), 38-54, January 1998.
 *  Heath, M., Sarkar, S., Sanocki, T. and Bowyer, K.W. A Robust Visual
 *    Method for Assessing the Relative Performance of Edge Detection
 *    Algorithms, IEEE Transactions on Pattern Analysis and Machine
 *    Intelligence 19 (12),  1338-1359, December 1997.
 *  ------------------------------------------------------
 *
 * PROGRAM: canny_edge
 * PURPOSE: This program implements a "Canny" edge detector. The processing
 * steps are as follows:
 *
 *   1) Convolve the image with a separable gaussian filter.
 *   2) Take the dx and dy the first derivatives using [-1,0,1] and [1,0,-1]'.
 *   3) Compute the magnitude: sqrt(dx*dx+dy*dy).
 *   4) Perform non-maximal suppression.
 *   5) Perform hysteresis.
 *
 * The user must input three parameters. These are as follows:
 *
 *   sigma = The standard deviation of the gaussian smoothing filter.
 *   tlow  = Specifies the low value to use in hysteresis. This is a
 *           fraction (0-1) of the computed high threshold edge strength value.
 *   thigh = Specifies the high value to use in hysteresis. This fraction (0-1)
 *           specifies the percentage point in a histogram of the gradient of
 *           the magnitude. Magnitude values of zero are not counted in the
 *           histogram.
 *
 * NAME: Mike Heath
 *       Computer Vision Laboratory
 *       University of South Floeida
 *       heath@csee.usf.edu
 *
 * DATE: 2/15/96
 *
 * Modified: 5/17/96 - To write out a floating point RAW headerless file of
 *                     the edge gradient "up the edge" where the angle is
 *                     defined in radians counterclockwise from the x direction.
 *                     (Mike Heath)
 *******************************************************************************/

#ifndef VERBOSE
#define VERBOSE 0
#endif

#define BOOSTBLURFACTOR 90.0

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>  
#include <omp.h>
#include <arm_neon.h>
#include <stdlib.h>

int read_pgm_image(char *infilename, unsigned char **image, int *rows,
                   int *cols);
int write_pgm_image(char *outfilename, unsigned char *image, int rows,
                    int cols, char *comment, int maxval);

void canny(unsigned char *image, int rows, int cols, float sigma,
           float tlow, float thigh, unsigned char **edge, char *fname);
short int* gaussian_smooth(unsigned char *image, int rows, int cols, float sigma);
void make_gaussian_kernel(float sigma, float **kernel, int *windowsize);
void derrivative_x_y(short int *smoothedim, int rows, int cols,
                     short int **delta_x, short int **delta_y);
void magnitude_x_y(short int *delta_x, short int *delta_y, int rows, int cols,
                   short int *magnitude);
void apply_hysteresis(short int *mag, unsigned char *nms, int rows, int cols,
                      float tlow, float thigh, unsigned char *edge);
void radian_direction(short int *delta_x, short int *delta_y, int rows,
                      int cols, float **dir_radians, int xdirtag, int ydirtag);
double angle_radians(double x, double y);

void non_max_supp(short *mag, short *gradx, short *grady, int nrows,
                  int ncols, unsigned char *result);

int main(int argc, char *argv[])
{
  char *infilename = NULL;  /* Name of the input image */
  char *dirfilename = NULL; /* Name of the output gradient direction image */
  char outfilename[128];    /* Name of the output "edge" image */
  char composedfname[128];  /* Name of the output "direction" image */
  unsigned char *image;     /* The input image */
  unsigned char *edge;      /* The output edge image */
  int rows, cols;           /* The dimensions of the image. */
  float sigma=2.5,          /* Standard deviation of the gaussian kernel. */
        tlow=0.5,           /* Fraction of the high threshold in hysteresis. */
        thigh=0.5;          /* High hysteresis threshold control. The actual
                               threshold is the (100 * thigh) percentage point
                               in the histogram of the magnitude of the
                               gradient image that passes non-maximal
                               suppression. */

/* --- TIMING VARIABELEN (TOTAAL) --- */
struct timeval start, end;
  double elapsed_ms;
  

                               

  /****************************************************************************
   * Get the command line arguments.
   ****************************************************************************/
  if (argc < 2) {
    fprintf(stderr,"USAGE: %s <image>\n", argv[0]);
    fprintf(stderr,"\n   <image>:    The image to process. Must be in ");
    fprintf(stderr,"PGM format.\n");
    exit(1);
  }

  infilename = argv[1];


  /****************************************************************************
   * Read in the image. This read function allocates memory for the image.
   ****************************************************************************/
  if (VERBOSE)
    printf("Reading the image %s.\n", infilename);

  if (read_pgm_image(infilename, &image, &rows, &cols) == 0) {
    fprintf(stderr, "Error reading the input image, %s.\n", infilename);
    exit(1);
  }

  /****************************************************************************
   * Perform the edge detection. All of the work takes place here.
   ****************************************************************************/
  if (VERBOSE)
    printf("Starting Canny edge detection.\n");

  if (dirfilename != NULL) {
    sprintf(composedfname, "%s_s_%3.2f_l_%3.2f_h_%3.2f.fim", infilename,
            sigma, tlow, thigh);
    dirfilename = composedfname;
  }
    // Start de timer
  gettimeofday(&start, NULL);
  //THE function
  canny(image, rows, cols, sigma, tlow, thigh, &edge, dirfilename);
  //THE function
  gettimeofday(&end, NULL);

  // Bereken en print de tijd
  elapsed_ms = (end.tv_sec - start.tv_sec) * 1000.0;
  elapsed_ms += (end.tv_usec - start.tv_usec) / 1000.0;

  printf("Execution time: %.4f ms\n", elapsed_ms);
  
  /****************************************************************************
   * Write out the edge image to a file.
   ****************************************************************************/
  sprintf(outfilename, "build/%s_s_%3.2f_l_%3.2f_h_%3.2f.pgm", infilename,
          sigma, tlow, thigh);

  if (VERBOSE)
    printf("Writing the edge iname in the file %s.\n", outfilename);

  if (write_pgm_image(outfilename, edge, rows, cols, "", 255) == 0) {
    fprintf(stderr, "Error writing the edge image, %s.\n", outfilename);
    exit(1);
  }

  free(image);
  free(edge);

  
  return 0;
}

/*******************************************************************************
 * PROCEDURE: fused_derivative_magnitude
 * PURPOSE: Combineert de afgeleiden en magnitude berekeningen
 *      Voorkomen van 'memory spilling' naar het RAM. Data blijft in registers.
 * NAME: Tristan Ploeger
 * DATE: 20/01/2026
 *******************************************************************************/
void fused_derivative_magnitude(short int *smoothedim, int rows, int cols,
                                short int *delta_x, short int *delta_y,
                                short int *magnitude)
{
   int r, c, pos;
   short dx, dy;
   int sq1, sq2;

   // OpenMP verdeelt de rijen over de 4 kernen
   #pragma omp parallel for private(c, pos, dx, dy, sq1, sq2)
   for(r=0; r<rows; r++){
      for(c=0; c<cols; c++){
         pos = r * cols + c;

          // AFGELEIDE X 
         if(c == 0) dx = smoothedim[pos+1] - smoothedim[pos];
         else if(c == cols-1) dx = smoothedim[pos] - smoothedim[pos-1];
         else dx = smoothedim[pos+1] - smoothedim[pos-1];
         delta_x[pos] = dx; // We slaan het nog op voor de NMS later

         // AFGELEIDE Y 
         if(r == 0) dy = smoothedim[pos+cols] - smoothedim[pos];
         else if(r == rows-1) dy = smoothedim[pos] - smoothedim[pos-cols];
         else dy = smoothedim[pos+cols] - smoothedim[pos-cols];
         delta_y[pos] = dy; // We slaan het nog op voor de NMS later

         // MAGNITUDE 
         sq1 = (int)dx * (int)dx;
         sq2 = (int)dy * (int)dy;
         magnitude[pos] = (short)(0.5 + sqrt((float)sq1 + (float)sq2));
      }
   }
}

/*******************************************************************************
 * PROCEDURE: canny
 * PURPOSE: To perform canny edge detection.
 * NAME: Mike Heath
 * DATE: 2/15/96
 *******************************************************************************/
void canny(unsigned char *image, int rows, int cols, float sigma,
           float tlow, float thigh, unsigned char **edge, char *fname)
{
  FILE *fpdir=NULL;          /* File to write the gradient image to.     */
  unsigned char *nms;        /* Points that are local maximal magnitude. */
  short int *smoothedim,     /* The image after gaussian smoothing.      */
            *delta_x,        /* The first devivative image, x-direction. */
            *delta_y,        /* The first derivative image, y-direction. */
            *magnitude;      /* The magnitude of the gadient image.      */
  float *dir_radians=NULL;   /* Gradient direction image.                */

  /****************************************************************************
   * Perform gaussian smoothing on the image using the input standard
   * deviation.
   ****************************************************************************/
  if (VERBOSE)
    printf("Smoothing the image using a gaussian kernel.\n");

  smoothedim = gaussian_smooth(image, rows, cols, sigma);

  /****************************************************************************
   * Compute the first derivative in the x and y directions.
   ****************************************************************************/
  if (VERBOSE)
    printf("Computing the X and Y first derivatives.\n");

  /****************************************************************************
   * Implementation of Loop Fusion: Pre-allocating buffers to allow 
   * the fused kernel to compute derivatives and magnitude in a single pass.
   ****************************************************************************/
  delta_x = (short *) malloc(rows*cols* sizeof(short));
  delta_y = (short *) malloc(rows*cols* sizeof(short));
  magnitude = (short *) malloc(rows*cols* sizeof(short));

  fused_derivative_magnitude(smoothedim, rows, cols, delta_x, delta_y, magnitude);

  /****************************************************************************
   * This option to write out the direction of the edge gradient was added
   * to make the information available for computing an edge quality figure
   * of merit.
   ****************************************************************************/
  if (fname != NULL) {

    /*************************************************************************
     * Compute the direction up the gradient, in radians that are
     * specified counteclockwise from the positive x-axis.
     *************************************************************************/
    radian_direction(delta_x, delta_y, rows, cols, &dir_radians, -1, -1);

    /*************************************************************************
     * Write the gradient direction image out to a file.
     *************************************************************************/
    if ((fpdir = fopen(fname, "wb")) == NULL) {
      fprintf(stderr, "Error opening the file %s for writing.\n", fname);
      exit(1);
    }

    fwrite(dir_radians, sizeof(float), rows*cols, fpdir);
    fclose(fpdir);
    free(dir_radians);
  }

  /****************************************************************************
   * Compute the magnitude of the gradient.
   ****************************************************************************/
  if (VERBOSE)
    printf("Computing the magnitude of the gradient.\n");

  /* Note: Magnitude has been computed within the fused_derivative_magnitude call 
     to optimize cache locality and reduce memory bus contention. */

  /****************************************************************************
   * Perform non-maximal suppression.
   ****************************************************************************/
  if (VERBOSE)
    printf("Doing the non-maximal suppression.\n");

  if ((nms = (unsigned char *) malloc(rows*cols*sizeof(unsigned char)))==NULL) {
    fprintf(stderr, "Error allocating the nms image.\n");
    exit(1);
  }
  non_max_supp(magnitude, delta_x, delta_y, rows, cols, nms);

  /****************************************************************************
   * Use hysteresis to mark the edge pixels.
   ****************************************************************************/
  if (VERBOSE)
    printf("Doing hysteresis thresholding.\n");

  if ( (*edge=(unsigned char *)malloc(rows*cols*sizeof(unsigned char))) == NULL ) {
    fprintf(stderr, "Error allocating the edge image.\n");
    exit(1);
  }

  apply_hysteresis(magnitude, nms, rows, cols, tlow, thigh, *edge);

  /****************************************************************************
   * Free all of the memory that we allocated except for the edge image that
   * is still being used to store out result.
   ****************************************************************************/
  free(smoothedim);
  free(delta_x);
  free(delta_y);
  free(magnitude);
  free(nms);
}

/*******************************************************************************
 * Procedure: radian_direction
 * Purpose: To compute a direction of the gradient image from component dx and
 * dy images. Because not all derriviatives are computed in the same way, this
 * code allows for dx or dy to have been calculated in different ways.
 *
 * FOR X:  xdirtag = -1  for  [-1 0  1]
 *         xdirtag =  1  for  [ 1 0 -1]
 *
 * FOR Y:  ydirtag = -1  for  [-1 0  1]'
 *         ydirtag =  1  for  [ 1 0 -1]'
 *
 * The resulting angle is in radians measured counterclockwise from the
 * xdirection. The angle points "up the gradient".
 *******************************************************************************/
void radian_direction(short int *delta_x, short int *delta_y, int rows,
                      int cols, float **dir_radians, int xdirtag, int ydirtag)
{
  int r, c, pos;
  float *dirim=NULL;
  double dx, dy;

  /****************************************************************************
   * Allocate an image to store the direction of the gradient.
   ****************************************************************************/
  if ((dirim = (float *) malloc(rows*cols* sizeof(float))) == NULL) {
    fprintf(stderr, "Error allocating the gradient direction image.\n");
    exit(1);
  }
  *dir_radians = dirim;

  for (r=0,pos=0; r<rows; r++) {
    for (c=0; c<cols; c++,pos++) {
      dx = (double)delta_x[pos];
      dy = (double)delta_y[pos];

      if (xdirtag == 1)
        dx = -dx;
      if (ydirtag == -1)
        dy = -dy;

      dirim[pos] = (float)angle_radians(dx, dy);
    }
  }
}

/*******************************************************************************
 * FUNCTION: angle_radians
 * PURPOSE: This procedure computes the angle of a vector with components x and
 * y. It returns this angle in radians with the answer being in the range
 * 0 <= angle <2*PI.
 *******************************************************************************/
double angle_radians(double x, double y)
{
  double xu, yu, ang;

  xu = fabs(x);
  yu = fabs(y);

  if ((xu == 0) && (yu == 0))
    return(0);

  ang = atan(yu/xu);

  if (x >= 0) {
    if (y >= 0)
      return(ang);
    else
      return(2*M_PI - ang);
  } else {
    if (y >= 0)
      return(M_PI - ang);
    else
      return(M_PI + ang);
  }
}

/*******************************************************************************
 * PROCEDURE: gaussian_smooth
 * PURPOSE: Blur an image with a gaussian filter.
 * NAME: Mike Heath
 * DATE: 2/15/96
 *******************************************************************************/
short int* gaussian_smooth(unsigned char *image, int rows, int cols, float sigma) {
    int r, c, rr, cc, windowsize, center;
    float *tempim, *kernel;

    // Kernel aanmaken 
    make_gaussian_kernel(sigma, &kernel, &windowsize);
    center = windowsize / 2;

    // Buffers alloceren
    tempim = (float *) malloc(rows * cols * sizeof(float));
    short int* smoothedim = (short int *) malloc(rows * cols * sizeof(short int));

    /****************************************************************************
     * Blur in de X - richting
     ****************************************************************************/
    #pragma omp parallel for private(c, cc)
    for (r = 0; r < rows; r++) {
        // 1. Linker rand: Scalar om correctheid te garanderen waar kernel buiten beeld valt
        for (c = 0; c < center; c++) {
            float dot = 0.0f, sum = 0.0f;
            for (cc = -center; cc <= center; cc++) {
                if (((c + cc) >= 0) && ((c + cc) < cols)) {
                    dot += (float)image[r * cols + (c + cc)] * kernel[center + cc];
                    sum += kernel[center + cc];
                }
            }
            tempim[r * cols + c] = dot / sum;
        }

        // NEON versnelling 
        float total_sum = 0.0f;
        for(int k=0; k<windowsize; k++) total_sum += kernel[k];
        float32x4_t v_inv_sum = vdupq_n_f32(1.0f / total_sum);

        for (; c <= cols - center - 4; c += 4) {
            float32x4_t v_dot = vdupq_n_f32(0.0f);
            for (cc = -center; cc <= center; cc++) {
                // Laad 4 pixels en converteer naar float
                uint8x8_t pix8 = vld1_u8(&image[r * cols + (c + cc)]);
                uint16x4_t pix16 = vget_low_u16(vmovl_u8(pix8));
                float32x4_t v_pix = vcvtq_f32_u32(vmovl_u16(pix16));
                
                v_dot = vmlaq_n_f32(v_dot, v_pix, kernel[center + cc]);
            }
            vst1q_f32(&tempim[r * cols + c], vmulq_f32(v_dot, v_inv_sum));
        }

        // Scalar
        for (; c < cols; c++) {
            float dot = 0.0f, sum = 0.0f;
            for (cc = -center; cc <= center; cc++) {
                if (((c + cc) >= 0) && ((c + cc) < cols)) {
                    dot += (float)image[r * cols + (c + cc)] * kernel[center + cc];
                    sum += kernel[center + cc];
                }
            }
            tempim[r * cols + c] = dot / sum;
        }
    }

    /****************************************************************************
     * Blur in de Y - richting
     ****************************************************************************/
    #pragma omp parallel for private(r, rr)
    for (c = 0; c <= cols - 4; c += 4) {
        for (r = 0; r < rows; r++) {
            float32x4_t v_dot = vdupq_n_f32(0.0f);
            float32x4_t v_sum = vdupq_n_f32(0.0f);

            for (rr = -center; rr <= center; rr++) {
                if (((r + rr) >= 0) && ((r + rr) < rows)) {
                    float32x4_t v_pix = vld1q_f32(&tempim[(r + rr) * cols + c]);
                    float32x4_t v_k = vdupq_n_f32(kernel[center + rr]);
                    v_dot = vmlaq_f32(v_dot, v_pix, v_k);
                    v_sum = vaddq_f32(v_sum, v_k);
                }
            }
            
            float32x4_t v_res = vdivq_f32(v_dot, v_sum);
            v_res = vmulq_n_f32(v_res, (float)BOOSTBLURFACTOR);
            
            // Gebruik 'vcvtaq_s32_f32' voor Round to Nearest, ties Away from zero
            int32x4_t v_int = vcvtaq_s32_f32(v_res); 
            int16x4_t v_short = vmovn_s32(v_int);
            vst1_s16(&smoothedim[r * cols + c], v_short);
        }
    }

    // Restant kolommen (als cols geen veelvoud van 4 is)
    for (int c_left = (cols / 4) * 4; c_left < cols; c_left++) {
        for (r = 0; r < rows; r++) {
            float sum = 0.0f, dot = 0.0f;
            for (rr = -center; rr <= center; rr++) {
                if (((r + rr) >= 0) && ((r + rr) < rows)) {
                    dot += tempim[(r + rr) * cols + c_left] * kernel[center + rr];
                    sum += kernel[center + rr];
                }
            }
            smoothedim[r * cols + c_left] = (short int)(dot * BOOSTBLURFACTOR / sum + 0.5f);
        }
    }

    free(tempim);
    free(kernel);
    return smoothedim;
}
/*******************************************************************************
 * PROCEDURE: make_gaussian_kernel
 * PURPOSE: Create a one dimensional gaussian kernel.
 * NAME: Mike Heath
 * DATE: 2/15/96
 *******************************************************************************/
void make_gaussian_kernel(float sigma, float **kernel, int *windowsize)
{
  int i, center;
  float x, fx, sum=0.0;

  *windowsize = 1 + 2 * ceil(2.5 * sigma);
  center = (*windowsize) / 2;

  if ((*kernel = (float *) malloc((*windowsize)* sizeof(float))) == NULL) {
    fprintf(stderr, "Error callocing the gaussian kernel array.\n");
    exit(1);
  }

  for (i=0; i<(*windowsize); i++) {
    x = (float)(i - center);
    fx = pow(2.71828, -0.5*x*x/(sigma*sigma)) / (sigma * sqrt(6.2831853));
    (*kernel)[i] = fx;
    sum += fx;
  }

  for (i=0; i<(*windowsize); i++)
    (*kernel)[i] /= sum;

  if (VERBOSE) {
    printf("The filter coefficients are:\n");
    for (i=0; i<(*windowsize); i++)
      printf("kernel[%d] = %f\n", i, (*kernel)[i]);
  }
}

