#include <stdio.h>
#include <stdlib.h>


int read_pgm_image(char *infilename, unsigned char **image, int *rows,
                   int *cols);


int main(int argc, char *argv[])
{
  if (argc != 3) {
    fprintf(stderr, "USAGE: %s <image1> <image2>\n", argv[0]);
    fprintf(stderr, "  <image1>:    The first PGM image to compare.\n");
    fprintf(stderr, "  <image2>:    The second PGM image to compare.\n");
    exit(1);
  }

  char *filename1 = argv[1];
  char *filename2 = argv[2];

  unsigned char *image1, *image2;
  int rows1, cols1, rows2, cols2;

  if (read_pgm_image(filename1, &image1, &rows1, &cols1) == 0) {
    fprintf(stderr, "image_compare: error reading the input image, %s.\n",
            filename1);
    exit(1);
  }

  if (read_pgm_image(filename2, &image2, &rows2, &cols2) == 0) {
    fprintf(stderr, "image_compare: error reading the input image, %s.\n",
            filename1);
    exit(1);
  }

  // First check: rows and columns must be equal
  if (rows1 != rows2 || cols1 != cols2) {
    // Oh, no use bothering the real compare, they are not equal!
    printf("The number of rows and/or columns don't even match bro!\n");
    printf("Too bad ..  maybe up your game?\n");
    printf("Exiting program, doesn't make sense to check the real stuff.\n");
    return 0;
  }

 // Phew, rows and columns do match.
  int errors = 0;
  for (int i = 0; i < rows1; i++) {
    for (int j = 0; j < cols1; j++) {
      // Bereken de huidige index
      int index = i * cols1 + j;

      if (image1[index] != image2[index]) {
        // Highlight de fout: print rij, kolom en de waarden
        printf("Whoops! Error at [Row: %d, Col: %d] -> Baseline: %u, Speedup: %u (Diff: %d)\n", 
                i, j, image1[index], image2[index], (int)image1[index] - (int)image2[index]);
        errors++;
      }
    }
  }

  // The verdict
  if (errors > 0) {
    printf("\nYou have errors bro, not good.\n");
    printf("The number of mistakes you made is %d.\n", errors);
    printf("Percentage error: %.6f%%\n", (double)errors / (rows1 * cols1) * 100);
    printf("Exiting program. Better luck next time!\n");
  } else {
    printf("Good job, didn't mess up. :-)\n");
    printf("The images are equal!\n");
    printf("Exiting program. Keep up the good work.\n");
  }
}