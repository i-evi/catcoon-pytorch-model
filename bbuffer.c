#ifdef __cplusplus
	extern "C" {
#endif

#include <stdio.h>
#include <string.h>

FILE *global_file_pointer = NULL;
char current_work_file[256] = {0};

void init(const char *filename)
{
	fprintf(stderr, "Open file: %s\n", filename);
	if (!global_file_pointer) {
		global_file_pointer = fopen(filename, "wb+");
		sprintf(current_work_file, "%s\n", filename);
	} else {
		fprintf(stderr,
				"Init failed: current work file %s\n", 
			current_work_file);
	}
	return;
}

void close()
{
	if (global_file_pointer) {
		fprintf(stderr, "Close file: %s\n", current_work_file);
		*current_work_file = '\0';
		fclose(global_file_pointer);
		global_file_pointer = NULL;
	}
	return;
}

void flush()
{
	if (global_file_pointer)
		fflush(global_file_pointer);
	else
		fprintf(stderr, "Must open file fisrt");
	return;
}

#define float32 float
#define float64 double

#define WRITE_DT_IMPLEMENTATION(dtype) \
int write_ ## dtype(dtype d)                               \
{                                                          \
	if (!global_file_pointer) {                        \
		fprintf(stderr, "Must open file fisrt");   \
		return -1;                                 \
	}                                                  \
	fwrite(&d, sizeof(dtype), 1, global_file_pointer); \
	return 0;                                          \
}

WRITE_DT_IMPLEMENTATION(float32)
WRITE_DT_IMPLEMENTATION(float64)

#ifdef __cplusplus
	}
#endif