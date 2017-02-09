CC := gcc
CFLAGS	:= -O3
#CFLAGS := -g
#CFLAGS := -g -lm -Wall -pthread
#CFLAGS := -msse -O3 -fPIC -fstrict-aliasing -fomit-frame-pointer -Wall -pthread
CLIBS := -lm

all: acupa acupam

acupa:	svmocas.c lib_svmlight_format.c evaluate_testing.c evaluate_multiclass.c sparse_mat.c ocas_helper.c ocas_helper.h libocas.h sparse_mat.h libocas.c features_double.h features_double.c version.h 
		$(CC) $(CFLAGS) -o $@ svmocas.c lib_svmlight_format.c evaluate_testing.c  evaluate_multiclass.c sparse_mat.c ocas_helper.c features_double.c libocas.c libqp_splx.c $(CLIBS)

acupam:	msvmocas.c lib_svmlight_format.c evaluate_testing.c evaluate_multiclass.c sparse_mat.c ocas_helper.c ocas_helper.h libocas.h sparse_mat.h libocas.c features_double.h features_double.c version.h 
		$(CC) $(CFLAGS) -o $@ msvmocas.c lib_svmlight_format.c evaluate_testing.c  evaluate_multiclass.c sparse_mat.c ocas_helper.c features_double.c libocas.c libqp_splx.c $(CLIBS)

clean: 
		rm -f *~ acupa acupam
