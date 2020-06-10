all: main
 
main: main.cu labeling.cu
		nvcc -arch -D_FORCE_INLINES sm_30 -Xcompiler "-O2 -fopenmp" -c labeling.cu -o labeling.o
		nvcc -arch -D_FORCE_INLINES sm_30 -Xcompiler "-O2 -fopenmp" main.cu labeling.o -o main