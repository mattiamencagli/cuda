NVCC = mpicxx
CFLAGS = -lm -acc -gpu=rdc,managed -Minfo=accel -Minline=1000 -O3
LIB = -lm -acc -gpu=rdc,managed

all: 
	${NVCC} ${LIB} matmul_multiN.cu -o matvec

clean:
	-rm newfloat*
