CINC          :=
CLIBS         := -lcufft
CFLAGS        := -O3 -std=c++14

all: 1d_r2c_example

1d_r2c_exampleC: main_cufft.cpp
	nvcc $(CFLAGS) $(CINC) main_cufft.cpp -o 1d_r2c_example $(CLIBS)

clean:
	rm -f 1d_r2c_exampleC

.PHONY: clean all