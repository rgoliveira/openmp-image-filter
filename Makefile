CC=/home/rgoliveira/intel/bin/icc
#CCFLAGS=-w3 -Wall -Werror -debug all -fopenmp
CCFLAGS=-fopenmp -debug none

all: dirs filter

dirs:
	mkdir -p ./build

filter:
	$(CC) -static src/filter.c -o ./build/filter $(CCFLAGS)

clean:
	rm -rf ./build
