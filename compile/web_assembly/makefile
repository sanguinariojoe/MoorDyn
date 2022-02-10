# makefile for MoorDyn compiled as libmoordyn.so.2.0.0, and linked as
# libmoordyn.so (so it can be linked with -lmoordyn)
#
# The following targets are provided:
#
# all: Builds the library libmoordyn.so.2.0.0, and the symbolic link
#      libmoordyn.so -> libmoordyn.so.2.0.0
#
# test: Build libmoordyn.so, if not yet built, and then the tests:
#        - test_minimal
#        - test_quasi_static_chain
#
# debug: Build both targets, "all" and "test", with debug information


LFLAGS = -DEMSCRIPT -DMoorDyn_EXPORTS
CFLAGS = -c -O3 -DEMSCRIPT -DMoorDyn_EXPORTS
CPPFLAGS = $(CFLAGS)

all: moordyn.js

moordyn.js: MoorDyn.o Line.o Connection.o Body.o Rod.o Waves.o Misc.o kiss_fft.o kiss_fftr.o
	emcc $(LFLAGS) -o $@ MoorDyn.o Line.o Connection.o Body.o Rod.o Waves.o Misc.o kiss_fft.o kiss_fftr.o

MoorDyn.o: ../../source/MoorDyn.cpp ../../source/MoorDyn.h ../../source/MoorDynAPI.h
	emcc $(CPPFLAGS) ../../source/MoorDyn.cpp -o $@

kiss_fft.o: ../../source/kiss_fft.c ../../source/kiss_fft.h
	emcc $(CFLAGS) ../../source/kiss_fft.c -o $@

kiss_fftr.o: ../../source/kiss_fftr.c ../../source/kiss_fftr.h
	emcc $(CFLAGS) ../../source/kiss_fftr.c -o $@

Line.o: ../../source/Line.cpp ../../source/Line.h ../../source/QSlines.h
	emcc $(CPPFLAGS) ../../source/Line.cpp -o $@

Connection.o: ../../source/Connection.cpp ../../source/Connection.h
	emcc $(CPPFLAGS) ../../source/Connection.cpp -o $@

Body.o: ../../source/Body.cpp ../../source/Body.h
	emcc $(CPPFLAGS) ../../source/Body.cpp -o $@

Rod.o: ../../source/Rod.cpp ../../source/Rod.h
	emcc $(CPPFLAGS) ../../source/Rod.cpp -o $@

Waves.o: ../../source/Waves.cpp ../../source/Waves.h
	emcc $(CPPFLAGS) ../../source/Waves.cpp -o $@

Misc.o: ../../source/Misc.cpp ../../source/Misc.h ../../source/MoorDynAPI.h
	emcc $(CPPFLAGS) ../../source/Misc.cpp -o $@

clean:
	rm -f *.o *.wasm
