.PHONY: harmonic_demo clean

CC ?= cc
CFLAGS ?= -std=c99 -Wall -Wextra -O2 -Iinclude

harmonic_demo: src/bh_rt_rk4.c src/demo_harmonic.c include/bh_rt_rk4.h
	$(CC) $(CFLAGS) -o harmonic_demo src/bh_rt_rk4.c src/demo_harmonic.c -lm

clean:
	rm -f harmonic_demo harmonic_demo.exe
