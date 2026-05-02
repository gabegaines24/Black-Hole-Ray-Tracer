.PHONY: all harmonic_demo schwarzschild_demo clean

CC ?= cc
CFLAGS ?= -std=c99 -Wall -Wextra -O2 -Iinclude

all: harmonic_demo schwarzschild_demo

harmonic_demo: src/bh_rt_rk4.c src/demo_harmonic.c include/bh_rt_rk4.h
	$(CC) $(CFLAGS) -o harmonic_demo src/bh_rt_rk4.c src/demo_harmonic.c -lm

schwarzschild_demo: src/bh_rt_rk4.c src/bh_rt_schwarzschild_u.c src/demo_schwarzschild_u.c \
		include/bh_rt_rk4.h include/bh_rt_schwarzschild_u.h
	$(CC) $(CFLAGS) -o schwarzschild_demo src/bh_rt_rk4.c src/bh_rt_schwarzschild_u.c \
		src/demo_schwarzschild_u.c -lm

clean:
	rm -f harmonic_demo harmonic_demo.exe schwarzschild_demo schwarzschild_demo.exe
