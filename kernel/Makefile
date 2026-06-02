.PHONY: all harmonic_demo schwarzschild_demo schwarzschild3d_demo phase2_batch_demo clean

CC ?= cc
CFLAGS ?= -std=c99 -Wall -Wextra -O2 -Iinclude

PHASE2_SRCS = src/bh_rt_rk4.c src/bh_rt_schwarzschild_phase2.c
PHASE2_BATCH_SRCS = $(PHASE2_SRCS) src/bh_rt_schwarzschild_phase2_batch.c

all: harmonic_demo schwarzschild_demo schwarzschild3d_demo phase2_batch_demo

harmonic_demo: src/bh_rt_rk4.c src/demo_harmonic.c include/bh_rt_rk4.h
	$(CC) $(CFLAGS) -o harmonic_demo src/bh_rt_rk4.c src/demo_harmonic.c -lm

schwarzschild_demo: src/bh_rt_rk4.c src/bh_rt_schwarzschild_u.c src/demo_schwarzschild_u.c \
		include/bh_rt_rk4.h include/bh_rt_schwarzschild_u.h
	$(CC) $(CFLAGS) -o schwarzschild_demo src/bh_rt_rk4.c src/bh_rt_schwarzschild_u.c \
		src/demo_schwarzschild_u.c -lm

schwarzschild3d_demo: src/bh_rt_rk4.c src/bh_rt_schwarzschild_3d.c src/demo_schwarzschild_3d.c \
		include/bh_rt_rk4.h include/bh_rt_status.h include/bh_rt_schwarzschild_3d.h
	$(CC) $(CFLAGS) -o schwarzschild3d_demo src/bh_rt_rk4.c src/bh_rt_schwarzschild_3d.c \
		src/demo_schwarzschild_3d.c -lm

phase2_batch_demo: $(PHASE2_BATCH_SRCS) src/demo_schwarzschild_phase2_batch.c \
		include/bh_rt_rk4.h include/bh_rt_schwarzschild_phase2.h \
		include/bh_rt_schwarzschild_phase2_batch.h \
		include/bh_rt_schwarzschild_phase2_internal.h \
		include/bh_rt_status.h
	$(CC) $(CFLAGS) -o phase2_batch_demo \
		$(PHASE2_BATCH_SRCS) src/demo_schwarzschild_phase2_batch.c -lm

clean:
	rm -f harmonic_demo harmonic_demo.exe \
	      schwarzschild_demo schwarzschild_demo.exe \
	      schwarzschild3d_demo schwarzschild3d_demo.exe \
	      phase2_batch_demo phase2_batch_demo.exe
