CC ?= cc
SODIUM_PREFIX ?= /opt/anaconda3
CPPFLAGS ?= -I$(SODIUM_PREFIX)/include
CFLAGS ?= -O3 -std=c11 -Wall -Wextra -Wpedantic -Werror
SODIUM_LIB ?= $(SODIUM_PREFIX)/lib/libsodium.a
LDLIBS ?= -lpthread

.PHONY: all check clean

all: psi_backend

psi_backend: psi_backend.c psi_backend.h
	$(CC) $(CPPFLAGS) $(CFLAGS) psi_backend.c $(SODIUM_LIB) $(LDLIBS) -o $@

check: psi_backend
	./psi_backend selftest

clean:
	rm -f psi_backend
