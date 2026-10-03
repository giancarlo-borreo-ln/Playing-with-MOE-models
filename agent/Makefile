# Minimal coding agent: a zero-dependency C binary + a Docker sandbox image.

CC      ?= cc
CFLAGS  ?= -O2 -std=c11 -Wall -Wextra -Wpedantic
IMAGE   ?= agent-sandbox:latest

.PHONY: all sandbox run clean

all: agent

agent: agent.c
	$(CC) $(CFLAGS) -o $@ $<

# Build the toolchain image the generated code runs in (needs network once).
sandbox:
	docker build -t $(IMAGE) sandbox

# Example task; override with: make run TASK="..."
TASK ?= write a C program that prints the first 20 prime numbers

run: agent sandbox
	./agent "$(TASK)"

clean:
	rm -f agent
