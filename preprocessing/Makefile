CC := clang
CFLAGS := -fobjc-arc -fmodules

CFILES := main.m correct.m

.PHONY: all clean

all: preprocess

preprocess: $(CFILES)
	$(CC) $(CFLAGS) $(CFILES) -o preprocess

clean:
	rm -rf preprocess
