SOURCE	:= saxpy.c
CC	:= gcc
TARGET	:= saxpy
LINK	:= -O3 -Wall -std=c99 -lpthread -lm

all: $(TARGET)

$(TARGET): $(SOURCE)
	$(CC) -o $(TARGET) $(SOURCE) $(LINK)

clean:
	rm -f $(TARGET)

