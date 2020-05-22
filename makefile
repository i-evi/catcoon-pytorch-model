cc = gcc

ifeq ($(OS),Windows_NT)
	RM = del
	CFLAG = -shared
else
	RM = rm
	CFLAG = -shared -fPIC
endif

bbuffer.so: bbuffer.c
	$(cc) -o $(@) bbuffer.c $(CFLAG)

clean:
	$(RM) bbuffer.so