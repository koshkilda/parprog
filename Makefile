CC = g++
TARGET = task2_mutex

CFLAGS = -Wall
# CFLAGS = -Wshadow -Winit-self -Wredundant-decls -Wcast-align -Wundef -Wfloat-equal -Winline -Wunreachable-code -Wmissing-declarations \
#          -Wmissing-include-dirs -Wswitch-enum -Wswitch-default -Weffc++ -Wmain -Wextra -Wall -g -pipe -fexceptions -Wcast-qual	      \
#          -Wconversion -Wctor-dtor-privacy -Wempty-body -Wformat-security -Wformat=2 -Wignored-qualifiers -Wlogical-op                 \
#          -Wmissing-field-initializers -Wnon-virtual-dtor -Woverloaded-virtual -Wpointer-arith -Wsign-promo -Wstack-usage=8192         \
#          -Wstrict-aliasing -Wstrict-null-sentinel -Wtype-limits -Wwrite-strings -D_DEBUG -D_EJUDGE_CLIENT_SIDE						  

PREF_SRC = ./src/
PREF_OBJ = ./obj/

SRC = $(PREF_SRC)$(TARGET).cpp
OBJ = $(PREF_OBJ)$(TARGET).o

all : $(TARGET)

$(TARGET) : $(OBJ)
	$(CC) $< $(CFLAGS) -lboost_thread -lboost_system -o $(TARGET)

$(OBJ) : $(SRC)
	$(CC) $(CFLAGS) -c $< -o $@

clean :
	rm $(TARGET) $(PREF_OBJ)*.o