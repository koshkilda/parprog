# Компилятор и флаги
CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -I.
LDFLAGS = -lboost_system -lpthread

# Цели
TARGETS = server ping_client sum_client echo_client

# Исходные файлы
SOURCES = \
    server.cpp \
    ping_client.cpp \
    sum_client.cpp \
	echo_client.cpp

# Объектные файлы
OBJECTS = $(SOURCES:.cpp=.o)

# Правила по умолчанию
all: $(TARGETS)

server: server.o
	$(CXX) $^ -o $@ $(LDFLAGS)

ping_client: ping_client.o
	$(CXX) $^ -o $@ $(LDFLAGS)

sum_client: sum_client.o
	$(CXX) $^ -o $@ $(LDFLAGS)

echo_client: echo_client.o
	$(CXX) $^ -o $@ $(LDFLAGS)

# Правило для объектных файлов
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Очистка
clean:
	rm -f $(TARGETS) $(OBJECTS)

# Фейковые цели
.PHONY: all clean