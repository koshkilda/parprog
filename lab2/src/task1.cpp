#include <iostream>
#include <thread>
#include <vector>
#include <semaphore>
#include <random>
#include <chrono>
#include <mutex>

std::counting_semaphore<5> turnstiles(3); 
std::mutex output_mutex;

void employee(int id, bool high_priority) {
    std::this_thread::sleep_for(std::chrono::milliseconds(rand() % 1000));

    if (high_priority) {
        std::lock_guard<std::mutex> lock(output_mutex);
        std::cout << "Высокоприоритетный сотрудники " << id << " пропускаются вне очереди.\n";
        return;
    }

    turnstiles.acquire(); //попытка войти через турникет
    {
        std::lock_guard<std::mutex> lock(output_mutex);
        std::cout << "Сотрудник " << id << " прошел через турникет.\n";
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(500)); 
    turnstiles.release();
}

int main() {
    std::vector<std::thread> employees;
    int people_flow = 20;

    if (people_flow > 15) {
        turnstiles.release(); // открытие 4го доп. турникета
        turnstiles.release(); // открытие 5го доп. турникета
    }

    for (int i = 0; i < people_flow; ++i) {
        bool high_priority = (i % 5 == 0);
        employees.emplace_back(employee, i, high_priority);
    }

    for (auto& e : employees) {
        e.join();
    }

    return 0;
}