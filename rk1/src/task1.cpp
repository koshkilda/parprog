#include <iostream>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>

const int NUM_STAGES = 4;

std::mutex              mtx;
std::condition_variable cv[NUM_STAGES];
std::atomic<int>        current_stage(0); 

void processStage(int stage) {
    std::unique_lock<std::mutex> lock(mtx);

    cv[stage].wait(lock, [stage]() { //ждём, пока не завершится предыдущий этап
        return current_stage == stage;
    });

    std::cout << "Этап " << stage + 1 << " выполняется\n";
    std::this_thread::sleep_for(std::chrono::seconds(1));  // имитация работы

    current_stage++;
    std::cout << "Этап " << stage + 1 << " завершен\n";

    // Уведомляем следующий этап
    if (stage + 1 < NUM_STAGES) {
        cv[stage + 1].notify_one();
    }
}

int main() {
    std::thread stages[NUM_STAGES];

    for (int i = 0; i < NUM_STAGES; i++) {
        stages[i] = std::thread(processStage, i);
    }

    // Запускаем первый этап
    cv[0].notify_one();

    for (int i = 0; i < NUM_STAGES; i++) {
        stages[i].join();
    }

    return 0;
}