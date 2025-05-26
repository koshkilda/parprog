#include <iostream>
#include <thread>
#include <mutex>
#include <queue>
#include <vector>
#include <semaphore>
#include <chrono>
#include <atomic>

const int NUM_WORKERS    = 5;
const int MAX_EXECUTIONS = 20;

std::queue<int> queue1;
std::queue<int> queue2;

std::mutex mtx1, mtx2, mtx_stdout;
std::counting_semaphore<20> sem(0);

std::atomic<int>  executed_tasks{0};
std::atomic<bool> done{false};

void fill_queues() {
    for (int i = 1; i <= 10; i++) {
        queue1.push(i);
        sem.release();

        queue2.push(i + 10);
        sem.release();
    }
}

void worker(int id) {
    while (!done) {
        int task = -1;

        sem.acquire(); 
        { //сначала ждём задачу из queue1
            std::lock_guard<std::mutex> lock(mtx1);
            if (!queue1.empty()) {
                task = queue1.front();
                queue1.pop();
            }
        }

        if (task == -1) { //если не нашли в queue1, пробуем queue2
            {
                std::lock_guard<std::mutex> lock(mtx2);
                if (!queue2.empty()) {
                    task = queue2.front();
                    queue2.pop();
                }
            }
        }

        if (task != -1) {
            {
                std::lock_guard<std::mutex> lock(mtx_stdout);
                std::cout << "Работник " << id << " выполняет задачу " << task << std::endl;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
            executed_tasks++;

            if (executed_tasks >= MAX_EXECUTIONS) {
                done = true;

                for (int i = 0; i < NUM_WORKERS - 1; i++) {
                    sem.release();
                }
            }
        }
    }

    std::lock_guard<std::mutex> lock(mtx_stdout);
    std::cout << "Работник " << id << " завершил работу.\n";
}

int main() {
    fill_queues();

    std::vector<std::thread> workers;
    for (int i = 0; i < NUM_WORKERS; i++) {
        workers.emplace_back(worker, i + 1);
    }

    for (auto& t : workers) {
        t.join();
    }

    std::cout << "Все задачи обработаны. Программа завершена.\n";
    return 0;
}