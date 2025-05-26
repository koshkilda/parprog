#include <iostream>
#include <thread>
#include <queue>
#include <random>
#include <semaphore>
#include <chrono>
#include <vector>
#include <mutex>

struct Task {
    int id;
    int priority;
    int duration;
};

struct ComparePriority {
    bool operator()(const Task& t1, const Task& t2) {
        return t1.priority > t2.priority; //чем меньше значение, тем выше приоритет
    }
};

std::priority_queue<Task, std::vector<Task>, ComparePriority> task_queue;
std::counting_semaphore<4> machines(4);
std::mutex output_mutex, queue_mutex;

void process_task(Task task) {
    machines.acquire();

    {
        std::lock_guard<std::mutex> lock(output_mutex);
        std::cout << "Выполняется " << task.id << " задача (приоритет " << task.priority << ")...\n";
    }

    std::this_thread::sleep_for(std::chrono::seconds(task.duration));

    {
        std::lock_guard<std::mutex> lock(output_mutex);
        std::cout << "Завершена " << task.id << " задача за " << task.duration << "с\n";
    }

    machines.release();
}

void task_generator(int num_tasks) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dur_dist(1, 5);
    std::uniform_int_distribution<> prio_dist(1, 5);

    for (int i = 0; i < num_tasks; ++i) {
        Task task = {i + 1, prio_dist(gen), dur_dist(gen)};
        {
            std::lock_guard<std::mutex> lock(queue_mutex);
            task_queue.push(task);
        }

        {
            std::lock_guard<std::mutex> lock(output_mutex);
            std::cout << "Добавлена задача " << task.id << " с приоритетом " << task.priority << "\n";
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(300));
    }
}

void worker() {
    while (true) {
        Task task;
        {
            std::lock_guard<std::mutex> lock(queue_mutex);
            if (task_queue.empty()) break;
            task = task_queue.top();
            task_queue.pop();
        }

        process_task(task);
    }
}

int main() {
    std::thread producer(task_generator, 10);
    producer.join();

    std::vector<std::thread> workers;
    for (int i = 0; i < 4; ++i) {
        workers.emplace_back(worker);
    }

    for (auto& t : workers) {
        t.join();
    }

    return 0;
}