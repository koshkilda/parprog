#include <boost/asio.hpp>
#include <iostream>
#include <string>
#include <thread>
#include <chrono>

using boost::asio::ip::tcp;

tcp::socket connect_with_retries(boost::asio::io_context& io) {
    tcp::socket socket(io);
    constexpr int max_attempts = 6;
    constexpr int retry_delay_sec = 5;
    int attempts = 0;
    
    while (attempts < max_attempts) {
        try {
            tcp::endpoint endpoint(boost::asio::ip::make_address("127.0.0.1"), 12345);
            socket.connect(endpoint);
            std::cout << "[Success] Connected to server\n";
            return socket;
        } catch (...) {
            std::cout << "[Waiting] Attempt " << ++attempts << "/" << max_attempts 
                      << " failed. Retrying in " << retry_delay_sec << " seconds...\n";
            std::this_thread::sleep_for(std::chrono::seconds(retry_delay_sec));
        }
    }
    throw std::runtime_error("[Error] Failed to connect to server");
}

void run_ping_client(tcp::socket& socket) {
    while (true) {
        std::cout << "Enter delay in seconds (or 'quit' to exit): ";
        std::string input;
        std::getline(std::cin, input);
        
        if (input == "quit") break;

        std::string request = "ping " + input + "\n";
        boost::asio::write(socket, boost::asio::buffer(request));

        boost::asio::streambuf response_buffer;
        boost::asio::read_until(socket, response_buffer, '\n');
        
        std::istream response_stream(&response_buffer);
        std::string response;
        std::getline(response_stream, response);
        std::cout << "Server reply: " << response << "\n";
    }
}

int main() {
    try {
        boost::asio::io_context io;
        tcp::socket socket = connect_with_retries(io);
        
        run_ping_client(socket);
        
        std::cout << "[Status] Ping client terminated\n";
    } catch (std::exception& e) {
        std::cerr << "[Error] " << e.what() << "\n";
        return 1;
    }
    return 0;
}