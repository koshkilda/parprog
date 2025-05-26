#include <boost/asio.hpp>
#include <iostream>
#include <string>
#include <thread>
#include <chrono>

using boost::asio::ip::tcp;

tcp::socket establish_connection(boost::asio::io_context& io) {
    tcp::socket socket(io);
    constexpr int max_retries = 6;
    constexpr int retry_interval = 5;
    
    for (int attempt = 1; attempt <= max_retries; ++attempt) {
        try {
            tcp::endpoint endpoint(boost::asio::ip::make_address("127.0.0.1"), 12345);
            socket.connect(endpoint);
            std::cout << "[Success] Connection established\n";
            return socket;
        } catch (...) {
            std::cout << "[Waiting] Connection attempt " << attempt << " of " 
                      << max_retries << " failed. Retrying...\n";
            std::this_thread::sleep_for(std::chrono::seconds(retry_interval));
        }
    }
    throw std::runtime_error("[Fatal Error] Connection failed");
}

void handle_communication(tcp::socket& socket) {
    while (true) {
        std::cout << "Enter message (type 'quit' to exit): ";
        std::string input;
        std::getline(std::cin, input);
        
        if (input == "quit") break;

        std::string request = "echo " + input + "\n";
        boost::asio::write(socket, boost::asio::buffer(request));

        boost::asio::streambuf response_buf;
        boost::asio::read_until(socket, response_buf, '\n');
        
        std::istream response_stream(&response_buf);
        std::string reply;
        std::getline(response_stream, reply);
        std::cout << "Server response: " << reply << "\n";
    }
}

int main() {
    try {
        boost::asio::io_context io;
        tcp::socket socket = establish_connection(io);
        
        handle_communication(socket);
        
        std::cout << "[Shutdown] Client terminating\n";
    } catch (std::exception& e) {
        std::cerr << "Error occurred: " << e.what() << "\n";
        return 1;
    }
    return 0;
}