#include <boost/asio.hpp>
#include <iostream>
#include <string>
#include <thread>
#include <chrono>

using boost::asio::ip::tcp;

tcp::socket establish_server_connection(boost::asio::io_context& io) {
    tcp::socket socket(io);
    constexpr int max_connection_attempts = 6;
    constexpr int retry_interval_seconds = 5;
    int attempt_count = 0;

    while (attempt_count < max_connection_attempts) {
        try {
            tcp::endpoint server_endpoint(boost::asio::ip::make_address("127.0.0.1"), 12345);
            socket.connect(server_endpoint);
            std::cout << "[Status] Successfully connected to server\n";
            return socket;
        } catch (...) {
            std::cout << "[Retry] Connection attempt " << ++attempt_count 
                      << " of " << max_connection_attempts << " failed. "
                      << "Waiting " << retry_interval_seconds << " seconds...\n";
            std::this_thread::sleep_for(std::chrono::seconds(retry_interval_seconds));
        }
    }
    throw std::runtime_error("[Error] All connection attempts failed");
}

void run_sum_client(tcp::socket& socket) {
    while (true) {
        std::cout << "Enter numbers separated by spaces (or 'quit' to exit): ";
        std::string user_input;
        std::getline(std::cin, user_input);
        
        if (user_input == "quit") break;

        std::string request = "sum " + user_input + "\n";
        boost::asio::write(socket, boost::asio::buffer(request));

        boost::asio::streambuf response_buffer;
        boost::asio::read_until(socket, response_buffer, '\n');
        
        std::istream response_stream(&response_buffer);
        std::string server_response;
        std::getline(response_stream, server_response);
        std::cout << "Calculation result: " << server_response << "\n";
    }
}

int main() {
    try {
        boost::asio::io_context io_context;
        tcp::socket connection = establish_server_connection(io_context);
        
        run_sum_client(connection);
        
        std::cout << "[Status] Sum client terminated\n";
    } catch (std::exception& error) {
        std::cerr << "[Critical] " << error.what() << "\n";
        return 1;
    }
    return 0;
}