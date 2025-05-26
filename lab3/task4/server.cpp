#include <boost/asio.hpp>
#include <iostream>
#include <string>
#include <locale>
#include <codecvt>
#include <sstream>
#include <vector>
#include <numeric>

using boost::asio::ip::tcp;

enum class command_type { Echo, Sum, Ping, Uknown };

command_type parse_command(const std::string& cmd) {
    if (cmd == "echo") return command_type::Echo;
    if (cmd == "sum") return command_type::Sum;
    if (cmd == "ping") return command_type::Ping;
    return command_type::Uknown;
}

std::string convert_to_uppercase_utf8(const std::string& input) {
    std::wstring_convert<std::codecvt_utf8<wchar_t>> converter;
    std::wstring wide_str = converter.from_bytes(input);
    std::locale locale("ru_RU.UTF-8");
    
    for (auto& ch : wide_str) {
        ch = std::toupper(ch, locale);
    }
    
    return converter.to_bytes(wide_str);
}

class session : public std::enable_shared_from_this<session> {
public:
    session(tcp::socket socket, boost::asio::io_context& io_context)
        : socket_(std::move(socket)), 
          io_context_(io_context), 
          timer_(io_context) {}

    void start() {
        log_connection();
        begin_read_operation();
    }

private:
    void log_connection() {
        std::cout << "[session] New connection from: " 
                  << socket_.remote_endpoint() << "\n";
    }

    void begin_read_operation() {
        auto self = shared_from_this();
        boost::asio::async_read_until(
            socket_, 
            buffer_, 
            '\n',
            [this, self](boost::system::error_code ec, std::size_t) {
                handle_read_complete(ec);
            });
    }

    void handle_read_complete(boost::system::error_code ec) {
        if (ec) {
            handle_network_error(ec);
            return;
        }
        
        process_client_request();
    }

    void handle_network_error(boost::system::error_code ec) {
        if (ec == boost::asio::error::eof || 
            ec == boost::asio::error::connection_reset) {
            std::cout << "[session] Client disconnected\n";
        } else {
            std::cerr << "[error] Network error: " << ec.message() << "\n";
        }
    }

    void process_client_request() {
        std::istream stream(&buffer_);
        std::string request_line;
        std::getline(stream, request_line);

        std::istringstream request_stream(request_line);
        std::string command_str;
        request_stream >> command_str;
        
        auto command = parse_command(command_str);

        switch (command) {
            case command_type::Echo: 
                handle_Echo_command(request_stream);
                break;
            case command_type::Sum: 
                handle_Sum_command(request_stream);
                break;
            case command_type::Ping: 
                handle_ping_command(request_stream);
                break;
            default:
                send_response("Unknown command\n");
                break;
        }

        if (command != command_type::Ping) {
            begin_read_operation();
        }
    }

    void handle_Echo_command(std::istringstream& request_stream) {
        std::string message;
        std::getline(request_stream, message);
        message = message.substr(1); // Remove leading space
        
        std::string response = convert_to_uppercase_utf8(message) + "\n";
        send_response(response);
    }

    void handle_Sum_command(std::istringstream& request_stream) {
        std::vector<int> numbers;
        int number;
        
        while (request_stream >> number) {
            numbers.push_back(number);
        }
        
        auto self = shared_from_this();
        boost::asio::post(io_context_, [this, self, numbers]() {
            int total = std::accumulate(numbers.begin(), numbers.end(), 0);
            send_response("Sum: " + std::to_string(total) + "\n");
        });
    }

    void handle_ping_command(std::istringstream& request_stream) {
        int delay_seconds = 3;
        request_stream >> delay_seconds;
        
        auto self = shared_from_this();
        timer_.expires_after(std::chrono::seconds(delay_seconds));
        timer_.async_wait([this, self](boost::system::error_code) {
            send_response("pong\n");
        });
    }

    void send_response(const std::string& response) {
        auto self = shared_from_this();
        boost::asio::async_write(
            socket_, 
            boost::asio::buffer(response),
            [this, self, response](boost::system::error_code ec, std::size_t) {
                handle_write_complete(ec, response);
            });
    }

    void handle_write_complete(boost::system::error_code ec, 
                             const std::string& response) {
        if (!ec) {
            std::cout << "[send] " << response;
            if (response != "pong\n") {
                begin_read_operation();
            }
        }
    }

    tcp::socket socket_;
    boost::asio::io_context& io_context_;
    boost::asio::streambuf buffer_;
    boost::asio::steady_timer timer_;
};

class tcp_server {
public:
    tcp_server(boost::asio::io_context& io_context, short port)
        : acceptor_(io_context, tcp::endpoint(tcp::v4(), port)), 
          io_context_(io_context) {
        start_accepting_connections();
    }

private:
    void start_accepting_connections() {
        acceptor_.async_accept(
            [this](boost::system::error_code ec, tcp::socket socket) {
                handle_new_connection(ec, std::move(socket));
            });
    }

    void handle_new_connection(boost::system::error_code ec, tcp::socket socket) {
        if (!ec) {
            std::make_shared<session>(std::move(socket), io_context_)->start();
        }
        start_accepting_connections();
    }

    tcp::acceptor acceptor_;
    boost::asio::io_context& io_context_;
};

int main() {
    try {
        std::setlocale(LC_ALL, "");
        boost::asio::io_context io_context;
        tcp_server server(io_context, 12345);
        
        std::cout << "Server started on 127.0.0.1:12345\n";
        io_context.run();
    } catch (std::exception& e) {
        std::cerr << "Server error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}