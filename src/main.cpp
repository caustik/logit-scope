#include "logit_scope/engine.h"
#include "server.h"

#include <chrono>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <string>
#include <thread>

#if defined(_WIN32)
#include <windows.h>
#include <shellapi.h>
#endif

namespace
{

struct Arguments
{
    logit_scope::EngineConfig engine;
    std::string host = "127.0.0.1";
    int port = 8080;
    bool open_browser = true;
};

void read_model_environment(Arguments& arguments)
{
#if defined(_WIN32)
    char* model = nullptr;
    std::size_t length = 0;
    if (_dupenv_s(&model, &length, "LOGIT_SCOPE_MODEL") == 0 && model != nullptr)
    {
        arguments.engine.model_path = model;
        std::free(model);
    }
#else
    if (const auto* model = std::getenv("LOGIT_SCOPE_MODEL")) arguments.engine.model_path = model;
#endif
}

void print_usage()
{
    std::cout << "Logit Scope 0.1\n\n"
                 "Usage: logit-scope --model <model.gguf> [options]\n\n"
                 "Options:\n"
                 "  --model <path>        GGUF model path (or LOGIT_SCOPE_MODEL)\n"
                 "  --host <address>      Bind address (default: 127.0.0.1)\n"
                 "  --port <number>       HTTP port (default: 8080)\n"
                 "  --ctx-size <tokens>   Context capacity (default: 4096)\n"
                 "  --max-tokens <count>  Response ceiling (default: 1024)\n"
                 "  --threads <count>     Inference threads (default: hardware minus "
                 "two)\n"
                 "  --gpu-layers <count>  Layers offloaded to the GPU (default: 0)\n"
                 "  --no-browser          Do not open the local UI automatically\n"
                 "  --help                Show this help\n";
}

int parse_integer(const char* option, const std::string& value, int minimum, int maximum)
{
    std::size_t consumed = 0;
    const auto parsed = std::stoi(value, &consumed);
    if (consumed != value.size() || parsed < minimum || parsed > maximum)
        throw std::runtime_error(std::string("Invalid value for ") + option + ": " + value);
    return parsed;
}

bool parse_arguments(int argc, char** argv, Arguments& arguments)
{
    read_model_environment(arguments);

    for (int index = 1; index < argc; ++index)
    {
        const std::string option = argv[index];
        const auto value = [&]() -> std::string
        {
            if (index + 1 >= argc) throw std::runtime_error("Missing value for " + option);
            return argv[++index];
        };

        if (option == "--help" || option == "-h")
        {
            print_usage();
            return false;
        }
        if (option == "--model")
            arguments.engine.model_path = value();
        else if (option == "--host")
            arguments.host = value();
        else if (option == "--port")
            arguments.port = parse_integer("--port", value(), 1, 65535);
        else if (option == "--ctx-size")
            arguments.engine.context_size = parse_integer("--ctx-size", value(), 512, 1048576);
        else if (option == "--max-tokens")
            arguments.engine.maximum_response_tokens = parse_integer("--max-tokens", value(), 1, 1048576);
        else if (option == "--threads")
            arguments.engine.threads = parse_integer("--threads", value(), 1, 1024);
        else if (option == "--gpu-layers")
            arguments.engine.gpu_layers = parse_integer("--gpu-layers", value(), 0, 10000);
        else if (option == "--no-browser")
            arguments.open_browser = false;
        else
            throw std::runtime_error("Unknown option: " + option);
    }

    if (arguments.engine.model_path.empty()) throw std::runtime_error("No model was provided. Use --model or LOGIT_SCOPE_MODEL.");
    return true;
}

void open_browser(int port)
{
    const auto url = "http://127.0.0.1:" + std::to_string(port) + "/";
#if defined(_WIN32)
    ShellExecuteA(nullptr, "open", url.c_str(), nullptr, nullptr, SW_SHOWNORMAL);
#elif defined(__APPLE__)
    const auto command = "open \"" + url + "\"";
    std::system(command.c_str());
#else
    const auto command = "xdg-open \"" + url + "\" >/dev/null 2>&1";
    std::system(command.c_str());
#endif
}

} // namespace

int main(int argc, char** argv)
{
    try
    {
        Arguments arguments;
        if (!parse_arguments(argc, argv, arguments)) return 0;

        logit_scope::Engine engine(arguments.engine);
        engine.start();
        logit_scope::Server server(engine);

        std::cout << "Logit Scope is running at http://" << arguments.host << ':' << arguments.port << "/\n"
                  << "Model: " << arguments.engine.model_path << "\n"
                  << "Press Ctrl+C to stop.\n";

        std::thread browser_thread;
        if (arguments.open_browser)
        {
            browser_thread = std::thread(
                [port = arguments.port]
                {
                    std::this_thread::sleep_for(std::chrono::milliseconds(500));
                    open_browser(port);
                });
        }

        const auto listened = server.listen(arguments.host, arguments.port);
        if (browser_thread.joinable()) browser_thread.join();
        engine.stop();
        if (!listened)
        {
            std::cerr << "Unable to listen on " << arguments.host << ':' << arguments.port << "\n";
            return 1;
        }
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << "Error: " << error.what() << "\n\n";
        print_usage();
        return 1;
    }
}
