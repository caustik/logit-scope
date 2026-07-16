#pragma once

#include <string>

namespace logit_scope
{

class Engine;

class Server
{
  public:
    explicit Server(Engine& engine);
    ~Server();

    Server(const Server&) = delete;
    Server& operator=(const Server&) = delete;

    bool listen(const std::string& host, int port);
    void stop();

  private:
    class Impl;
    Impl* impl_;
};

} // namespace logit_scope
