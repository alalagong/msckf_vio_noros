#ifndef TIME_DEBUGGER_H
#define TIME_DEBUGGER_H

#include <chrono>
#include <iostream>
#include <string>

class TimeDebugger
{
public:
    TimeDebugger(std::string info) :
        start_time(std::chrono::system_clock::now()),
        message(info)
    {}

    ~TimeDebugger()
    {
        std::chrono::system_clock::time_point cur = std::chrono::system_clock::now();
        std::chrono::milliseconds time_elapse =
            std::chrono::duration_cast<std::chrono::milliseconds>(cur - start_time);
        std::cout << message << time_elapse.count() << " ms" << std::endl;
    }

private:
    std::chrono::system_clock::time_point start_time;
    std::string message;
};

#endif
