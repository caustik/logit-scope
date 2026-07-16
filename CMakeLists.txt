cmake_minimum_required(VERSION 3.21)

project(logit-scope VERSION 0.1.0 LANGUAGES C CXX)

include(GNUInstallDirs)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(LOGIT_SCOPE_BUILD_TESTS "Build Logit Scope tests" ON)

if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/third_party/llama.cpp/CMakeLists.txt")
    message(FATAL_ERROR "llama.cpp is missing. Clone with --recurse-submodules or run: git submodule update --init --recursive")
endif()

set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_COMMON OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)

add_subdirectory(third_party/llama.cpp EXCLUDE_FROM_ALL)

find_package(Threads REQUIRED)

add_library(logit_scope_httplib STATIC
    third_party/llama.cpp/vendor/cpp-httplib/httplib.cpp
)
target_include_directories(logit_scope_httplib PUBLIC third_party/llama.cpp/vendor/cpp-httplib)
target_link_libraries(logit_scope_httplib PRIVATE Threads::Threads)
target_compile_features(logit_scope_httplib PRIVATE cxx_std_17)

if(MSVC)
    target_compile_options(logit_scope_httplib PRIVATE /w)
else()
    target_compile_options(logit_scope_httplib PRIVATE -w)
endif()

add_library(logit_scope_core
    src/rank_profile.cpp
)

target_include_directories(logit_scope_core PUBLIC include)

set(LOGIT_SCOPE_GENERATED_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated")
file(MAKE_DIRECTORY "${LOGIT_SCOPE_GENERATED_DIR}")

set(LOGIT_SCOPE_WEB_ASSETS "${LOGIT_SCOPE_GENERATED_DIR}/web_assets.cpp")
add_custom_command(
    OUTPUT "${LOGIT_SCOPE_WEB_ASSETS}"
    COMMAND "${CMAKE_COMMAND}"
        -DLOGIT_SCOPE_WEB_DIR=${CMAKE_CURRENT_SOURCE_DIR}/web
        -DLOGIT_SCOPE_TEMPLATE=${CMAKE_CURRENT_SOURCE_DIR}/cmake/web_assets.cpp.in
        -DLOGIT_SCOPE_OUTPUT=${LOGIT_SCOPE_WEB_ASSETS}
        -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/EmbedWeb.cmake"
    DEPENDS
        web/index.html
        web/app.js
        web/style.css
        cmake/web_assets.cpp.in
        cmake/EmbedWeb.cmake
    VERBATIM
)

add_executable(logit-scope
    src/main.cpp
    src/engine.cpp
    src/server.cpp
    "${LOGIT_SCOPE_WEB_ASSETS}"
)

target_include_directories(logit-scope PRIVATE
    include
    src
    third_party/llama.cpp/include
    third_party/llama.cpp/ggml/include
    third_party/llama.cpp/vendor/cpp-httplib
    third_party/llama.cpp/vendor
)

target_link_libraries(logit-scope PRIVATE logit_scope_core logit_scope_httplib llama Threads::Threads)

if(WIN32)
    target_link_libraries(logit-scope PRIVATE shell32)
endif()

if(MSVC)
    target_compile_options(logit_scope_core PRIVATE /W4 /permissive-)
    target_compile_options(logit-scope PRIVATE /W4 /permissive- /bigobj)
else()
    target_compile_options(logit_scope_core PRIVATE -Wall -Wextra -Wpedantic)
    target_compile_options(logit-scope PRIVATE -Wall -Wextra -Wpedantic)
endif()

if(LOGIT_SCOPE_BUILD_TESTS)
    enable_testing()
    add_executable(rank-profile-tests tests/rank_profile_tests.cpp)
    target_link_libraries(rank-profile-tests PRIVATE logit_scope_core)
    add_test(NAME rank-profile-tests COMMAND rank-profile-tests)
endif()

install(TARGETS logit-scope RUNTIME DESTINATION bin)
install(FILES LICENSE NOTICE THIRD_PARTY_NOTICES.md DESTINATION ${CMAKE_INSTALL_DATADIR}/logit-scope)
install(FILES third_party/llama.cpp/LICENSE DESTINATION ${CMAKE_INSTALL_DATADIR}/logit-scope/licenses RENAME llama.cpp-LICENSE)
install(FILES third_party/llama.cpp/vendor/cpp-httplib/LICENSE DESTINATION ${CMAKE_INSTALL_DATADIR}/logit-scope/licenses
        RENAME cpp-httplib-LICENSE)
install(FILES licenses/nlohmann-json-LICENSE DESTINATION ${CMAKE_INSTALL_DATADIR}/logit-scope/licenses)
