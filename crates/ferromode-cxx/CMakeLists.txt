cmake_minimum_required(VERSION 3.18)
project(ferromode_cxx LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)  # header uses std::span
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(FERROMODE_CXX_BUILD_TESTS "Build tests" OFF)

# Path to the cargo-built shared library (libferromode_cxx.so/.dylib/.dll).
# Build it first with: cargo build -p ferromode-cxx [--release]
set(FERROMODE_CXX_LIB "" CACHE FILEPATH "Path to the built ferromode_cxx shared library")

# Header-only convenience target (plus the linked Rust cdylib, if provided).
add_library(ferromode_cxx INTERFACE)
target_include_directories(ferromode_cxx INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_compile_features(ferromode_cxx INTERFACE cxx_std_20)
if(FERROMODE_CXX_LIB)
    target_link_libraries(ferromode_cxx INTERFACE ${FERROMODE_CXX_LIB})
endif()

include(GNUInstallDirs)
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
        FILES_MATCHING PATTERN "*.hpp")

if(FERROMODE_CXX_BUILD_TESTS)
    enable_testing()
    include(FetchContent)
    FetchContent_Declare(
        Catch2
        GIT_REPOSITORY https://github.com/catchorg/Catch2.git
        GIT_TAG v3.4.0
    )
    FetchContent_MakeAvailable(Catch2)

    add_executable(test_ferromode_cxx tests/test_ferromode.cpp)
    target_link_libraries(test_ferromode_cxx PRIVATE ferromode_cxx Catch2::Catch2WithMain)

    include(Catch)
    catch_discover_tests(test_ferromode_cxx)
endif()
