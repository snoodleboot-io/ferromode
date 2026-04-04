cmake_minimum_required(VERSION 3.18)
project(ferromode-cxx LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(FERROMODE_CXX_BUILD_TESTS "Build tests" OFF)

# Fetch pre-built ferromode library
include(FetchContent)

FetchContent_Declare(
    ferromode
    GIT_REPOSITORY https://github.com/ferromode/ferromode.git
    GIT_TAG main
    SOURCE_SUBDIR crates/ferromode
)

# Find or fetch ferromode
find_package(ferromode QUIET)
if(NOT ferromode_FOUND)
    message(STATUS "Ferromode not found, fetching from repository")
    FetchContent_MakeAvailable(ferromode)
endif()

# Header-only library target
add_library(ferromode_cxx INTERFACE)
target_include_directories(ferromode_cxx INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(ferromode_cxx INTERFACE ferromode::ferromode)

# Require C++17
target_compile_features(ferromode_cxx INTERFACE cxx_std_17)

# Install rules
include(GNUInstallDirs)

install(TARGETS ferromode_cxx
    EXPORT ferromode_cxx-targets
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(DIRECTORY include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    FILES_MATCHING PATTERN "*.hpp"
)

install(EXPORT ferromode_cxx-targets
    FILE ferromode_cxx-config.cmake
    NAMESPACE ferromode::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/ferromode_cxx
)

# Tests
if(FERROMODE_CXX_BUILD_TESTS)
    enable_testing()

    FetchContent_Declare(
        Catch2
        GIT_REPOSITORY https://github.com/catchorg/Catch2.git
        GIT_TAG v3.4.0
    )
    FetchContent_MakeAvailable(Catch2)

    add_executable(test_ferromode_cxx tests/test_ferromode.cpp)
    target_link_libraries(test_ferromode_cxx PRIVATE
        ferromode_cxx
        Catch2::Catch2WithMain
    )

    include(Catch)
    catch_discover_tests(test_ferromode_cxx)
endif()
