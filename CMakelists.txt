cmake_minimum_required(VERSION 3.16)
project(simple_nn C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

add_executable(simple-nn
    main.c
    matrix.c
)

target_include_directories(simple-nn
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(simple-nn
    PRIVATE
        m
)
