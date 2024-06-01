cmake_minimum_required(VERSION 3.20)
project(mlp)


set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(SRC_DIR ../src)
file(GLOB CC_SOURCES "./src/*.cc")
set(TESTS_DIR ../tests)
include_directories(../include ../include/top)
add_executable(app ${CC_SOURCES} ../mainfile.cc)
link_directories(${SRC_DIR}/SFML)
target_link_libraries(app sfml-window sfml-system sfml-graphics)

add_executable(mlp-test ${TESTS_DIR}/mlp-test.cc ${SRC_DIR}/mlp.cc)
enable_testing()

add_executable(activations-test ${TESTS_DIR}/activations-test.cc ${SRC_DIR}/activations.cc ${SRC_DIR}/activations_table.cc ${SRC_DIR}/mlp.cc)
add_test(NAME activations-test COMMAND activations-test --force-colors -d)



add_test(NAME mlp-test COMMAND mlp-test --force-colors -d)

