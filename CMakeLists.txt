cmake_minimum_required(VERSION 3.20)
project(MLP net)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_BUILD_TYPE Debug)
add_compile_options(-fconcepts-diagnostics-depth=2)

file(GLOB CC_SOURCES "./src/*.cc")
set(TESTS_DIR ../tests)

include_directories(./include ./include/top ./dependencies/SFML/include)

add_executable(app ../mainfile.cc ${CC_SOURCES})
link_directories(../dependencies/SFML/src)

target_link_libraries(app sfml-window sfml-system sfml-graphics)

enable_testing()

add_executable(mlp-test ${TESTS_DIR}/mlp-test.cc ${CC_SOURCES})
add_test(NAME mlp-test COMMAND mlp-test --force-colors -d)

add_executable(activations-test ${TESTS_DIR}/activations-test.cc ${CC_SOURCES})
add_test(NAME activations-test COMMAND activations-test --force-colors -d)

add_executable(mlpsaver-test ${TESTS_DIR}/mlpsaver-test.cc ${CC_SOURCES})
add_test(NAME mlpsaver-test COMMAND mlpsaver-test --force-colors -d)

add_executable(console-test ${TESTS_DIR}/console-test.cc ${CC_SOURCES})
add_test(NAME console-test COMMAND console-test --force-colors -d)

target_link_libraries(mlp-test sfml-window sfml-system sfml-graphics)
target_link_libraries(mlpsaver-test sfml-window sfml-system sfml-graphics)
target_link_libraries(activations-test sfml-window sfml-system sfml-graphics)
target_link_libraries(console-test sfml-window sfml-system sfml-graphics)


