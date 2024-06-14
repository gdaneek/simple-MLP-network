cmake_minimum_required(VERSION 3.20)
project(MLPnet)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

#set(CMAKE_BUILD_TYPE Debug)
#add_compile_options(-fconcepts-diagnostics-depth=2)

include(FetchContent)
FetchContent_Declare(sfml 
			GIT_REPOSITORY "https://github.com/SFML/SFML.git" 
			GIT_TAG  "2.5.1"
			)
FetchContent_MakeAvailable(sfml)


file(GLOB CC_SOURCES "./src/*.cc")
set(TESTS_DIR ../tests)

include_directories(./include ./include/top) #./dependencies/SFML/include)

# Главное приложение
add_executable(app ../mainfile.cc ${CC_SOURCES})
#link_directories(../dependencies/SFML/src)
set(SFML_LIBS sfml-window sfml-system sfml-graphics)

target_link_libraries(app ${SFML_LIBS})

# Генератор датасетов
add_executable(dataset-gen ./src/dataset-gen/dgen.cc)

target_link_libraries(dataset-gen sfml-window ${SFML_LIBS})

# Тесты 

enable_testing()

add_executable(mlp-test ${TESTS_DIR}/mlp-test.cc ${CC_SOURCES})
add_test(NAME mlp-test COMMAND mlp-test --force-colors -d)

add_executable(activations-test ${TESTS_DIR}/activations-test.cc ${CC_SOURCES})
add_test(NAME activations-test COMMAND activations-test --force-colors -d)

add_executable(mlpsaver-test ${TESTS_DIR}/mlpsaver-test.cc ${CC_SOURCES})
add_test(NAME mlpsaver-test COMMAND mlpsaver-test --force-colors -d)

add_executable(console-test ${TESTS_DIR}/console-test.cc ${CC_SOURCES})
add_test(NAME console-test COMMAND console-test --force-colors -d)

target_link_libraries(mlp-test ${SFML_LIBS})
target_link_libraries(mlpsaver-test ${SFML_LIBS})
target_link_libraries(activations-test ${SFML_LIBS})
target_link_libraries(console-test ${SFML_LIBS})


