cmake_minimum_required(VERSION 3.0)

project(pgdl)

# C++17 support
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-register")

set(SET_VERBOSE_MAKEFILE ON)

find_package(PostgreSQL REQUIRED)
find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)

set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
set(CMAKE_INSTALL_RPATH ${TORCH_LIBRARIES})
# nlp text tokenize
#find_package(SentencePiece REQUIRED)

# add lib source file
FILE(GLOB LIB_SOURCES "./src/pgdl/*.cpp" "./src/pgdl/*.c" "./src/external_process/*.cpp" "./src/external_process/*.c")
#FILE(GLOB EXTERNAL_SOURCE "./src/external_process/*.cpp" "./src/external_process/*.c")
add_library(pgdl SHARED ${LIB_SOURCES})
#add_library(external SHARED ${EXTERNAL_SOURCE})
set_target_properties(pgdl PROPERTIES PREFIX "")
#set_target_properties(external PROPERTIES PREFIX "")

set(TORCH_INCLUDE_DIRS "${TORCH_INSTALL_PREFIX}/include;${TORCH_INSTALL_PREFIX}/torch/csrc/api/include")
message(STATUS "TORCH_INCLUDE_DIRS:${TORCH_INCLUDE_DIRS}")

# Check if pg_config command is executable 
find_program(PG_CONFIG_EXECUTABLE pg_config)
if(PG_CONFIG_EXECUTABLE)
    # set postgres share dir
    execute_process( COMMAND ${PG_CONFIG_EXECUTABLE} --sharedir OUTPUT_VARIABLE PostgreSQL_SHARE_DIR OUTPUT_STRIP_TRAILING_WHITESPACE )
    # set postgres pkglib dir
    execute_process( COMMAND ${PG_CONFIG_EXECUTABLE} --pkglibdir OUTPUT_VARIABLE PostgreSQL_PKGLIB_DIR OUTPUT_STRIP_TRAILING_WHITESPACE )
else()
    message(ERROR "pg_config not exist")
endif()

# torch include
include_directories(${Torch_INCLUDE_DIRS})

# opencv include
include_directories(${OpenCV_INCLUDE_DIRS})

# postgres include
include_directories(${PostgreSQL_INCLUDE_DIRS})

#include_directories(${SentencePiece_INCLUDE_DIRS})


include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src/external_process)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src/pgdl)


# torch libs
target_link_libraries(pgdl ${TORCH_LIBRARIES})

# opencv libs
target_link_libraries(pgdl ${OpenCV_LIBS})

# postgres libs
target_link_libraries(pgdl ${PostgreSQL_LIBRARIES})


#set(SentencePiece_INCLUDE_DIR /home/lhh/sentence/include/) 
#set(SentencePiece_LIB_DIR /home/lhh/sentence/lib64/libsentencepiece.so)

#set_target_properties(pgdl PROPERTIES INSTALL_RPATH ${TORCH_LIBRARIES})

#include_directories(${SentencePiece_INCLUDE_DIR})
target_link_libraries(pgdl sentencepiece)

# install extension share lib
install(FILES build/pgdl.so
        DESTINATION "${PostgreSQL_PKGLIB_DIR}"
)

# install control file and sql file
install(FILES pgdl.control sql/pgdl--1.0.0.sql
        DESTINATION "${PostgreSQL_SHARE_DIR}/extension"
)
