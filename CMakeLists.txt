CMAKE_MINIMUM_REQUIRED(VERSION 2.8 FATAL_ERROR)

PROJECT(general_tracker)

FIND_PACKAGE(OpenCV REQUIRED)

INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIRS})

ADD_EXECUTABLE(track
               lm/lmmin.h
               lm/lmmin_float.h
               lm/lmmin.cpp
               lm/lmmin_float.cpp
               track.cpp
               )
TARGET_LINK_LIBRARIES(track
                      ${OpenCV_LIBRARIES}
                      )
