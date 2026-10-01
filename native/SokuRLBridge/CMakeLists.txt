cmake_minimum_required(VERSION 3.20)
project(SokuRLBridge LANGUAGES CXX)

if(NOT WIN32 OR NOT CMAKE_SIZEOF_VOID_P EQUAL 4)
    message(FATAL_ERROR "SokuRLBridge must be configured for Win32/x86")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

get_filename_component(SOKUMODS_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../third_party/SokuMods" ABSOLUTE)
if(NOT EXISTS "${SOKUMODS_ROOT}/SokuLib/CMakeLists.txt")
    message(FATAL_ERROR "SokuLib was not found under ${SOKUMODS_ROOT}")
endif()

add_subdirectory("${SOKUMODS_ROOT}/SokuLib" "${CMAKE_BINARY_DIR}/SokuLib" EXCLUDE_FROM_ALL)

add_library(SokuRLBridge MODULE
    ControlledInput.hpp
    ControlledInput.cpp
    HeldInput.hpp
    HeldInput.cpp
    FrameState.hpp
    FrameState.cpp
    FrameRecords.hpp
    FrameRecords.cpp
    FramePresentation.hpp
    FramePresentation.cpp
    ProfileDeck.hpp
    LocalStart.hpp
    LocalStart.cpp
    SceneReset.hpp
    SceneReset.cpp
    NetworkStart.hpp
    NetworkStart.cpp
    NetworkSelection.hpp
    NetworkState.hpp
    NetworkState.cpp
    NetworkInput.hpp
    NetworkInputSchedule.hpp
    NetworkInput.cpp
    NetworkInputEvents.hpp
    NetworkInputEvents.cpp
    AudioMute.hpp
    AudioMute.cpp
    ControlBlock.hpp
    ImageCapture.hpp
    ImageCapture.cpp
    RenderState.hpp
    RenderState.cpp
    SokuRLBridge.cpp
)
target_link_libraries(SokuRLBridge PRIVATE SokuLib)
target_compile_definitions(SokuRLBridge PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX _CRT_SECURE_NO_WARNINGS)
if(MSVC)
    target_compile_options(SokuRLBridge PRIVATE /W4)
endif()
set_target_properties(SokuRLBridge PROPERTIES PREFIX "" OUTPUT_NAME "SokuRLBridge")
set_property(SOURCE NetworkInput.cpp APPEND PROPERTY OBJECT_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/NetworkInputSchedule.hpp")
set_property(SOURCE NetworkInput.cpp tests/NetworkSelectionTest.cpp APPEND PROPERTY OBJECT_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/NetworkSelection.hpp")
set_property(SOURCE LocalStart.cpp APPEND PROPERTY OBJECT_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/ProfileDeck.hpp")
set_property(SOURCE RenderState.cpp ImageCapture.cpp NetworkState.cpp NetworkInput.cpp SokuRLBridge.cpp
    APPEND PROPERTY OBJECT_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/RenderState.hpp")
set_property(SOURCE NetworkState.cpp NetworkInput.cpp SokuRLBridge.cpp APPEND PROPERTY OBJECT_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/NetworkState.hpp")

include(CTest)
if(BUILD_TESTING)
    add_executable(ControlledInputTest tests/ControlledInputTest.cpp ControlledInput.cpp FrameRecords.cpp FrameState.cpp)
    target_link_libraries(ControlledInputTest PRIVATE SokuLib)
    target_compile_definitions(ControlledInputTest PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX)
    add_test(NAME ControlledInput COMMAND ControlledInputTest)
    add_executable(FrameRecordsTest tests/FrameRecordsTest.cpp FrameRecords.cpp FrameState.cpp)
    target_link_libraries(FrameRecordsTest PRIVATE SokuLib)
    target_compile_definitions(FrameRecordsTest PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX)
    add_test(NAME FrameRecords COMMAND FrameRecordsTest)
    add_executable(FrameStateHashTest tests/FrameStateHashTest.cpp FrameState.cpp)
    target_link_libraries(FrameStateHashTest PRIVATE SokuLib)
    target_compile_definitions(FrameStateHashTest PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX)
    add_test(NAME FrameStateHash COMMAND FrameStateHashTest)
    add_executable(ControlledStepTest tests/ControlledStepTest.cpp)
    add_test(NAME ControlledStep COMMAND ControlledStepTest)
    add_executable(NetworkInputScheduleTest tests/NetworkInputScheduleTest.cpp)
    add_test(NAME NetworkInputSchedule COMMAND NetworkInputScheduleTest)
    add_executable(RenderObjectSelectionTest tests/RenderObjectSelectionTest.cpp)
    add_test(NAME RenderObjectSelection COMMAND RenderObjectSelectionTest)
    add_executable(NetworkSeatTest tests/NetworkSeatTest.cpp)
    add_test(NAME NetworkSeat COMMAND NetworkSeatTest)
    add_executable(NetworkSelectionTest tests/NetworkSelectionTest.cpp)
    add_test(NAME NetworkSelection COMMAND NetworkSelectionTest)
endif()
