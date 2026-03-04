cmake_minimum_required(VERSION 3.20)

project(${SKBUILD_PROJECT_NAME} LANGUAGES C)

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module NumPy)

set(PYX_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/archetypes/numpy/_projection.pyx")
set(CYTHON_C_FILE "${CMAKE_CURRENT_BINARY_DIR}/_projection.c")

add_custom_command(
  OUTPUT "${CYTHON_C_FILE}"
  COMMAND "${Python_EXECUTABLE}" -m cython
          -X language_level=3
          --fast-fail
          -X boundscheck=False
          -X wraparound=False
          -X nonecheck=False
          -X cdivision=True
          -X binding=True
          --output-file "${CYTHON_C_FILE}"
          "${PYX_SOURCE}"
  DEPENDS "${PYX_SOURCE}"
  VERBATIM
)

Python_add_library(_projection MODULE "${CYTHON_C_FILE}" WITH_SOABI)
set_target_properties(_projection PROPERTIES OUTPUT_NAME "_projection")

install(
  TARGETS _projection
  LIBRARY DESTINATION archetypes/numpy
  RUNTIME DESTINATION archetypes/numpy
)
