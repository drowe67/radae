include_directories(${CMAKE_CURRENT_SOURCE_DIR} ${OPUS_DIR}/dnn ${OPUS_DIR}/celt ${OPUS_DIR}/include)

add_executable(lpcnet_demo lpcnet_demo.c)
target_link_libraries(lpcnet_demo ${OPUS_DIR}/libopus.a m)
