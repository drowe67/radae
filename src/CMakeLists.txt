include_directories(${CMAKE_CURRENT_SOURCE_DIR} ${OPUS_DIR}/dnn ${OPUS_DIR}/celt ${OPUS_DIR}/include)

# build static library from previously built opus object files
# TODO (a) pull git repo & build opus rather than separate step (b) build libopus.a from sources opus/*.mk files
if(NOT EXISTS ${OPUS_DIR}/libopus.a)
    message(STATUS "Building libopus.a")
    execute_process(COMMAND ${CMAKE_SOURCE_DIR}/build_libopus.sh ${OPUS_DIR}
                    COMMAND_ECHO STDOUT
                    COMMAND_ERROR_IS_FATAL ANY)
else()
    message(STATUS "libopus.a already exists")
endif()

add_executable(lpcnet_demo lpcnet_demo.c)
target_link_libraries(lpcnet_demo ${OPUS_DIR}/libopus.a m)
