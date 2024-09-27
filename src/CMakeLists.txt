include_directories(${CMAKE_CURRENT_SOURCE_DIR})

add_executable(lpcnet_demo lpcnet_demo.c)
target_link_libraries(lpcnet_demo opus m)

add_executable(radae_tx radae_tx.c rade_api.c)
set_source_files_properties(rade_api.c PROPERTIES COMPILE_FLAGS "${PYTHON_C_FLAGS}")
target_link_libraries(radae_tx ${PYTHON_LD_LIBS})

add_executable(radae_rx radae_rx.c)
set_source_files_properties(radae_rx.c PROPERTIES COMPILE_FLAGS "${PYTHON_C_FLAGS}")
target_link_libraries(radae_rx ${PYTHON_LD_LIBS})
