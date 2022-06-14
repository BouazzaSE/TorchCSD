add_subdirectory(activations)
add_subdirectory(layers)
add_library(modules INTERFACE)
target_link_libraries(modules INTERFACE activations layers)
