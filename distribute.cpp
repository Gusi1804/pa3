#include <algorithm>
#include <vector>
#include <iostream>
#include <fstream>
#include <cassert>
#include <cmath>
#include <string>
#include <sstream>
#include "functions.h"

void distribute_matrix_2d(int m, int n, std::vector<std::pair<std::pair<int, int>, int>> &full_matrix,
                          std::vector<std::pair<std::pair<int, int>, int>> &local_matrix,
                          int root, MPI_Comm comm_2d)
{
    // Get the process rank and size information
    int rank, size;
    MPI_Comm_rank(comm_2d, &rank);
    MPI_Comm_size(comm_2d, &size);

    // Calculate grid dimensions for the 2D processor topology
    int grid_dim = static_cast<int>(std::sqrt(size));
    
    // Calculate block sizes for row and column distribution
    int row_block_size = m / grid_dim;
    int col_block_size = n / grid_dim;

    if (rank == root) {
        // Root process distributes the matrix to all processes
        for (int dest = 0; dest < size; dest++) {
            // Prepare buffer for data to send to current destination
            std::vector<std::pair<std::pair<int, int>, int>> partition_buffer;
            
            // Get 2D coordinates of the destination process
            int dest_coords[2];
            MPI_Cart_coords(comm_2d, dest, 2, dest_coords);
            
            // Calculate block boundaries for this process
            int row_start = dest_coords[0] * row_block_size;
            // Handle edge case for last process in row dimension
            int row_end = (dest_coords[0] == grid_dim - 1) ? m : (row_start + row_block_size);
            
            int col_start = dest_coords[1] * col_block_size;
            // Handle edge case for last process in column dimension
            int col_end = (dest_coords[1] == grid_dim - 1) ? n : (col_start + col_block_size);
            
            // Select matrix elements that belong to this process
            for (const auto &entry : full_matrix) {
                int row = entry.first.first;
                int col = entry.first.second;
                
                // Check if this entry belongs to the current partition
                if (row >= row_start && row < row_end && col >= col_start && col < col_end) {
                    partition_buffer.push_back(entry);
                }
            }
            
            // Handle data for this process based on whether it's the root or not
            if (dest == root) {
                // For root process, directly assign to local_matrix
                local_matrix = partition_buffer;
            } else {
                // For other processes, send the data via MPI
                int buffer_size = partition_buffer.size();
                
                // First send the size of the buffer
                MPI_Send(&buffer_size, 1, MPI_INT, dest, 0, comm_2d);
                
                // Then send the actual data
                MPI_Send(partition_buffer.data(), 
                         buffer_size * sizeof(partition_buffer[0]), 
                         MPI_BYTE, dest, 1, comm_2d);
            }
        }
    } else {
        // Non-root processes receive their portion of the matrix
        int recv_count;
        
        // First receive the size of the incoming data
        MPI_Recv(&recv_count, 1, MPI_INT, root, 0, comm_2d, MPI_STATUS_IGNORE);
        
        // Resize the local buffer and receive the data
        local_matrix.resize(recv_count);
        MPI_Recv(local_matrix.data(), 
                 recv_count * sizeof(local_matrix[0]), 
                 MPI_BYTE, root, 1, comm_2d, MPI_STATUS_IGNORE);
    }
}
