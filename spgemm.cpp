#include <vector>
#include <map>
#include <algorithm>
#include <utility>
#include <iostream>
#include <mpi.h>
#include <cassert>
#include <unordered_map>
#include "functions.h"

void spgemm_2d(int m, int p, int n,
               std::vector<std::pair<std::pair<int,int>, int>> &A,
               std::vector<std::pair<std::pair<int,int>, int>> &B,
               std::vector<std::pair<std::pair<int,int>, int>> &C,
               std::function<int(int, int)> plus, std::function<int(int, int)> times,
               MPI_Comm row_comm, MPI_Comm col_comm)
{
    // Get the communicator information
    int row_rank, row_size, col_rank, col_size;
    MPI_Comm_rank(row_comm, &row_rank);
    MPI_Comm_size(row_comm, &row_size);
    MPI_Comm_rank(col_comm, &col_rank);
    MPI_Comm_size(col_comm, &col_size);

    // Map to store the result of local computations
    std::map<std::pair<int, int>, int> result_matrix;
    
    // Temporary matrices for broadcast operations
    std::vector<std::pair<std::pair<int,int>, int>> A_block;
    std::vector<std::pair<std::pair<int,int>, int>> B_block;
    
    // Implement 2D SUMMA algorithm
    for (int k = 0; k < row_size; k++) {
        // Broadcast A block across rows
        if (row_rank == k) {
            A_block = A;
        }
        
        // Broadcast A's size first
        int A_size = (row_rank == k) ? A_block.size() : 0;
        MPI_Bcast(&A_size, 1, MPI_INT, k, row_comm);
        
        // Resize and broadcast the actual data
        if (row_rank != k) {
            A_block.resize(A_size);
        }
        MPI_Bcast(A_block.data(), A_size * sizeof(A[0]), MPI_BYTE, k, row_comm);
        
        // Broadcast B block down columns
        if (col_rank == k) {
            B_block = B;
        }
        
        // Broadcast B's size first
        int B_size = (col_rank == k) ? B_block.size() : 0;
        MPI_Bcast(&B_size, 1, MPI_INT, k, col_comm);
        
        // Resize and broadcast the actual data
        if (col_rank != k) {
            B_block.resize(B_size);
        }
        MPI_Bcast(B_block.data(), B_size * sizeof(B[0]), MPI_BYTE, k, col_comm);
        
        // Create hash map for faster lookup of B entries
        std::unordered_map<int, std::vector<std::pair<int, int>>> B_hashmap;
        for (const auto& entry : B_block) {
            int row = entry.first.first;
            int col = entry.first.second;
            int val = entry.second;
            B_hashmap[row].emplace_back(col, val);
        }
        
        // Perform local matrix multiplication
        for (const auto& a_entry : A_block) {
            int a_row = a_entry.first.first;
            int a_col = a_entry.first.second;
            int a_val = a_entry.second;
            
            // Find matching rows in B
            auto it = B_hashmap.find(a_col);
            if (it != B_hashmap.end()) {
                // For each matching entry, compute product and update result
                for (const auto& [b_col, b_val] : it->second) {
                    std::pair<int, int> matrix_pos(a_row, b_col);
                    int product = times(a_val, b_val);
                    
                    // Add to local results using the provided plus function
                    if (result_matrix.find(matrix_pos) == result_matrix.end()) {
                        result_matrix[matrix_pos] = product;
                    } else {
                        result_matrix[matrix_pos] = plus(result_matrix[matrix_pos], product);
                    }
                }
            }
        }
    }
    
    // Convert result map to the expected output format
    C.clear();
    for (const auto& [coords, value] : result_matrix) {
        C.push_back({coords, value});
    }
}
