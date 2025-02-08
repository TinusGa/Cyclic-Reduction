## About this repository

This repository attempts to implement parallel cyclic reduction for the solution of block tridiagonal linear systems. In other words, solving $Mx=f$ faaast. This is a work in progress and only some of the code is stable. 

The first implementation reconstructed the algorithm proposed by P, Amodio and N. Mastronardi in [this paper](https://doi.org/10.1016/0167-8191(93)90031-F), together with [this paper](https://doi.org/10.1016/0898-1221(93)90109-9) also by P. Amodio. The corresponding code can be found under CR_v1/cyclic_reduction_v8 (yes it took 8 versions to complete). If you want to test the code, good luck. Some tests for various problem sizes are provided, but you'll have to navigate the code yourself. I recently moved some files to clean the repository so running the aforementioned script will most likely fail, but it shouldn't be too hard to fix.

Currently, work is focused on impementing parallel cyclic reduction for the solution of lower block bidiagonal linear systems. This implementation is self made, so it will be interesting to see how it turns out. Most of the work is complete, with only the final step (back-substitution) unfinished. Hopefully, this won't take too long, inshallah. 

![Image](https://github.com/user-attachments/assets/6cf300b1-10b6-4e9b-aa55-fba2bf204bbf)
