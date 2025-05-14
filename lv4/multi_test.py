from multiprocessing import Pool
import numpy as np
import time

def is_prime(n):
    """Helper function to check if a number is prime"""
    if n <= 1:
        return False
    if n <= 3:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True

def func(x):
    """CPU-intensive function - counts primes and does matrix operations"""
    global coef  # This needs to be initialized with the initializer function
    
    # Prime number counting - CPU intensive
    count = 0
    range_size = 1000000 + int(x * 20000)  # Different workload based on input
    for num in range(2, range_size):
        if is_prime(num):
            count += 1
    
    # Matrix operations - also CPU intensive
    size = 2000
    matrix_a = np.random.rand(size, size)
    matrix_b = np.random.rand(size, size)
    result = np.dot(matrix_a, matrix_b)
    
    return coef * x, count, np.sum(result)

def initializer(coef_value):
    global coef
    coef = coef_value

def main():
    global coef
    coef = 4
    args = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30]
    
    # Test with multiple processes
    start_time = time.time()
    with Pool(4, initializer=initializer, initargs=(coef,)) as p:
        res_multi = p.map(func, args)
    multi_time_4= time.time() - start_time
    print(f"Multiprocessing time (4 processes): {multi_time_4:.2f} seconds")
    
    start_time = time.time()
    with Pool(6, initializer=initializer, initargs=(coef,)) as p:
        res_multi = p.map(func, args)
    multi_time_6 = time.time() - start_time
    print(f"Multiprocessing time (6 processes): {multi_time_6:.2f} seconds")

    start_time = time.time()
    with Pool(8, initializer=initializer, initargs=(coef,)) as p:
        res_multi = p.map(func, args)
    multi_time_8 = time.time() - start_time
    print(f"Multiprocessing time (8 processes): {multi_time_8:.2f} seconds")


    # For single process, we need to initialize the global variable
    # Test with single process
    start_time = time.time()
    res_single = [func(x) for x in args]
    single_time = time.time() - start_time
    print(f"Single process time: {single_time:.2f} seconds")
    
    # Calculate speedup
    print(f"Speedup (4 processes): {single_time/multi_time_4:.2f}x")
    print(f"Speedup (6 processes): {single_time/multi_time_6:.2f}x")
    print(f"Speedup (8 processes): {single_time/multi_time_8:.2f}x")

if __name__ == '__main__':
    main()