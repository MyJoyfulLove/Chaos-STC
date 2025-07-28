import random
import numpy as np
from scipy.signal import convolve2d


class Chaos:
    def __init__(self, x0: float = 0.1, r: float = 3.95, length: int = 1000):
        """
        :param x0: initial value
        :param r: key
        :param length: length of chaos
        """
        self.x0 = x0
        self.r = r
        self.length = length

    def logistic_map(self):
        sequence = []
        x = self.x0
        for _ in range(self.length):
            x = self.r * x * (1 - x)
            sequence.append(x)
        print("logistic_map chaos")
        return np.array(sequence)

    def tent_map(self):
        sequence = []
        x = self.x0
        for _ in range(self.length):
            if x < 0.5:
                x = self.r * x
            else:
                x = self.r * (1 - x)
            sequence.append(x)
        print("tent_map chaos")
        return np.array(sequence)

    def improved_lorenz(self, x=0.01, y=0.02, z=0.03, a=20, b=50, c=8, tau=1, dt=0.01):
        def lorenz_system(x, y, z, a, b, c, dt):
            x_dot = a * (y - x)
            y_dot = x * (b - z) - y
            z_dot = x * y - c * z
            x = x + x_dot * dt
            y = y + y_dot * dt
            z = z + z_dot * dt
            return x, y, z

        x_vals, y_vals, z_vals = [], [], []
        x_hist, y_hist, z_hist = [x], [y], [z]
        for i in range(self.length):
            if i > tau:
                x_mod = (x + x_hist[i - tau] + np.sin(i * dt)) % 1
                y_mod = (y + y_hist[i - tau] + np.sin(i * dt)) % 1
                z_mod = (z + z_hist[i - tau] + np.sin(i * dt)) % 1
            else:
                x_mod, y_mod, z_mod = x, y, z
            x, y, z = lorenz_system(x_mod, y_mod, z_mod, a, b, c, dt)
            x_hist.append(x)
            y_hist.append(y)
            z_hist.append(z)
            x_vals.append(x)
            y_vals.append(y)
            z_vals.append(z)
        print("improved_lorenz chaos")
        return np.array(random.choice([x_vals, y_vals, z_vals]))


class ChaoticMatrix:
    def __init__(self, height, width, chaos: np.array):
        self.height = height
        self.width = width
        self.chaos = chaos

    def generate_chaotic_qc_ldpc(self, p=2, q=3, seed=None) -> np.array:
        """
        Parity-check matrix based on QC-LDPC
        :param p: rows of basic matrix (<height)
        :param q: cols of basic matrix (<width)
        :param seed: random seed
        :return: parity-check matrix
        """
        if seed is not None:
            np.random.seed(seed)

        z = max(self.height // p, self.width // q)

        base_matrix = np.zeros((p, q), dtype=int)
        chaos_index = 0
        for i in range(p):
            for j in range(q):
                base_matrix[i, j] = 1 if self.chaos[chaos_index] > 0.5 else 0
                chaos_index += 1

        chaos_index = 0
        shift_matrix = np.zeros((p, q), dtype=int)
        for i in range(p):
            for j in range(q):
                shift_matrix[i, j] = int(self.chaos[chaos_index] * z) % z
                chaos_index += 1

        H = np.zeros((self.height, self.width), dtype=int)
        for i in range(p):
            for j in range(q):
                if base_matrix[i, j] == 1:
                    shift = shift_matrix[i, j]
                    block = np.zeros((z, z), dtype=int)
                    for k in range(z):
                        block[(k + shift) % z, k] = 1

                    start_row = i * z
                    start_col = j * z
                    end_row = min(start_row + z, self.height)
                    end_col = min(start_col + z, self.width)

                    block_height = end_row - start_row
                    block_width = end_col - start_col

                    if block_height > 0 and block_width > 0:
                        H[start_row:end_row, start_col:end_col] = block[:block_height, :block_width]
        print("Parity-check matrix based on QC-LDPC")
        return H

    def generate_chaotic_conv_matrix(self, kernel_size=3, density=0.3) -> np.array:
        """
        Parity-check matrix based on convolution
        :param kernel_size: kernel size
        :param density: density of matrix
        :return: parity-check
        """
        H_init = np.zeros((self.height, self.width))
        chaos_index = 0
        for i in range(self.height):
            for j in range(self.width):
                if self.chaos[chaos_index] < density:
                    H_init[i, j] = 1
                chaos_index += 1

        kernel = np.ones((kernel_size, kernel_size))
        H_conv = convolve2d(H_init, kernel, mode='same')

        H = (H_conv > np.percentile(H_conv, 70)).astype(int)

        # 确保尺寸正确
        print("Parity-check matrix based on convolution")
        return H[:self.height, :self.width]

    def generate_chaotic_structured_matrix(self, h=3, w=4, seed=None):
        """
        Parity-check matrix based on structured
        :param h: height
        :param w: width
        :param seed: random seed
        :return: parity-check matrix
        """
        if seed is not None:
            np.random.seed(seed)

        H = np.zeros((self.height, self.width), dtype=int)
        chaos_index = 0

        for diag in range(0, self.height + self.width - 1):
            i_start = max(0, diag - self.width + 1)
            i_end = min(diag + 1, self.height)
            for i in range(i_start, i_end):
                j = diag - i
                if self.chaos[chaos_index] > 0.7:
                    for dx in range(-h // 2, h // 2 + 1):
                        for dy in range(-w // 2, w // 2 + 1):
                            ni, nj = i + dx, j + dy
                            if 0 <= ni < self.height and 0 <= nj < self.width:
                                if self.chaos[chaos_index + 1] > 0.5:
                                    H[ni, nj] = 1
                    chaos_index += 2
                else:
                    chaos_index += 1
            pass
        return H


class DetermineMatrix:
    def __init__(self, message, pc_matrix):
        self.message = message.reshape((-1, 1))
        self.pc_matrix = pc_matrix

    def is_full_rank(self):
        rank_pc_matrix = np.linalg.matrix_rank(self.pc_matrix)
        print("校验矩阵是否是行满秩：", rank_pc_matrix == self.pc_matrix.shape[0])
        return rank_pc_matrix == self.pc_matrix.shape[0]

    def is_rank_equal(self):
        rank_pc_matrix = np.linalg.matrix_rank(self.pc_matrix)
        print("The rank of parity-check matrix is ", rank_pc_matrix)
        augmented_matrix = np.hstack((self.pc_matrix, self.message))
        rank_augmented_matrix = np.linalg.matrix_rank(augmented_matrix)
        print("The rank of augmented matrix is ", rank_augmented_matrix)
        print("Check whether there is a solution：", rank_pc_matrix == rank_augmented_matrix)
        return rank_pc_matrix == rank_augmented_matrix

    def is_sparse(self):
        height, width = self.pc_matrix.shape
        total_elements = height * width
        total_ones = sum(sum(row) for row in self.pc_matrix)
        total_zeros = total_elements - total_ones
        print("Check whether the matrix is sparse：", total_zeros > total_ones)
        return total_zeros > total_ones

    def is_1_exist(self):
        is_exist = all(any(row) for row in self.pc_matrix)
        print("Check whether at least one 1 exists in each row of the matrix：", is_exist)
        return is_exist


if __name__ == "__main__":
    chaos_seq = Chaos(x0=0.2, r=3.65).logistic_map()
    # chaos_seq = Chaos(x0=0.1, r=3.65).improved_lorenz()
    print(chaos_seq)
    H_qc = ChaoticMatrix(8, 10, chaos_seq).generate_chaotic_qc_ldpc()
    print("QC-LDPC:\n")
    print(H_qc)

    H_conv = ChaoticMatrix(8, 12, chaos_seq).generate_chaotic_conv_matrix()
    print("Convolution:\n")
    print(H_conv)

    H_struct = ChaoticMatrix(8, 12, chaos_seq).generate_chaotic_structured_matrix(h=3, w=4, seed=42)
    print("\nStructured:\n")
    print(H_struct)

    H_chaos = np.array([ [0,1,1,1,0,0,0,0,0,0,0,0],
                         [1,1,1,1,0,0,0,0,0,0,0,0],
                         [0,0,0,0,0,0,1,0,0,0,0,0],
                         [0,0,0,0,1,0,0,0,0,0,0,0],
                         [0,0,0,1,1,1,0,0,0,0,0,0],
                         [0,0,1,1,1,1,0,0,0,0,1,0],
                         [0,0,0,1,1,1,1,0,0,0,1,0],
                         [0,0,0,0,1,1,1,0,0,0,0,0]])
    determine = DetermineMatrix(np.array([0, 1, 0, 1, 0, 1, 0, 0]), H_chaos)
    determine.is_full_rank()
    determine.is_rank_equal()
    determine.is_sparse()
    determine.is_1_exist()
