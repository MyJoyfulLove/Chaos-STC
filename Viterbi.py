import random
import numpy as np
from typing import Tuple, List
from Chaos_Matrix import Chaos, ChaoticMatrix


class Viterbi:
    def __init__(self, carrier: np.array, message: np.array, pc_submatrix: np.array, pc_matrix: np.array):
        self.carrier = carrier
        self.message = message
        self.pc_submatrix = pc_submatrix
        self.pc_matrix = pc_matrix
        self.rho = self.rhos()

    def rhos(self):
        """
        The cost function, is used to specify the weight of each position.
        The larger the value, the more likely the position is on the modification path,
        that is, the more likely it is to change
        :return: Weight per position
        """
        chaos_rho = Chaos(length=self.carrier.size).improved_lorenz(x=1.4, y=0.1, z=1.9)
        return np.asarray(chaos_rho)
        # return np.ones(self.carrier.size)
        # return np.random.randint(0, 2, self.carrier.size)

    def viterbi_sub_matrix(self) -> Tuple[list, int]:
        """
        Viterbi algorithm based on sub-matrix
        :return: Tuple[list, int]:
                First item is coded bits array
                Second item is embedding cost
        """
        height, width = self.pc_submatrix.shape
        wght = [float('inf')] * (2 ** height)
        wght[0] = 0
        indx = 0
        indm = 0
        path = [{} for _ in range(self.carrier.size)]

        for _ in range(self.message.size):
            for j in range(width):
                new_wght = wght.copy()
                for k in range(2 ** height):
                    w0 = wght[k] + self.carrier[indx] * self.rho[indx]
                    curCol = int(''.join(str(h_hat_item) for h_hat_item in reversed(self.pc_submatrix[:, j])), 2)
                    w1 = wght[k ^ curCol] + (1 - self.carrier[indx]) * self.rho[indx]
                    path[indx][k] = 1 if w1 < w0 else 0
                    new_wght[k] = min(w0, w1)
                indx += 1
                wght = new_wght.copy()
            # pruning
            for j in range(2 ** (height - 1)):
                wght[j] = wght[2 * j + int(self.message[indm])]
            wght[2 ** (height - 1):2 ** height] = [float('inf')] * (2 ** height - 2 ** (height - 1))
            indm += 1
            if indm >= len(self.message):
                break

        # backward
        indx -= 1
        indm -= 1
        embedding_cost = min(wght)
        state = wght.index(embedding_cost)
        state = 2 * state + int(self.message[indm])
        indm -= 1
        y = [0] * len(path)
        for _ in range(self.message.size):
            for j in range(width - 1, -1, -1):
                y[indx] = path[indx].get(state) or 0
                curCol = int(''.join(str(h_hat_item) for h_hat_item in reversed(self.pc_submatrix[:, j])), 2)
                state = state ^ (y[indx] * curCol)
                indx -= 1
            if indm < 0:
                break
            state = 2 * state + int(self.message[indm])
            indm -= 1
        return y, embedding_cost

    def viterbi_full_matrix(self) -> Tuple[List, float]:
        """
        Viterbi algorithm based on complete check matrix
        :return: Tuple[List, float]
        """
        height, width = self.pc_matrix.shape

        if self.message.size != height:
            raise ValueError(f"message length {self.message.size} is not matching height of matrix {height}")

        if self.carrier.size != width:
            raise ValueError(f"cover length {self.carrier.size} is not matching width of matrix {width}")

        # state space size is 2^m
        num_states = 2 ** height

        # initial
        wght = [float('inf')] * num_states
        wght[0] = 0  # initial state

        # record the optimal selection of each position and each state
        path = [{} for _ in range(width)]

        # forward
        for pos in range(width):
            new_wght = [float('inf')] * num_states
            h_col = self.pc_matrix[:, pos]
            col_value = 0
            for i in range(height):
                col_value += h_col[i] * (2 ** i)
            for state in range(num_states):
                if wght[state] == float('inf'):
                    continue
                cost_0 = wght[state] + (self.carrier[pos] * self.rho[pos])
                new_state_0 = state
                cost_1 = wght[state] + ((1 - self.carrier[pos]) * self.rho[pos])
                new_state_1 = state ^ col_value
                if cost_0 < new_wght[new_state_0]:
                    new_wght[new_state_0] = cost_0
                    path[pos][new_state_0] = 0
                if cost_1 < new_wght[new_state_1]:
                    new_wght[new_state_1] = cost_1
                    path[pos][new_state_1] = 1
                pass
            wght = new_wght
            # print(f"smallest weight in position {pos} : {min([w for w in wght if w != float('inf')])}")
        target_state = 0
        for i in range(height):
            target_state += self.message[i] * (2 ** i)

        # check whether the target state is reachable
        if wght[target_state] == float('inf'):
            raise ValueError("The target state is unreachable, and the embedding cannot be completed")
        embedding_cost = wght[target_state]

        # backward
        y = [0] * width
        current_state = target_state
        for pos in range(width - 1, -1, -1):
            choice = path[pos].get(current_state, 0)
            y[pos] = choice
            h_col = self.pc_matrix[:, pos]
            col_value = 0
            for i in range(height):
                col_value += h_col[i] * (2 ** i)
            if choice == 1:
                current_state = current_state ^ col_value
            # print(f"pos {pos}: choice {choice}, current_state {current_state}")

        return y, embedding_cost


def generate_parity_matrix(message_length, carrier_length, pc_submatrix: np.array):
    """
    Generate check matrix H according to sub matrix H_hat
    :param message_length: length of message
    :param carrier_length: length of cover
    :param pc_submatrix: sub-matrix
    :return: parity-check matrix
    """
    w, h = pc_submatrix.shape
    pc_matrix = np.int16(np.zeros((message_length, carrier_length)))
    row_start = 0
    col_start = 0
    while row_start < message_length and col_start < carrier_length:
        rows_to_place = min(w, message_length - row_start)
        cols_to_place = min(h, carrier_length - col_start)
        pc_matrix[row_start: row_start + rows_to_place, col_start: col_start + cols_to_place] \
            = pc_submatrix[:rows_to_place,: cols_to_place]
        row_start += 1
        col_start += h
        pass
    return pc_matrix


if __name__ == "__main__":
    print('############# sub-matrix ##############')
    x1 = np.random.randint(0, 2, 8)
    print("original cover:\t", x1)
    m1 = np.random.randint(0, 2, 4)
    print("message:\t", m1)
    H_hat = np.array([[1, 0],
                      [1, 1]])  # 子矩阵
    #################################################
    V1 = Viterbi(x1, m1, H_hat, None)
    stego1, cost1 = V1.viterbi_sub_matrix()
    stego1 = np.asarray(stego1)
    print("embedded cover:\t", stego1)
    print("cost:", cost1)
    H = generate_parity_matrix(len(m1), len(x1), H_hat)
    print("matrix × stego \n extracted message：", np.mod(np.dot(H, stego1), 2))
    print('########## Yes ##########' if (np.mod(np.dot(H, stego1), 2) == m1).all() else '########## No ##########')

    print('############# whole matrix ##############')
    x2 = np.random.randint(0, 2, 10)
    print("original cover:\t", x2)
    m2 = np.random.randint(0, 2, 4)
    print("message:\t", m2)
    H2 = np.array([[1, 0, 0, 0, 0, 1, 0, 0, 0, 1],
                   [1, 1, 1, 0, 0, 0, 0, 1, 0, 0],
                   [0, 0, 1, 0, 1, 0, 1, 0, 0, 0],
                   [0, 0, 0, 0, 1, 0, 0, 0, 1, 0]])
    #################################################
    V2 = Viterbi(x2, m2, None, H2)
    stego2, cost2 = V2.viterbi_full_matrix()
    stego2 = np.asarray(stego2)
    print("embedded cover:\t", stego2)
    print("cost:", cost2)
    print("matrix × stego2 \n extracted message：", np.mod(np.dot(H2, stego2), 2))
    print('########## Yes ##########' if (np.mod(np.dot(H2, stego2), 2) == m2).all() else '########## No ##########')
    # quit()

    print('############# chaotic matrix ##############')
    x3 = np.random.randint(0, 2, 10)
    print("original cover:\t", x3)
    m3 = np.random.randint(0, 2, 4)
    print("message:\t", m3)
    original_state = random.getstate()
    chaos_seq = Chaos(x0=0.1).logistic_map()
    H_chaos = ChaoticMatrix(len(m3), len(x3), chaos_seq).generate_chaotic_qc_ldpc()
    print("chaotic matrix:\n", H_chaos)
    #################################################
    V_chaos = Viterbi(x3, m3, None, H_chaos)
    stego3, cost3 = V_chaos.viterbi_full_matrix()
    stego3 = np.asarray(stego3)
    print("embedded cover:\t", stego3)
    print("cost:", cost3)
    print("matrix × stego2 \n extracted message：", np.mod(np.dot(H_chaos, stego3), 2))
    print(
        '########## Yes ##########' if (np.mod(np.dot(H_chaos, stego3), 2) == m3).all() else '########## No ##########')
