from Chaos_Matrix import Chaos, ChaoticMatrix, DetermineMatrix
from Viterbi import Viterbi, generate_parity_matrix
import random
import cv2
import numpy as np


class STCEmbed:
    def __init__(self, image, bit_list, parity_check_matrix):
        self.image = image
        self.height, self.width, *self.channel = image.shape
        self.bit_list = bit_list
        self.bit_list_length = len(bit_list)
        self.parity_check_matrix = parity_check_matrix
        self.matrix_height, self.matrix_width = parity_check_matrix.shape
        self.groups_num = self.bit_list_length // self.matrix_height

    def select_pixels(self):
        selected_indices = [i for i in range(self.groups_num * self.matrix_width)]
        flat_image = self.image.reshape(-1)
        selected_pixels = flat_image[selected_indices]
        return selected_pixels, selected_indices

    def restore_pixels(self, modified_pixels, selected_indices):
        embedded_image = np.copy(self.image)
        flat_image = embedded_image.reshape(-1)
        flat_image[selected_indices] = modified_pixels
        embedded_image = flat_image.reshape(self.height, self.width, *self.channel)
        return embedded_image

    def embed(self):
        print("Embedding。。。")
        selected_pixels, selected_indices = self.select_pixels()

        for i in range(self.groups_num):
            block_pixels = selected_pixels[i * self.matrix_width: i * self.matrix_width + self.matrix_width]
            block_bit_list = self.bit_list[i * self.matrix_height: i * self.matrix_height + self.matrix_height]
            block_new_pixels = self.embed_bits_in_block(block_pixels, block_bit_list)
            selected_pixels[i * self.matrix_width: i * self.matrix_width + self.matrix_width] = block_new_pixels
            pass

        embedded_image = self.restore_pixels(selected_pixels, selected_indices)

        print("Finish Embedding")
        return embedded_image

    def embed_bits_in_block(self, block_pixels, block_bit_list) -> np.array:
        block_pixels_lsb = block_pixels & 0x01

        viterbi = Viterbi(np.asarray(block_pixels_lsb), np.asarray(block_bit_list), None, self.parity_check_matrix)
        block_pixels_new_lsb, cost = viterbi.viterbi_full_matrix()

        block_pixels = block_pixels & 0xFE  # 0xFE = 11111110
        block_pixels = block_pixels | np.asarray(block_pixels_new_lsb)
        return block_pixels


class STCExtract:
    def __init__(self, embedded_image, bit_list_length, parity_check_matrix):
        self.embedded_image = embedded_image
        self.height, self.width, *self.channel = embedded_image.shape
        self.bit_list_length = bit_list_length
        self.parity_check_matrix = parity_check_matrix
        self.matrix_height, self.matrix_width = parity_check_matrix.shape
        self.groups_num = self.bit_list_length // self.matrix_height

    def select_pixels(self):
        selected_indices = [i for i in range(self.groups_num * self.matrix_width)]
        flat_image = self.embedded_image.reshape(-1)
        selected_pixels = flat_image[selected_indices]
        return selected_pixels, selected_indices

    def extract(self):
        print("Extracting。。。")
        selected_pixels, selected_indices = self.select_pixels()
        extracted_lsb = []
        for i in range(self.groups_num):
            block_pixels = selected_pixels[i * self.matrix_width: i * self.matrix_width + self.matrix_width]
            block_lsb = self.extract_bits_from_block(block_pixels)
            extracted_lsb += list(block_lsb)
            pass
        print("Finish Extracting")
        return extracted_lsb

    def extract_bits_from_block(self, block_pixels) -> np.array:
        block_lsb = block_pixels & 0x01
        extracted_bits = np.mod(np.dot(self.parity_check_matrix, block_lsb), 2)
        return extracted_bits


class ChaosSTCEmbed:
    def __init__(self, image, bit_list, chaos_sequence, parity_check_matrix):
        self.image = image
        self.height, self.width, *self.channel = image.shape
        self.bit_list = bit_list
        self.bit_list_length = len(bit_list)
        self.chaos_sequence = chaos_sequence
        self.parity_check_matrix = parity_check_matrix
        self.matrix_height, self.matrix_width = parity_check_matrix.shape
        self.groups_num = self.bit_list_length // self.matrix_height

    def select_pixels(self):
        if self.channel:
            total_pixels = self.height * self.width * 3
        else:
            total_pixels = self.height * self.width
        normalized_chaos = ((self.chaos_sequence - np.min(self.chaos_sequence))
                            / (np.max(self.chaos_sequence) - np.min(self.chaos_sequence)))
        indices = np.arange(total_pixels)
        np.random.seed(42)
        np.random.shuffle(indices)

        chaos_indices = np.argsort(normalized_chaos[:total_pixels])
        permuted_indices = indices[chaos_indices]
        selected_indices = permuted_indices[:self.groups_num * self.matrix_width]
        flat_image = self.image.reshape(-1)
        selected_pixels = flat_image[selected_indices]
        return selected_pixels, selected_indices

    def restore_pixels(self, modified_pixels, selected_indices):
        embedded_image = np.copy(self.image)
        flat_image = embedded_image.reshape(-1)
        flat_image[selected_indices] = modified_pixels
        embedded_image = flat_image.reshape(self.height, self.width, *self.channel)
        return embedded_image

    def embed(self):
        print("Embedding。。。")
        selected_pixels, selected_indices = self.select_pixels()

        for i in range(self.groups_num):
            block_pixels = selected_pixels[i * self.matrix_width: i * self.matrix_width + self.matrix_width]
            block_bit_list = self.bit_list[i * self.matrix_height: i * self.matrix_height + self.matrix_height]
            block_new_pixels = self.embed_bits_in_block(block_pixels, block_bit_list)
            selected_pixels[i * self.matrix_width: i * self.matrix_width + self.matrix_width] = block_new_pixels
            pass

        embedded_image = self.restore_pixels(selected_pixels, selected_indices)

        print("Finish Embedding")
        return embedded_image

    def embed_bits_in_block(self, block_pixels, block_bit_list) -> np.array:
        block_pixels_lsb = block_pixels & 0x01

        viterbi = Viterbi(np.asarray(block_pixels_lsb), np.asarray(block_bit_list), None, self.parity_check_matrix)
        block_pixels_new_lsb, cost = viterbi.viterbi_full_matrix()

        block_pixels = block_pixels & 0xFE  # 0xFE = 11111110
        block_pixels = block_pixels | np.asarray(block_pixels_new_lsb)
        return block_pixels


class ChaosSTCExtract:
    def __init__(self, embedded_image, bit_list_length, chaos_sequence, parity_check_matrix):
        self.embedded_image = embedded_image
        self.height, self.width, *self.channel = embedded_image.shape
        self.bit_list_length = bit_list_length
        self.chaos_sequence = chaos_sequence
        self.parity_check_matrix = parity_check_matrix
        self.matrix_height, self.matrix_width = parity_check_matrix.shape
        self.groups_num = self.bit_list_length // self.matrix_height

    def select_pixels(self):
        if self.channel:
            total_pixels = self.height * self.width * 3
        else:
            total_pixels = self.height * self.width
        normalized_chaos = ((self.chaos_sequence - np.min(self.chaos_sequence))
                            / (np.max(self.chaos_sequence) - np.min(self.chaos_sequence)))
        indices = np.arange(total_pixels)
        np.random.seed(42)
        np.random.shuffle(indices)

        chaos_indices = np.argsort(normalized_chaos[:total_pixels])
        permuted_indices = indices[chaos_indices]
        selected_indices = permuted_indices[:self.groups_num * self.matrix_width]
        flat_image = self.embedded_image.reshape(-1)
        selected_pixels = flat_image[selected_indices]
        return selected_pixels, selected_indices

    def extract(self):
        print("Extracting。。。")
        selected_pixels, selected_indices = self.select_pixels()
        extracted_lsb = []
        for i in range(self.groups_num):
            block_pixels = selected_pixels[i * self.matrix_width: i * self.matrix_width + self.matrix_width]
            block_lsb = self.extract_bits_from_block(block_pixels)
            extracted_lsb += list(block_lsb)
            pass
        print("Finish Extracting")
        return extracted_lsb

    def extract_bits_from_block(self, block_pixels) -> np.array:
        block_lsb = block_pixels & 0x01
        extracted_bits = np.mod(np.dot(self.parity_check_matrix, block_lsb), 2)
        return extracted_bits


if __name__ == "__main__":
    ####################################################################################################
    root = "data"
    image_index = "11"
    image_path = "{}/{}.png".format(root, image_index)
    pc_matrix_height = 5  # length of message for each round
    pc_matrix_width = 20  # length of cover for each round
    message_length = 200000  # total length of message
    ####################################################################################################
    if message_length % pc_matrix_height != 0:
        raise ValueError(f"length of message ({message_length}) needs to be divided by the height of matrix ({pc_matrix_height})")
    original_img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if original_img is None:
        raise FileNotFoundError(f"can't read: {image_path}")
    # message
    bit_message = [random.randint(0, 1) for _ in range(message_length)]
    ####################################################################################################
    # STC
    h_hat = np.array([[1, 0],
                      [1, 1],
                      [0, 1]])
    h = generate_parity_matrix(pc_matrix_width, pc_matrix_height, h_hat)
    print("parity-check matrix:\n", h)
    stc_embed = STCEmbed(original_img, bit_message, h)
    embedded_img = stc_embed.embed()
    stc_extract = STCExtract(embedded_img, message_length, h)
    extract_message = stc_extract.extract()

    cv2.imwrite(
        "{}/{}_stc_h{}_w{}_msg{}.png".format(root, image_index, pc_matrix_height, pc_matrix_width, message_length),
        embedded_img)
    print(
        "######## Yes ########" if np.equal(np.asarray(bit_message), extract_message).all() else "######## No ########")

    ####################################################################################################
    # ChaosSTC
    chaos_seq = Chaos(x0=0.3, r=3.68, length=message_length * 2).logistic_map()
    h_chaos = ChaoticMatrix(pc_matrix_height, pc_matrix_width, chaos_seq).generate_chaotic_qc_ldpc()
    determine = DetermineMatrix(message=np.array([1] * message_length), pc_matrix=h_chaos)
    while not determine.is_full_rank():
        chaos_seq = Chaos(x0=random.random(), r=3.68, length=message_length * 2).logistic_map()
        h_chaos = ChaoticMatrix(pc_matrix_height, pc_matrix_width, chaos_seq).generate_chaotic_qc_ldpc()
        determine = DetermineMatrix(message=np.array([1] * message_length), pc_matrix=h_chaos)
    print("chaotic parity-check matrix:\n", h_chaos)
    chaos_stc_embed = ChaosSTCEmbed(original_img, bit_message, chaos_seq, h_chaos)
    chaos_embedded_img = chaos_stc_embed.embed()
    chaos_stc_extract = ChaosSTCExtract(chaos_embedded_img, message_length, chaos_seq, h_chaos)
    chaos_extract_message = chaos_stc_extract.extract()

    cv2.imwrite("{}/{}_chaos_stc_h{}_w{}_msg{}.png".format(root, image_index, pc_matrix_height, pc_matrix_width,
                                                           message_length), chaos_embedded_img)
    print("######## Yes ########" if np.equal(np.asarray(bit_message),
                                              chaos_extract_message).all() else "######## No ########")
