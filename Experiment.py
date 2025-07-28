import matplotlib.pyplot as plt
import numpy as np
import cv2
from scipy.stats import entropy
import os
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import roc_curve, auc
from skimage.metrics import structural_similarity as ssim


class Comparison:
    def __init__(self, original_image, embedded_image):
        self.original_image = original_image
        self.embedded_image = embedded_image
        _, _, *self.channel = self.original_image.shape
        if self.channel:
            self.channel = 3
        else:
            self.channel = 1

    def information(self):
        diff = np.mod(self.original_image - self.embedded_image, 255)
        original_image = np.mod(self.original_image, 2)
        embedded_image = np.mod(self.embedded_image, 2)
        print("Original image:\n Odd pixels: {}; Even pixels: {}".format(np.sum(original_image != 0), np.sum(original_image == 0)))
        print("Embedded image:\n Odd pixels: {}; Even pixels: {}".format(np.sum(embedded_image != 0), np.sum(embedded_image == 0)))
        print("Number of pixels changed：", np.sum(diff != 0))

    def display_image(self):
        if len(self.original_image.shape) == 2:
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))
            axes[0].imshow(self.original_image)
            axes[0].set_title('Original Image')
            axes[0].axis('off')

            axes[1].imshow(self.embedded_image)
            axes[1].set_title('Embedded Image')
            axes[1].axis('off')
        else:
            original_rgb = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
            embedded_rgb = cv2.cvtColor(self.embedded_image, cv2.COLOR_BGR2RGB)

            fig, axes = plt.subplots(1, 2, figsize=(12, 6))
            axes[0].imshow(original_rgb)
            axes[0].set_title('Original Image')
            axes[0].axis('off')

            axes[1].imshow(embedded_rgb)
            axes[1].set_title('Embedded Image')
            axes[1].axis('off')

        plt.tight_layout()
        plt.show()

    def display_modification_position(self):
        if self.channel == 3:
            change_map = [np.mod(np.zeros(self.original_image.shape[:2], dtype=np.int32) - 1, 255),
                          np.mod(np.zeros(self.original_image.shape[:2], dtype=np.int32) - 1, 255),
                          np.mod(np.zeros(self.original_image.shape[:2], dtype=np.int32) - 1, 255)]
            change_num = [0] * self.channel
            kl = [0] * self.channel

            for i in range(self.channel):
                channel_original_image = self.original_image[:, :, i]
                channel_embedded_image = self.embedded_image[:, :, i]
                for y in range(self.original_image.shape[0]):
                    for x in range(self.original_image.shape[1]):
                        if not np.array_equal(channel_original_image[y, x], channel_embedded_image[y, x]):
                            change_map[i][y, x] = 0

                changed_pixels = np.sum(change_map[i] == 0)
                total_pixels = self.original_image.shape[0] * self.original_image.shape[1]
                change_num[i] = (changed_pixels / total_pixels) * 100

                hist_original, _ = np.histogram(channel_original_image, bins=256, range=(0, 255), density=True)
                hist_embedded, _ = np.histogram(channel_embedded_image, bins=256, range=(0, 255), density=True)
                hist_original = np.clip(hist_original, 1e-10, None)
                hist_embedded = np.clip(hist_embedded, 1e-10, None)
                kl[i] = entropy(hist_original, hist_embedded)

            plt.figure(figsize=(6, 4), num='Red')
            # plt.subplot(131)
            plt.imshow(change_map[0], cmap='gray', vmin=0, vmax=255)
            plt.title("Red, {}, KL:{}".format(change_num[0], kl[0]))

            plt.figure(figsize=(6, 4), num='Green')
            # plt.subplot(132)
            plt.imshow(change_map[1], cmap='gray', vmin=0, vmax=255)
            plt.title("Green, {}, KL:{}".format(change_num[1], kl[1]))

            plt.figure(figsize=(6, 4), num='Blue')
            # plt.subplot(133)
            plt.imshow(change_map[2], cmap='gray', vmin=0, vmax=255)
            plt.title("Blue, {}, KL:{}".format(change_num[2], kl[2]))

        else:
            change_map = np.mod(np.zeros(self.original_image.shape[:2], dtype=np.int32) - 1, 255)
            change_num = 0
            kl = 0

            for i in range(self.channel):
                channel_original_image = self.original_image
                channel_embedded_image = self.embedded_image
                for y in range(self.original_image.shape[0]):
                    for x in range(self.original_image.shape[1]):
                        if not np.array_equal(channel_original_image[y, x], channel_embedded_image[y, x]):
                            change_map[y, x] = 0

                changed_pixels = np.sum(change_map == 255)
                total_pixels = self.original_image.shape[0] * self.original_image.shape[1]
                change_num = (changed_pixels / total_pixels) * 100

                hist_original, _ = np.histogram(channel_original_image, bins=256, range=(0, 255), density=True)
                hist_embedded, _ = np.histogram(channel_embedded_image, bins=256, range=(0, 255), density=True)
                hist_original = np.clip(hist_original, 1e-10, None)
                hist_embedded = np.clip(hist_embedded, 1e-10, None)
                kl = entropy(hist_original, hist_embedded)

            plt.figure(figsize=(10, 6))
            plt.imshow(change_map, cmap='gray', vmin=0, vmax=255)
            plt.title("Gray, {}, KL:{}".format(change_num, kl))
            plt.axis('off')

        plt.show()
        pass

    def metric_mse(self):
        mse = np.mean((self.original_image - self.embedded_image) ** 2)
        print("MSE: ", mse)
        return mse

    def metric_psnr(self):
        mse = np.mean((self.original_image - self.embedded_image) ** 2)
        if mse == 0:
            psnr = float('inf')
        else:
            max_pixel = 255.0
            psnr = 20 * np.log10(max_pixel / np.sqrt(mse))

        print("PSNR: ", psnr)
        return psnr

    def metric_ssim(self):
        ssim_score, _ = ssim(self.original_image, self.embedded_image, multichannel=True, channel_axis=2, full=True)
        print("SSIM: ", ssim_score)
        return ssim_score

    def metric_kl_divergence(self):
        hist_original, _ = np.histogram(self.original_image.flatten(), bins=256, range=(0, 255), density=True)
        hist_embedded, _ = np.histogram(self.embedded_image.flatten(), bins=256, range=(0, 255), density=True)
        hist_original = np.clip(hist_original, 1e-10, None)
        hist_embedded = np.clip(hist_embedded, 1e-10, None)
        kl_divergence = entropy(hist_original, hist_embedded)
        print("KL divergence: ", kl_divergence)
        return kl_divergence


class Steganalysis:
    def __init__(self, original_image, embedded_image, show_result_picture=True):
        self.original_image = original_image
        self.embedded_image = embedded_image
        _, _, *self.channel = self.original_image.shape
        if self.channel:
            self.channel = 3
        else:
            self.channel = 1
        self.cnt = 1
        self.show_result_picture = show_result_picture

    def lsb_distribution_analysis(self, threshold=0.1):
        print("LSB analysis")

        def cal_distribution(image):
            lsb_values = (image.flatten() & 1).astype(np.int32)
            count_0 = np.sum(lsb_values == 0)
            count_1 = np.sum(lsb_values == 1)
            total = count_0 + count_1
            percent_0 = count_0 / total
            percent_1 = count_1 / total
            return percent_0, percent_1

        original_image_0, original_image_1 = cal_distribution(self.original_image)
        embedded_image_0, embedded_image_1 = cal_distribution(self.embedded_image)

        print("Original image: \n 0bit {}, 1bit {}".format(original_image_0, original_image_1))
        print("Embedding image: \n 0bit {}, 1bit {}".format(embedded_image_0, embedded_image_1))

        is_stego = False
        if abs(embedded_image_0 - embedded_image_1) > threshold:
            is_stego = True

        if self.show_result_picture:
            plt.figure(figsize=(5, 4), num="Original image")
            # plt.subplot(1, 2, 1)
            plt.bar(['0', '1'], [original_image_0, original_image_1], color=['#F8B9A8', '#AEB899'])
            plt.title("Original image")
            plt.xlabel('LSB')
            plt.ylabel('LSB ratio')
            plt.grid(axis='y', linestyle='--', alpha=0.7)

            plt.figure(figsize=(5, 4), num="Embedded image")
            # plt.subplot(1, 2, 2)
            plt.bar(['0', '1'], [embedded_image_0, embedded_image_1], color=['#F8B9A8', '#AEB899'])
            plt.title("Embedded image")
            plt.xlabel('LSB')
            plt.ylabel('LSB ratio')
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.show()
        return is_stego

    def distribution_analysis(self):
        def image_distribution(image):
            if self.channel == 3:
                img_array = np.array(image)
                bins = 256
                channels = ['Red', 'Green', 'Blue']
                colors = ['#B55489', '#4C6C43', '#3FA0C0']
                plt.figure(figsize=(16, 4), num="Image distribution - {}".format(self.cnt))
                self.cnt += 1
                for i, channel in enumerate(channels):
                    channel_data = img_array[..., i].flatten()
                    plt.subplot(1, 3, i + 1)
                    hist, bins = np.histogram(channel_data, bins=bins, range=(0, 255))
                    bin_centers = (bins[:-1] + bins[1:]) / 2
                    plt.bar(bin_centers, hist, width=1, color=colors[i])
                    plt.ylabel('Count')
                    plt.title(f'{channel} Channel')
                    plt.xlabel('Pixel Value')
                    plt.grid(axis='y', alpha=0.3)
                    plt.xlim(0, 255)
                    plt.text(0.98, 0.95,
                             f'Mean: {channel_data.mean():.1f}\nStd: {channel_data.std():.1f}',
                             transform=plt.gca().transAxes,
                             ha='right', va='top',
                             bbox=dict(facecolor='white', alpha=0.8))
                # plt.subplot(2, 2, 4)
                plt.figure(figsize=(6, 4), num="Image distribution - {}".format(self.cnt))
                for i, channel in enumerate(channels):
                    channel_data = img_array[..., i].flatten()
                    hist, bins = np.histogram(channel_data, bins=bins, range=(0, 255))
                    bin_centers = (bins[:-1] + bins[1:]) / 2
                    plt.plot(bin_centers, hist, color=colors[i], label=channel)
                    plt.ylabel('Count')
                    pass
                plt.title('All Channels')
                plt.xlabel('Pixel Value')
                plt.grid(alpha=0.3)
                plt.xlim(0, 255)
                plt.legend()
                pass
            else:
                img_array = np.array(image)
                bins = 256
                plt.figure(figsize=(8, 6), num="Image distribution - {}".format(self.cnt))
                self.cnt += 1

                channel_data = img_array.flatten()
                hist, bins = np.histogram(channel_data, bins=bins, range=(0, 255))
                bin_centers = (bins[:-1] + bins[1:]) / 2
                plt.bar(bin_centers, hist, width=1)
                plt.ylabel('Count')
                plt.xlabel('Pixel Value')
                plt.grid(axis='y', alpha=0.3)
                plt.xlim(0, 255)
                plt.text(0.98, 0.95,
                         f'Mean: {channel_data.mean():.1f}\nStd: {channel_data.std():.1f}',
                         transform=plt.gca().transAxes,
                         ha='right', va='top',
                         bbox=dict(facecolor='white', alpha=0.8))
            plt.show()

        image_distribution(self.original_image)
        image_distribution(self.embedded_image)

    def relativity_and_entropy_analysis(self):
        print("entropy analysis：")
        diff_bg = np.mean(
            np.abs(self.embedded_image[:, :, 0].astype(np.float32) - self.embedded_image[:, :, 1].astype(np.float32)))
        diff_gr = np.mean(
            np.abs(self.embedded_image[:, :, 1].astype(np.float32) - self.embedded_image[:, :, 2].astype(np.float32)))

        lsb_plane = self.embedded_image & 1
        entropy = -np.sum(lsb_plane * np.log2(lsb_plane + 1e-10))

        if diff_bg < 5 or diff_gr < 5 or entropy > 0.9:
            return True
        else:
            return False

    def rs_analysis(self, threshold=0.05):
        print("RS analysis: ")

        def f(g):
            return abs(g[1] - g[0]) - abs(g[1] - g[2])

        def cal_groups(pixels):
            groups = []
            for i in range(1, len(pixels) - 1, 2):
                groups.append((pixels[i - 1], pixels[i], pixels[i + 1]))
            return groups

        pixels_origin = self.embedded_image.flatten()
        groups_origin = cal_groups(pixels_origin)
        r_m = sum(1 for g in groups_origin if f(g) > 0) / len(groups_origin)
        s_m = sum(1 for g in groups_origin if f(g) < 0) / len(groups_origin)

        pixels_copy = pixels_origin.copy()
        pixels_copy[1::2] = 255 - pixels_copy[1::2]
        groups_copy = cal_groups(pixels_copy)
        r_f = sum(1 for g in groups_copy if f(g) > 0) / len(groups_copy)
        s_f = sum(1 for g in groups_copy if f(g) < 0) / len(groups_copy)

        delta_r = r_m - r_f
        delta_s = s_m - s_f

        embedding_ratio = np.abs(delta_r) / 2
        is_stego = (np.abs(delta_r) > threshold) or (np.abs(delta_s) > threshold)
        print("embedding ratio {}, is stego {}".format(embedding_ratio, is_stego))

        if self.show_result_picture:
            plt.figure(figsize=(6, 4), num="RS Analysis")
            plt.bar(['R_M', 'S_M', 'R_F1', 'ΔR'], [r_m, s_m, r_f, delta_r], color=['blue', 'red', 'blue', 'green'])
            plt.axhline(y=0, color='k', linestyle='--')
            plt.title(f"RS Analysis: p={embedding_ratio:.4f}, Stego={is_stego}")
            plt.show()

        # return embedding_ratio, is_stego
        return is_stego

    def rqp_analysis(self, embed_rate=0.1):
        print("RQP analysis:")
        pixels = self.embedded_image.flatten()
        unique_colors, counts = np.unique(pixels, axis=0, return_counts=True)
        total_pairs = len(unique_colors) * (len(unique_colors) - 1) // 2

        adjacent_pairs = 0
        for i in range(len(unique_colors)):
            for j in range(i + 1, len(unique_colors)):
                if np.all(np.abs(unique_colors[i] - unique_colors[j]) <= 1):
                    adjacent_pairs += counts[i] * counts[j]
        Q1 = adjacent_pairs / total_pairs if total_pairs > 0 else 0

        pixels_copy = pixels.copy()
        mask = np.random.rand(len(pixels_copy)) < embed_rate
        pixels_copy[mask] ^= 1

        unique_reverse, counts_reverse = np.unique(pixels_copy, axis=0, return_counts=True)
        adjacent_pairs_reverse = 0
        for i in range(len(unique_reverse)):
            for j in range(i + 1, len(unique_reverse)):
                if np.all(np.abs(unique_reverse[i] - unique_reverse[j]) <= 1):
                    adjacent_pairs_reverse += counts_reverse[i] * counts_reverse[j]
        total_pairs_reverse = len(unique_reverse) * (len(unique_reverse) - 1) // 2
        Q2 = adjacent_pairs_reverse / total_pairs_reverse if total_pairs_reverse > 0 else 0

        R = Q2 / Q1 if Q1 > 0 else float('inf')
        is_stego = 0.8 < R < 1.2

        print("R value {}, is stego {}".format(R, is_stego))
        return is_stego

    def spa_analysis(self):
        print("SPA analysis:")
        pixels = self.embedded_image.flatten()

        cnt_equal, cnt_diff_1, cnt_trans = 0, 0, 0
        for i in range(0, len(pixels) - 1, 2):
            if pixels[i] == pixels[i + 1]:
                cnt_equal += 1
            elif abs(pixels[i] - pixels[i + 1]) == 1:
                cnt_diff_1 += 1
            if pixels[i] % 2 == pixels[i + 1] % 2:
                cnt_trans += 1

        denominator = cnt_equal + cnt_diff_1 - 2 * cnt_trans
        p = 1 - np.sqrt(cnt_trans / denominator) if denominator > 0 else 0
        print("Steganography probability {}".format(p))
        return p


def analyze_folder(folder_path, analysis_method=1):
    image_files = [f for f in os.listdir(folder_path)
                   if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    image_files.sort()
    true_labels = []
    predictions = []
    print(f"{len(image_files)} images ")
    print("-" * 60)
    for img_file in image_files:
        img_path = os.path.join(folder_path, img_file)
        to_analyze_img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED).astype(np.int32)
        true_label = 1 if len(img_file) > 17 else 0
        steganalysis = Steganalysis(to_analyze_img, to_analyze_img, show_result_picture=False)
        if analysis_method == 1:
            predict_result = steganalysis.lsb_distribution_analysis()
        elif analysis_method == 2:
            predict_result = steganalysis.rs_analysis()
        elif analysis_method == 3:
            predict_result = steganalysis.rqp_analysis()
        elif analysis_method == 4:
            predict_result = steganalysis.relativity_and_entropy_analysis()
        else:
            raise ValueError("This method does not exist, only 1,2,3,4")
        true_labels.append(true_label)
        predictions.append(predict_result)

        # 打印当前图片分析结果
        status = "Correct" if predict_result == true_label else "False"
        print(f"Image: {img_file:<20} | True: {true_label} | Predict: {predict_result} | {status}")

    print("-" * 60)

    if len(true_labels) > 0:
        accuracy = accuracy_score(true_labels, predictions)
        tn, fp, fn, tp = confusion_matrix(true_labels, predictions).ravel()
        total = len(true_labels)
        false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
        false_negative_rate = fn / (fn + tp) if (fn + tp) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        # 打印统计结果
        print("\n Analysis result:")
        print(f"Total pictures: {total}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"False Positive Rate: {false_positive_rate:.4f}")
        print(f"False Negative Rate: {false_negative_rate:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"TN: {tn} | FP: {fp}")
        print(f"FN: {fn} | TP: {tp}")
        return true_labels, predictions
    else:
        print("No valid pictures found")
        return None


def plot_roc_with_probabilities(y_true, y_prob, windows_title="ROC"):
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 5), num=windows_title)
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'ROC (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Reference Line')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=14)
    plt.legend(loc="lower right", fontsize=12)
    plt.grid(True, alpha=0.3)
    # Mark the best threshold (the point closest to the upper left corner)
    gmeans = np.sqrt(tpr * (1 - fpr))
    ix = np.argmax(gmeans)
    plt.scatter(fpr[ix], tpr[ix], marker='o', color='red', s=100,
                label=f'best thresholds={thresholds[ix]:.2f}')
    plt.show()
    return fpr, tpr, roc_auc


if __name__ == "__main__":
    #################################
    # root = "chaos_img"
    root = "stc_img"
    original_image_path = "{}/3.png".format(root)
    embedded_image_path = "{}/3_chaos_stc_h8_w10_msg400000.png".format(root)
    #################################
    print("############# Comparison between original image and steganography #############")
    compare = Comparison(cv2.imread(original_image_path, cv2.IMREAD_UNCHANGED),
                         cv2.imread(embedded_image_path, cv2.IMREAD_UNCHANGED))
    compare.information()
    compare.display_image()
    compare.display_modification_position()
    compare.metric_mse()
    compare.metric_psnr()
    compare.metric_ssim()
    compare.metric_kl_divergence()

    print("############# Steganalysis of steganography #############")
    steganalysis = Steganalysis(cv2.imread(original_image_path, cv2.IMREAD_UNCHANGED).astype(np.int32),
                                cv2.imread(embedded_image_path, cv2.IMREAD_UNCHANGED).astype(np.int32),
                                show_result_picture=True)
    steganalysis.lsb_distribution_analysis()
    steganalysis.distribution_analysis()
    steganalysis.rs_analysis()
    steganalysis.rqp_analysis()
    steganalysis.spa_analysis()

    print("############# Steganalysis #############")
    """1: LSB; 2: RS; 3: RQP; 4: Entropy"""
    true, pred = analyze_folder(root, analysis_method=2)
    plot_roc_with_probabilities(true, pred, "ROC_RS")
