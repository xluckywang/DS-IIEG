import os
import numpy as np
import torch
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from classification import (Classification, cvtColor, letterbox_image,
                            preprocess_input)
from utils.utils import letterbox_image
from utils.utils_metrics import evaluteTop1_5

# ------------------------------------------------------#
#   test_annotation_path    Test image path and labels
# ------------------------------------------------------#
test_annotation_path = 'cls_test.txt'
# ------------------------------------------------------#
#   metrics_out_path        Folder for saving indicators
# ------------------------------------------------------#
metrics_out_path = "results_eval/Ori/MTD/Vgg/metrics_out"


class Eval_Classification(Classification):
    def detect_image(self, image):
        # ---------------------------------------------------------#
        # Convert the image to an RGB image here to prevent errors in predicting grayscale images.
        # The code only supports prediction of RGB images, all other types of images will be converted to RGB
        # ---------------------------------------------------------#
        image = cvtColor(image)
        # ---------------------------------------------------#
        #   Resize the image without distortion
        # ---------------------------------------------------#
        image_data = letterbox_image(image, [self.input_shape[1], self.input_shape[0]], self.letterbox_image)
        # ---------------------------------------------------------#
        #   Normalize+add batch_2 dimension+transpose
        # ---------------------------------------------------------#
        image_data = np.transpose(np.expand_dims(preprocess_input(np.array(image_data, np.float32)), 0), (0, 3, 1, 2))

        with torch.no_grad():
            photo = torch.from_numpy(image_data).type(torch.FloatTensor)
            if self.cuda:
                photo = photo.cuda()
            # ---------------------------------------------------#
            #   Image input to the network for prediction
            # ---------------------------------------------------#
            preds = torch.softmax(self.model(photo)[0], dim=-1).cpu().numpy()

        return preds


def calculate_f1(precision, recall):
    """Calculate F1 score"""
    return 2 * (precision * recall) / (precision + recall)


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt=".2f" if normalize else "d", cmap=cmap,
                xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    return plt


if __name__ == "__main__":
    if not os.path.exists(metrics_out_path):
        os.makedirs(metrics_out_path)

    classfication = Eval_Classification()

    with open(test_annotation_path, "r") as f:
        lines = f.readlines()
    top1, top5, Recall, Precision, lab, pre = evaluteTop1_5(classfication, lines, metrics_out_path)

    # Calculate F1 score
    mean_precision = np.mean(Precision)
    mean_recall = np.mean(Recall)
    f1_score = calculate_f1(mean_precision, mean_recall)

    # Generate confusion matrix
    cm = confusion_matrix(lab, pre)
    class_names = [str(i) for i in range(len(np.unique(lab)))]  # Assuming the category is a continuous number starting from 0

    # Calculate the accuracy of each category
    class_accuracy = {}
    for i in range(len(class_names)):
        true_positives = cm[i, i]
        total_samples = np.sum(cm[i, :])
        class_accuracy[f"Class {i}"] = f"{true_positives}/{total_samples} = {true_positives / total_samples:.2%}"

    # Print results
    print("top-1 accuracy = %.2f%%" % (top1 * 100))
    print("mean Recall = %.2f%%" % (mean_recall * 100))
    print("mean Precision = %.2f%%" % (mean_precision * 100))
    print("F1 score = %.2f%%" % (f1_score * 100))

    print("\nConfusion Matrix:")
    print(cm)

    print("\nNormalized Confusion Matrix (row-wise):")
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print(cm_normalized)

    print("\nPer-class Accuracy:")
    for cls, acc in class_accuracy.items():
        print(f"{cls}: {acc}")

    # Save the results to a txt file
    result_txt_path = os.path.join(metrics_out_path, "evaluation_results.txt")
    with open(result_txt_path, "w") as f:
        f.write("top-1 accuracy = %.2f%%\n" % (top1 * 100))
        f.write("mean Recall = %.2f%%\n" % (mean_recall * 100))
        f.write("mean Precision = %.2f%%\n" % (mean_precision * 100))
        f.write("F1 score = %.2f%%\n" % (f1_score * 100))

        f.write("\nConfusion Matrix:\n")
        np.savetxt(f, cm, fmt='%d')

        f.write("\nNormalized Confusion Matrix (row-wise):\n")
        np.savetxt(f, cm_normalized, fmt='%.4f')

        f.write("\nPer-class Accuracy:\n")
        for cls, acc in class_accuracy.items():
            f.write(f"{cls}: {acc}\n")

    # Save confusion matrix image
    plt = plot_confusion_matrix(cm, class_names, title='Confusion Matrix')
    plt.savefig(os.path.join(metrics_out_path, 'confusion_matrix.png'))
    plt.close()

    plt = plot_confusion_matrix(cm, class_names, normalize=True, title='Normalized Confusion Matrix')
    plt.savefig(os.path.join(metrics_out_path, 'normalized_confusion_matrix.png'))
    plt.close()