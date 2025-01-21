import os
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torch


def display_adv(paras, attack_method):

    adv_folder_path = '../adv_imgs/' + paras.model + '/' + paras.dataset + '/' + attack_method + '/'
    org_folder_path = '../org_imgs/' + paras.model + '/' + paras.dataset + '/'
    adv_file_list = [file for file in os.listdir(adv_folder_path) if file.startswith('show_all')]
    org_file_list = [file for file in os.listdir(org_folder_path)]
    adv_file_list.sort(key=lambda x: int(x[9: -4]))
    org_file_list.sort(key=lambda x: int(x[8: -4]))
    adv = []
    org = []

    for i, f in enumerate(adv_file_list):
        adv.extend(np.load(adv_folder_path + f))

    for i, f in enumerate(org_file_list):
        org.extend(np.load(org_folder_path + f))

    adv = np.asarray(adv)
    org = np.asarray(org)
    adv = adv[: len(org)]
    dl1 = torch.utils.data.DataLoader(adv, batch_size=32, shuffle=False, num_workers=4)
    dl2 = torch.utils.data.DataLoader(org, batch_size=32, shuffle=False, num_workers=4)
    dl2 = tqdm(dl2, desc='ADV')

    norm = []
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    count = -1

    with torch.no_grad():
        for batch_adv_x, batch_test_x in zip(dl1, dl2):
            batch_adv_x = batch_adv_x.cpu().numpy()
            batch_test_x = batch_test_x.cpu().numpy()

            batch_r = (batch_adv_x - batch_test_x)
            batch_norm = [np.linalg.norm(r) for r in batch_r]
            norm.extend(batch_norm)

            for i in range(len(batch_adv_x)):
                count += 1
                org_img = batch_test_x[i]
                adv_img = batch_adv_x[i]
                org_img = np.uint8(org_img * 255)
                adv_img = np.uint8(adv_img * 255)
                org_img = org_img.transpose((1, 2, 0))
                adv_img = adv_img.transpose((1, 2, 0))
                org_img = Image.fromarray(org_img)
                adv_img = Image.fromarray(adv_img)

                axes[0].imshow(org_img)
                axes[0].set_title('Original Image')
                axes[0].axis('off')

                axes[1].imshow(adv_img)
                axes[1].set_title('Adversarial Image')
                axes[1].axis('off')

                plt.tight_layout()
                plt.savefig('/home/lyj/LLMs_Multi_Labels/src/results/origin_and_adv/'
                            + attack_method + '/compare_{}'.format(count))

        print(np.mean(norm))


def display_adv_compare():
    org_image_folder = f'/home/lyj/LLMs_Multi_Labels/src/results/org_images/image'
    gpt_image_folder = f'/home/lyj/LLMs_Multi_Labels/src/results/adv_images/gpt/image'
    bim_folder = f'/home/lyj/LLMs_Multi_Labels/src/results/adv_images/i_fgsm/image'
    pgd_image_folder = f'/home/lyj/LLMs_Multi_Labels/src/results/adv_images/pgd/image'
    mifgsm_image_folder = f'/home/lyj/LLMs_Multi_Labels/src/results/adv_images/mi_fgsm/image'
    image_path = [org_image_folder, bim_folder, pgd_image_folder, mifgsm_image_folder, gpt_image_folder]

    fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(14, 6))
    title = ['Original', 'BIM', 'PGD', 'MI-FGSM', 'LGA']
    index = [17, 87]

    images = [[Image.open(path + str(idx) + '.jpg') for path in image_path] for idx in index]

    for row in range(2):
        for col in range(5):
            image = images[row][col]
            axes[row][col].imshow(image)
            axes[row][col].axis('off')

    title_fontdict = {'fontsize': 20, 'fontweight': 'bold'}
    for ax, col in zip(axes[0], title):
        ax.set_title(col, fontdict=title_fontdict)
    plt.tight_layout()
    plt.show()
    # plt.savefig('/home/lyj/LLMs_Multi_Labels/src/results/origin_and_adv/compare_image.pdf')


def display_single_image(paras, attack_method):
    adv_folder_path = '../adv_imgs/' + paras.model + '/' + paras.dataset + '/' + attack_method + '/'
    org_folder_path = '../org_imgs/' + paras.model + '/' + paras.dataset + '/'
    adv_file_list = [file for file in os.listdir(adv_folder_path) if file.startswith('show_all')]
    org_file_list = [file for file in os.listdir(org_folder_path)]
    adv_file_list.sort(key=lambda x: int(x[9: -4]))
    org_file_list.sort(key=lambda x: int(x[8: -4]))
    adv = []
    org = []
    index = 17

    for i, f in enumerate(adv_file_list):
        adv.extend(np.load(adv_folder_path + f))

    for i, f in enumerate(org_file_list):
        org.extend(np.load(org_folder_path + f))

    adv = np.asarray(adv)
    org = np.asarray(org)
    # adv = adv[: len(org)]
    adv_image = adv[index:index + 1]
    org_image = org[index:index + 1]
    adv_image = np.squeeze(adv_image, axis=0)
    org_image = np.squeeze(org_image, axis=0)
    r = adv_image - org_image

    org_img = np.uint8(org_image * 255)
    adv_img = np.uint8(adv_image * 255)
    org_img = org_img.transpose((1, 2, 0))
    adv_img = adv_img.transpose((1, 2, 0))
    org_img = Image.fromarray(org_img)
    adv_img = Image.fromarray(adv_img)

    org_img.save(f'/home/lyj/LLMs_Multi_Labels/src/results/org_images/image{index}.jpg')
    adv_img.save(f'/home/lyj/LLMs_Multi_Labels/src/results/adv_images/{attack_method}/image{index}.jpg')

    l2_norm = np.linalg.norm(r)
    print(f'对抗扰动的l2范数为{l2_norm}')

# if __name__ == '__main__':
#     path_folder = '/home/lyj/LLMs_Multi_Labels/src/results/origin_and_adv/image'
#
#     fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(25, 10))
#     index = [17, 87]
#     for ax, idx in zip(axes, index):
#         image = Image.open(path_folder + str(idx) + '.png')
#         ax.imshow(image)
#         ax.axis('off')
#     plt.tight_layout()
#     plt.show()

