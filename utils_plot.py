import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import os
import gc
from tqdm import tqdm
import pickle
import random
import copy
import torch.utils.data
from PIL import Image, ImageDraw
import pickle
from utils import individual, gene

def pickle_dump(data, path):
    with open(path, 'wb') as fp:
        pickle.dump(data, fp)

def pickle_load(path):
    with open(path, 'rb') as fp:
        return pickle.load(fp)

# plot the fitness value graph of best individuals over generations, 1 to 10000
def plot_fitness_all(directory, save_dir_fit):
    # load the data
    best_indvs_fit = pickle_load(directory + 'all_best_individuals.pickle')
    #print("size: ", str(len(best_indvs_fit)))

    # generate x-axis values as indices of the data array
    x = range(len(best_indvs_fit))

    str_fitness = []
    # store the fitness
    for i in range(len(best_indvs_fit)):
        if i != 0 and best_indvs_fit[i].fitness < best_indvs_fit[i-1].fitness:
            best_indvs_fit[i].fitness = best_indvs_fit[i-1].fitness
        str_fitness.append(best_indvs_fit[i].fitness) 

    # plot the data
    plt.plot(x, str_fitness)

    # set x-axis and y-axis labels
    plt.xlabel('Generations')
    plt.ylabel('Fitness values')

    # set the subplot title
    plt.suptitle('Fitness plot from generation 1 to 10000 ')

    if not os.path.exists(save_dir_fit):
        os.makedirs(save_dir_fit)
    # save the plot
    plt.savefig(save_dir_fit + 'fitness_1_10000')
    cv.destroyAllWindows()
    plt.clf()
    plt.close()

    # show the plot
    #plt.show()

# plot the fitness value graph of best individuals over generations, 1000 to 10000
def plot_fitness_part(directory, save_dir_fit):
    # load the data
    best_indvs_fit = pickle_load(directory + 'all_best_individuals.pickle')
    #print("size: ", str(len(best_indvs_fit)))

    # generate x-axis values as indices of the data array
    #x = range(len(best_indvs_fit[1000:]))
    start_num = 1001
    end_num = 10000
    # create an array using a list comprehension
    x = [num for num in range(start_num, end_num + 1)]

    str_fitness = []
    # store the fitness
    for i in range(len(x)):
        if i != 0 and best_indvs_fit[i+999].fitness < best_indvs_fit[i+998].fitness:
            best_indvs_fit[i+999].fitness = best_indvs_fit[i+998].fitness
        str_fitness.append(best_indvs_fit[i+999].fitness) 

    # plot the data
    plt.plot(x, str_fitness)

    # set x-axis and y-axis labels
    plt.xlabel('Generations')
    plt.ylabel('Fitness values')

    # set the subplot title
    plt.suptitle('Fitness plot from generation 1000 to 10000 ')

    if not os.path.exists(save_dir_fit):
        os.makedirs(save_dir_fit)
    # save the plot
    plt.savefig(save_dir_fit + 'fitness_1000_10000')
    plt.clf()
    plt.close()

    # show the plot
    #plt.show()

# combine the images
def combine_imgs(directory, save_dir):
    # get the list of image filenames in the directory
    image_files = sorted([file for file in os.listdir(directory) if file.endswith(".png")])

    # number of rows and columns for the subplot grid
    num_rows = 4
    num_cols = 3

    # create a new figure and set the size
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 9))

    # iterate over the image files and plot them in subplots
    for i, image_file in enumerate(image_files):
        # open the image file using PIL
        image_path = os.path.join(directory, image_file)
        image = plt.imread(image_path)

        # calculate the row and column index for the current subplot
        row = i // num_cols
        col = i % num_cols

        # plot the image in the corresponding subplot
        axes[row, col].imshow(image)

        # set the axes limits to the size of the image
        axes[row, col].set_xlim(0, image.shape[0])
        axes[row, col].set_ylim(0, image.shape[1])

        # set the subplot title
        if i == 0:
            axes[row, col].set_title('Generation ' + str(i+1))
        else:
            axes[row, col].set_title('Generation ' + str(i*1000))

    num_images = len(image_files)

    # remove the extra empty subplots if necessary
    if num_images < num_rows * num_cols:
        for i in range(num_images, num_rows * num_cols):
            row = i // num_cols
            col = i % num_cols
            fig.delaxes(axes[row, col])

    # adjust the spacing between subplots
    plt.tight_layout()

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    plt.savefig(save_dir + 'combined')
    plt.clf()
    plt.close()

    # show the combined plot
    #plt.show()

def get_best_of_bests(directory_def, directory1, directory2, directory3, directory4):
    # load the data
    best_indvs_fit_def = pickle_load(directory_def + 'all_best_individuals.pickle')
    best_indvs_fit_1 = pickle_load(directory1 + 'all_best_individuals.pickle')
    best_indvs_fit_2 = pickle_load(directory2 + 'all_best_individuals.pickle')
    best_indvs_fit_3 = pickle_load(directory3 + 'all_best_individuals.pickle')
    best_indvs_fit_4 = pickle_load(directory4 + 'all_best_individuals.pickle')

    # print the values
    print("for dir_def: ", str(best_indvs_fit_def[-1].fitness))
    print("for dir_1: ", str(best_indvs_fit_1[-1].fitness))
    print("for dir_2: ", str(best_indvs_fit_2[-1].fitness))
    print("for dir_3: ", str(best_indvs_fit_3[-1].fitness))
    print("for dir_4: ", str(best_indvs_fit_4[-1].fitness))

if __name__ == '__main__':
    directory_img = 'D:/Ercihan/Deconvolution/Deconvolution/49/images/trial/'
    directory_fit = 'D:/Ercihan/Deconvolution/Deconvolution/49/indvs/trial/'
    save_dir = 'D:/Ercihan/Deconvolution/Deconvolution/49/plots/combined/'
    save_dir_fit = 'D:/Ercihan/Deconvolution/Deconvolution/49/plots/fitness/'

    directory_def = 'D:/Ercihan/Deconvolution/Deconvolution/49/indvs/suggestion_1/'
    directory1 = 'D:/Ercihan/Deconvolution/Deconvolution/49/indvs/suggestion_2/'
    directory2 = 'D:/Ercihan/Deconvolution/Deconvolution/49/indvs/mutation_prob_0_75/'
    directory3 = 'D:/Ercihan/Deconvolution/Deconvolution/49/indvs/num_genes_120/'
    directory4 = 'D:/Ercihan/Deconvolution/Deconvolution/49/indvs/num_inds_60/'

    # get_best_of_bests(directory_def, directory1, directory2, directory3, directory4)

    ### suggestions ###
    combine_imgs('D:/Ercihan/Deconvolution/Deconvolution/49/images/suggestion_1/', 'D:/Ercihan/Deconvolution/Deconvolution/49/plots/combined/suggestion_1/')
    plot_fitness_all('D:/Ercihan/Deconvolution/Deconvolution/49/indvs/suggestion_1/', 'D:/Ercihan/Deconvolution/Deconvolution/49/plots/fitness/suggestion_1/')
    plot_fitness_part('D:/Ercihan/Deconvolution/Deconvolution/49/indvs/suggestion_1/', 'D:/Ercihan/Deconvolution/Deconvolution/49/plots/fitness_part/suggestion_1/')

    combine_imgs('D:/Ercihan/Deconvolution/Deconvolution/49/images/suggestion_2/', 'D:/Ercihan/Deconvolution/Deconvolution/49/plots/combined/suggestion_2/')
    plot_fitness_all('D:/Ercihan/Deconvolution/Deconvolution/49/indvs/suggestion_2/', 'D:/Ercihan/Deconvolution/Deconvolution/49/plots/fitness/suggestion_2/')
    plot_fitness_part('D:/Ercihan/Deconvolution/Deconvolution/49/indvs/suggestion_2/', 'D:/Ercihan/Deconvolution/Deconvolution/49/plots/fitness_part/suggestion_2/')

    combine_imgs('D:/Ercihan/Deconvolution/Deconvolution/49/images/suggestion_3/', 'D:/Ercihan/Deconvolution/Deconvolution/49/plots/combined/suggestion_3/')
    plot_fitness_all('D:/Ercihan/Deconvolution/Deconvolution/49/indvs/suggestion_3/', 'D:/Ercihan/Deconvolution/Deconvolution/49/plots/fitness/suggestion_3/')
    plot_fitness_part('D:/Ercihan/Deconvolution/Deconvolution/49/indvs/suggestion_3/', 'D:/Ercihan/Deconvolution/Deconvolution/49/plots/fitness_part/suggestion_3/')

    ### num_inds trials ###
    combine_imgs('D:/Ercihan/Deconvolution/Deconvolution/49/images/num_inds_5/', 'D:/Ercihan/Deconvolution/Deconvolution/49/plots/combined/num_inds_5/')
    plot_fitness_all('D:/Ercihan/Deconvolution/Deconvolution/49/indvs/num_inds_5/', 'D:/Ercihan/Deconvolution/Deconvolution/49/plots/fitness/num_inds_5/')
    plot_fitness_part('D:/Ercihan/Deconvolution/Deconvolution/49/indvs/num_inds_5/', 'D:/Ercihan/Deconvolution/Deconvolution/49/plots/fitness_part/num_inds_5/')

    combine_imgs('D:/Ercihan/Deconvolution/Deconvolution/49/images/num_inds_10/', 'D:/Ercihan/Deconvolution/Deconvolution/49/plots/combined/num_inds_10/')
    plot_fitness_all('D:/Ercihan/Deconvolution/Deconvolution/49/indvs/num_inds_10/', 'D:/Ercihan/Deconvolution/Deconvolution/49/plots/fitness/num_inds_10/')
    plot_fitness_part('D:/Ercihan/Deconvolution/Deconvolution/49/indvs/num_inds_10/', 'D:/Ercihan/Deconvolution/Deconvolution/49/plots/fitness_part/num_inds_10/')

    combine_imgs('D:/Ercihan/Deconvolution/Deconvolution/49/images/num_inds_20/', 'D:/Ercihan/Deconvolution/Deconvolution/49/plots/combined/num_inds_20/')
    plot_fitness_all('D:/Ercihan/Deconvolution/Deconvolution/49/indvs/num_inds_20/', 'D:/Ercihan/Deconvolution/Deconvolution/49/plots/fitness/num_inds_20/')
    plot_fitness_part('D:/Ercihan/Deconvolution/Deconvolution/49/indvs/num_inds_20/', 'D:/Ercihan/Deconvolution/Deconvolution/49/plots/fitness_part/num_inds_20/')

    combine_imgs('D:/Ercihan/Deconvolution/Deconvolution/49/images/num_inds_40/', 'D:/Ercihan/Deconvolution/Deconvolution/49/plots/combined/num_inds_40/')
    plot_fitness_all('D:/Ercihan/Deconvolution/Deconvolution/49/indvs/num_inds_40/', 'D:/Ercihan/Deconvolution/Deconvolution/49/plots/fitness/num_inds_40/')
    plot_fitness_part('D:/Ercihan/Deconvolution/Deconvolution/49/indvs/num_inds_40/', 'D:/Ercihan/Deconvolution/Deconvolution/49/plots/fitness_part/num_inds_40/')

    combine_imgs('D:/Ercihan/Deconvolution/Deconvolution/49/images/num_inds_60/', 'D:/Ercihan/Deconvolution/Deconvolution/49/plots/combined/num_inds_60/')
    plot_fitness_all('D:/Ercihan/Deconvolution/Deconvolution/49/indvs/num_inds_60/', 'D:/Ercihan/Deconvolution/Deconvolution/49/plots/fitness/num_inds_60/')
    plot_fitness_part('D:/Ercihan/Deconvolution/Deconvolution/49/indvs/num_inds_60/', 'D:/Ercihan/Deconvolution/Deconvolution/49/plots/fitness_part/num_inds_60/')

    ### num_genes trials ###
    combine_imgs('D:/Ercihan/Deconvolution/Deconvolution/49/images/num_genes_15/', 'D:/Ercihan/Deconvolution/Deconvolution/49/plots/combined/num_genes_15/')
    plot_fitness_all('D:/Ercihan/Deconvolution/Deconvolution/49/indvs/num_genes_15/', 'D:/Ercihan/Deconvolution/Deconvolution/49/plots/fitness/num_genes_15/')
    plot_fitness_part('D:/Ercihan/Deconvolution/Deconvolution/49/indvs/num_genes_15/', 'D:/Ercihan/Deconvolution/Deconvolution/49/plots/fitness_part/num_genes_15/')

    combine_imgs('D:/Ercihan/Deconvolution/Deconvolution/49/images/num_genes_30/', 'D:/Ercihan/Deconvolution/Deconvolution/49/plots/combined/num_genes_30/')
    plot_fitness_all('D:/Ercihan/Deconvolution/Deconvolution/49/indvs/num_genes_30/', 'D:/Ercihan/Deconvolution/Deconvolution/49/plots/fitness/num_genes_30/')
    plot_fitness_part('D:/Ercihan/Deconvolution/Deconvolution/49/indvs/num_genes_30/', 'D:/Ercihan/Deconvolution/Deconvolution/49/plots/fitness_part/num_genes_30/')

    combine_imgs('D:/Ercihan/Deconvolution/Deconvolution/49/images/num_genes_50/', 'D:/Ercihan/Deconvolution/Deconvolution/49/plots/combined/num_genes_50/')
    plot_fitness_all('D:/Ercihan/Deconvolution/Deconvolution/49/indvs/num_genes_50/', 'D:/Ercihan/Deconvolution/Deconvolution/49/plots/fitness/num_genes_50/')
    plot_fitness_part('D:/Ercihan/Deconvolution/Deconvolution/49/indvs/num_genes_50/', 'D:/Ercihan/Deconvolution/Deconvolution/49/plots/fitness_part/num_genes_50/')

    combine_imgs('D:/Ercihan/Deconvolution/Deconvolution/49/images/num_genes_80/', 'D:/Ercihan/Deconvolution/Deconvolution/49/plots/combined/num_genes_80/')
    plot_fitness_all('D:/Ercihan/Deconvolution/Deconvolution/49/indvs/num_genes_80/', 'D:/Ercihan/Deconvolution/Deconvolution/49/plots/fitness/num_genes_80/')
    plot_fitness_part('D:/Ercihan/Deconvolution/Deconvolution/49/indvs/num_genes_80/', 'D:/Ercihan/Deconvolution/Deconvolution/49/plots/fitness_part/num_genes_80/')

    combine_imgs('D:/Ercihan/Deconvolution/Deconvolution/49/images/num_genes_120/', 'D:/Ercihan/Deconvolution/Deconvolution/49/plots/combined/num_genes_120/')
    plot_fitness_all('D:/Ercihan/Deconvolution/Deconvolution/49/indvs/num_genes_120/', 'D:/Ercihan/Deconvolution/Deconvolution/49/plots/fitness/num_genes_120/')
    plot_fitness_part('D:/Ercihan/Deconvolution/Deconvolution/49/indvs/num_genes_120/', 'D:/Ercihan/Deconvolution/Deconvolution/49/plots/fitness_part/num_genes_120/')

    ### tm_size trials ###
    combine_imgs('D:/Ercihan/Deconvolution/Deconvolution/49/images/tm_size_2/', 'D:/Ercihan/Deconvolution/Deconvolution/49/plots/combined/tm_size_2/')
    plot_fitness_all('D:/Ercihan/Deconvolution/Deconvolution/49/indvs/tm_size_2/', 'D:/Ercihan/Deconvolution/Deconvolution/49/plots/fitness/tm_size_2/')
    plot_fitness_part('D:/Ercihan/Deconvolution/Deconvolution/49/indvs/tm_size_2/', 'D:/Ercihan/Deconvolution/Deconvolution/49/plots/fitness_part/tm_size_2/')

    combine_imgs('D:/Ercihan/Deconvolution/Deconvolution/49/images/tm_size_5/', 'D:/Ercihan/Deconvolution/Deconvolution/49/plots/combined/tm_size_5/')
    plot_fitness_all('D:/Ercihan/Deconvolution/Deconvolution/49/indvs/tm_size_5/', 'D:/Ercihan/Deconvolution/Deconvolution/49/plots/fitness/tm_size_5/')
    plot_fitness_part('D:/Ercihan/Deconvolution/Deconvolution/49/indvs/tm_size_5/', 'D:/Ercihan/Deconvolution/Deconvolution/49/plots/fitness_part/tm_size_5/')

    combine_imgs('D:/Ercihan/Deconvolution/Deconvolution/49/images/tm_size_8/', 'D:/Ercihan/Deconvolution/Deconvolution/49/plots/combined/tm_size_8/')
    plot_fitness_all('D:/Ercihan/Deconvolution/Deconvolution/49/indvs/tm_size_8/', 'D:/Ercihan/Deconvolution/Deconvolution/49/plots/fitness/tm_size_8/')
    plot_fitness_part('D:/Ercihan/Deconvolution/Deconvolution/49/indvs/tm_size_8/', 'D:/Ercihan/Deconvolution/Deconvolution/49/plots/fitness_part/tm_size_8/')

    combine_imgs('D:/Ercihan/Deconvolution/Deconvolution/49/images/tm_size_16/', 'D:/Ercihan/Deconvolution/Deconvolution/49/plots/combined/tm_size_16/')
    plot_fitness_all('D:/Ercihan/Deconvolution/Deconvolution/49/indvs/tm_size_16/', 'D:/Ercihan/Deconvolution/Deconvolution/49/plots/fitness/tm_size_16/')
    plot_fitness_part('D:/Ercihan/Deconvolution/Deconvolution/49/indvs/tm_size_16/', 'D:/Ercihan/Deconvolution/Deconvolution/49/plots/fitness_part/tm_size_16/')

    ### frac_elites trials ###
    combine_imgs('D:/Ercihan/Deconvolution/Deconvolution/49/images/frac_elites_0_2/', 'D:/Ercihan/Deconvolution/Deconvolution/49/plots/combined/frac_elites_0_2/')
    plot_fitness_all('D:/Ercihan/Deconvolution/Deconvolution/49/indvs/frac_elites_0_2/', 'D:/Ercihan/Deconvolution/Deconvolution/49/plots/fitness/frac_elites_0_2/')
    plot_fitness_part('D:/Ercihan/Deconvolution/Deconvolution/49/indvs/frac_elites_0_2/', 'D:/Ercihan/Deconvolution/Deconvolution/49/plots/fitness_part/frac_elites_0_2/')

    combine_imgs('D:/Ercihan/Deconvolution/Deconvolution/49/images/frac_elites_0_04/', 'D:/Ercihan/Deconvolution/Deconvolution/49/plots/combined/frac_elites_0_04/')
    plot_fitness_all('D:/Ercihan/Deconvolution/Deconvolution/49/indvs/frac_elites_0_04/', 'D:/Ercihan/Deconvolution/Deconvolution/49/plots/fitness/frac_elites_0_04/')
    plot_fitness_part('D:/Ercihan/Deconvolution/Deconvolution/49/indvs/frac_elites_0_04/', 'D:/Ercihan/Deconvolution/Deconvolution/49/plots/fitness_part/frac_elites_0_04/')

    combine_imgs('D:/Ercihan/Deconvolution/Deconvolution/49/images/frac_elites_0_35/', 'D:/Ercihan/Deconvolution/Deconvolution/49/plots/combined/frac_elites_0_35/')
    plot_fitness_all('D:/Ercihan/Deconvolution/Deconvolution/49/indvs/frac_elites_0_35/', 'D:/Ercihan/Deconvolution/Deconvolution/49/plots/fitness/frac_elites_0_35/')
    plot_fitness_part('D:/Ercihan/Deconvolution/Deconvolution/49/indvs/frac_elites_0_35/', 'D:/Ercihan/Deconvolution/Deconvolution/49/plots/fitness_part/frac_elites_0_35/')

    ### frac_parents trials ###
    combine_imgs('D:/Ercihan/Deconvolution/Deconvolution/49/images/frac_parents_0_3/', 'D:/Ercihan/Deconvolution/Deconvolution/49/plots/combined/frac_parents_0_3/')
    plot_fitness_all('D:/Ercihan/Deconvolution/Deconvolution/49/indvs/frac_parents_0_3/', 'D:/Ercihan/Deconvolution/Deconvolution/49/plots/fitness/frac_parents_0_3/')
    plot_fitness_part('D:/Ercihan/Deconvolution/Deconvolution/49/indvs/frac_parents_0_3/', 'D:/Ercihan/Deconvolution/Deconvolution/49/plots/fitness_part/frac_parents_0_3/')

    combine_imgs('D:/Ercihan/Deconvolution/Deconvolution/49/images/frac_parents_0_6/', 'D:/Ercihan/Deconvolution/Deconvolution/49/plots/combined/frac_parents_0_6/')
    plot_fitness_all('D:/Ercihan/Deconvolution/Deconvolution/49/indvs/frac_parents_0_6/', 'D:/Ercihan/Deconvolution/Deconvolution/49/plots/fitness/frac_parents_0_6/')
    plot_fitness_part('D:/Ercihan/Deconvolution/Deconvolution/49/indvs/frac_parents_0_6/', 'D:/Ercihan/Deconvolution/Deconvolution/49/plots/fitness_part/frac_parents_0_6/')

    combine_imgs('D:/Ercihan/Deconvolution/Deconvolution/49/images/frac_parents_0_15/', 'D:/Ercihan/Deconvolution/Deconvolution/49/plots/combined/frac_parents_0_15/')
    plot_fitness_all('D:/Ercihan/Deconvolution/Deconvolution/49/indvs/frac_parents_0_15/', 'D:/Ercihan/Deconvolution/Deconvolution/49/plots/fitness/frac_parents_0_15/')
    plot_fitness_part('D:/Ercihan/Deconvolution/Deconvolution/49/indvs/frac_parents_0_15/', 'D:/Ercihan/Deconvolution/Deconvolution/49/plots/fitness_part/frac_parents_0_15/')

    combine_imgs('D:/Ercihan/Deconvolution/Deconvolution/49/images/frac_parents_0_75/', 'D:/Ercihan/Deconvolution/Deconvolution/49/plots/combined/frac_parents_0_75/')
    plot_fitness_all('D:/Ercihan/Deconvolution/Deconvolution/49/indvs/frac_parents_0_75/', 'D:/Ercihan/Deconvolution/Deconvolution/49/plots/fitness/frac_parents_0_75/')
    plot_fitness_part('D:/Ercihan/Deconvolution/Deconvolution/49/indvs/frac_parents_0_75/', 'D:/Ercihan/Deconvolution/Deconvolution/49/plots/fitness_part/frac_parents_0_75/')

    ### mutation_prob trials ###
    combine_imgs('D:/Ercihan/Deconvolution/Deconvolution/49/images/mutation_prob_0_1/', 'D:/Ercihan/Deconvolution/Deconvolution/49/plots/combined/mutation_prob_0_1/')
    plot_fitness_all('D:/Ercihan/Deconvolution/Deconvolution/49/indvs/mutation_prob_0_1/', 'D:/Ercihan/Deconvolution/Deconvolution/49/plots/fitness/mutation_prob_0_1/')
    plot_fitness_part('D:/Ercihan/Deconvolution/Deconvolution/49/indvs/mutation_prob_0_1/', 'D:/Ercihan/Deconvolution/Deconvolution/49/plots/fitness_part/mutation_prob_0_1/')

    combine_imgs('D:/Ercihan/Deconvolution/Deconvolution/49/images/mutation_prob_0_2/', 'D:/Ercihan/Deconvolution/Deconvolution/49/plots/combined/mutation_prob_0_2/')
    plot_fitness_all('D:/Ercihan/Deconvolution/Deconvolution/49/indvs/mutation_prob_0_2/', 'D:/Ercihan/Deconvolution/Deconvolution/49/plots/fitness/mutation_prob_0_2/')
    plot_fitness_part('D:/Ercihan/Deconvolution/Deconvolution/49/indvs/mutation_prob_0_2/', 'D:/Ercihan/Deconvolution/Deconvolution/49/plots/fitness_part/mutation_prob_0_2/')

    combine_imgs('D:/Ercihan/Deconvolution/Deconvolution/49/images/mutation_prob_0_4/', 'D:/Ercihan/Deconvolution/Deconvolution/49/plots/combined/mutation_prob_0_4/')
    plot_fitness_all('D:/Ercihan/Deconvolution/Deconvolution/49/indvs/mutation_prob_0_4/', 'D:/Ercihan/Deconvolution/Deconvolution/49/plots/fitness/mutation_prob_0_4/')
    plot_fitness_part('D:/Ercihan/Deconvolution/Deconvolution/49/indvs/mutation_prob_0_4/', 'D:/Ercihan/Deconvolution/Deconvolution/49/plots/fitness_part/mutation_prob_0_4/')

    combine_imgs('D:/Ercihan/Deconvolution/Deconvolution/49/images/mutation_prob_0_75/', 'D:/Ercihan/Deconvolution/Deconvolution/49/plots/combined/mutation_prob_0_75/')
    plot_fitness_all('D:/Ercihan/Deconvolution/Deconvolution/49/indvs/mutation_prob_0_75/', 'D:/Ercihan/Deconvolution/Deconvolution/49/plots/fitness/mutation_prob_0_75/')
    plot_fitness_part('D:/Ercihan/Deconvolution/Deconvolution/49/indvs/mutation_prob_0_75/', 'D:/Ercihan/Deconvolution/Deconvolution/49/plots/fitness_part/mutation_prob_0_75/')

    ### mutation_type trials ###
    combine_imgs('D:/Ercihan/Deconvolution/Deconvolution/49/images/mutation_type_guided/', 'D:/Ercihan/Deconvolution/Deconvolution/49/plots/combined/mutation_type_guided/')
    plot_fitness_all('D:/Ercihan/Deconvolution/Deconvolution/49/indvs/mutation_type_guided/', 'D:/Ercihan/Deconvolution/Deconvolution/49/plots/fitness/mutation_type_guided/')
    plot_fitness_part('D:/Ercihan/Deconvolution/Deconvolution/49/indvs/mutation_type_guided/', 'D:/Ercihan/Deconvolution/Deconvolution/49/plots/fitness_part/mutation_type_guided/')

    combine_imgs('D:/Ercihan/Deconvolution/Deconvolution/49/images/mutation_type_unguided/', 'D:/Ercihan/Deconvolution/Deconvolution/49/plots/combined/mutation_type_unguided/')
    plot_fitness_all('D:/Ercihan/Deconvolution/Deconvolution/49/indvs/mutation_type_unguided/', 'D:/Ercihan/Deconvolution/Deconvolution/49/plots/fitness/mutation_type_unguided/')
    plot_fitness_part('D:/Ercihan/Deconvolution/Deconvolution/49/indvs/mutation_type_unguided/', 'D:/Ercihan/Deconvolution/Deconvolution/49/plots/fitness_part/mutation_type_unguided/')
