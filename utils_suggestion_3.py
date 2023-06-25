import numpy as np
import cv2 as cv
import os
import pickle
import random
import copy
import pickle

def pickle_dump(data, path):
    with open(path, 'wb') as fp:
        pickle.dump(data, fp)

def pickle_load(path):
    with open(path, 'rb') as fp:
        return pickle.load(fp)

# current directory
curr_dir = os.getcwd()

# gene class
class gene():
    def __init__(self, im_size):
        super(gene, self).__init__()
        self.im_size = im_size

        center_x, center_y = random.randrange(int(1.8*self.im_size[0])), random.randrange(int(1.8*self.im_size[1]))
        radius = random.randrange(int(max(im_size[0], im_size[1])/2))

        # randomize the parameters
        r, g, b, a = random.randrange(256), random.randrange(256), random.randrange(256), random.random()
        color = [b, g, r, a]

        # check the validation of the parameter
        while not self.check_valid(center_x, center_y, radius):
            # perform the randomization again
            center_x, center_y = random.randrange(int(1.8*self.im_size[0])), random.randrange(int(1.8*self.im_size[1]))
            radius = random.randrange(int(max(im_size[0], im_size[1])/2))

        self.center_x = center_x
        self.center_y = center_y
        self.radius = radius
        self.color = color

    # check if the given x, y and r parameters intersect with the painting space
    def check_valid(self, center_x, center_y, radius):
        # evaluate the distance of the center
        distance_x = abs(center_x - self.im_size[0]/2)
        distance_y = abs(center_y - self.im_size[1]/2)

        # check the situations
        if distance_x > (self.im_size[0]/2 + radius):
            return False
        if distance_y > (self.im_size[1]/2 + radius):
            return False
        if distance_x <= (self.im_size[0]/2):
            return True
        if distance_y <= (self.im_size[1]/2):
            return True

        # calculate the square distance
        dist_sq = (distance_x - self.im_size[0]/2)**2 + (distance_y - self.im_size[1]/2)**2

        return (dist_sq <= (radius**2))

# individual class
class individual():
    def __init__(self, gene_num, im_size):
        super(individual, self).__init__()
        self.gene_num = gene_num
        self.im_size = im_size
        # create the chromosome, sort the descending list, chromosome is a list of genes
        self.chrom = [gene(im_size = self.im_size) for _ in range(self.gene_num)]
        self.fitness = None

    def set(self):
        # loop for all of the genes to initialize them
        for gene in self.chrom:
            gene.__init__(im_size = self.im_size)
        self.chrom = sorted(self.chrom, key=lambda x: x.radius, reverse=True)

    def sort_chrom(self):
        self.chrom = sorted(self.chrom, key=lambda x: x.radius, reverse=True)

    def set_fitness(self, fitness):
        self.fitness = fitness

# population class
class population():
    def __init__(self, ind_num, gen_num, img_size, next=None):
        super(population, self).__init__()
        self.next = next
        self.ind_num = ind_num
        self.gen_num = gen_num
        self.img_size = img_size
        if self.next == None:
            self.indvs = [individual(self.gen_num, self.img_size) for _ in range(self.ind_num)]
        else:
            self.indvs = next

    def set(self, source_img):
        if self.next == None:
            for indv in self.indvs:
                # create the individuals in the population one by one by
                indv.set()
                # set the fitness of the individual one by one by
                indv = evaluate(indv, source_img)
        else:
            for indv in self.indvs:
                # set the fitness of the individual one by one by
                indv = evaluate(indv, source_img)

# evaluation and drawing function
def evaluate(individual, source_img):
    # be sure that chromosome is sorted
    individual.sort_chrom()

    # create a completely white image with the same size as the source image
    image = np.zeros((source_img.shape[0], source_img.shape[1], 3), dtype = np.uint8)
    image.fill(255)

    # draw circles on the image using the genes of the individual
    for gene in individual.chrom:

        # draw the circle on a new overlay image
        overlay = copy.deepcopy(image)
        cv.circle(overlay, (gene.center_x, gene.center_y), gene.radius, (gene.color[0], gene.color[1], gene.color[2]), -1)

        # update the image
        image = cv.addWeighted(overlay, gene.color[3], image, 1.0-gene.color[3], 0.0, image)

    # compute the fitness of the individual using the given relation
    fitness = -np.sum(np.square(np.subtract(np.array(source_img, dtype=np.int64), np.array(image, dtype=np.int64))))
    individual.set_fitness(fitness)

    return individual

# get the image of the individual
def get_image(individual, source_img):
    # be sure that chromosome is sorted
    individual.sort_chrom()

    image = np.zeros((source_img.shape[0], source_img.shape[1], 3), dtype = np.uint8)
    image.fill(255)

    for gene in individual.chrom:

        # draw the circle on a new overlay image
        overlay = copy.deepcopy(image)
        cv.circle(overlay, (gene.center_x, gene.center_y), gene.radius, (gene.color[0], gene.color[1], gene.color[2]), -1)

        # update the image
        image = cv.addWeighted(overlay, gene.color[3], image, 1.0-gene.color[3], 0.0, image)

    return image

# elitism selection function
def elitism_selection(pop, num_elites):
    # sort the population based on fitness in descending order
    sorted_population = sorted(pop.indvs, key=lambda ind_fit: ind_fit.fitness, reverse=True)

    # select the elites from the sorted population
    elites = [ind_fit for ind_fit in sorted_population[:num_elites]]

    # return the elite individuals
    return elites

# tournament selection function for picking parents
def tournament_selection(pop, num_elites, num_parents, tm_size):
    # sort the population based on fitness in descending order
    sorted_population = sorted(pop.indvs, key=lambda ind_fit: ind_fit.fitness, reverse=True)
    others = [ind_fit for ind_fit in sorted_population[num_elites:]]
    all = [ind_fit for ind_fit in sorted_population]
    parents = []
    # perform the tournament
    for i in range(num_parents):
        best_indv = random.randrange(len(others))
        for j in range(tm_size):
            temp_indv = random.randrange(len(others))
            if others[temp_indv].fitness > others[best_indv].fitness:
                best_indv = temp_indv
        # store the best "num_parents" parents
        parents.append(others.pop(best_indv))

    # return the parents and others
    return parents, others

# crossover function
def crossover(pop, num_elites, num_parents, tm_size, source_img):
    # fix the parent number if required
    if num_parents % 2 == 1:
        num_parents = num_parents + 1

    # select the parents from the remaining population using tournament selection
    parents, next_generation_others = tournament_selection(pop, num_elites, num_parents, tm_size)

    # perform crossover operation to create children
    children = []
    for i in range(0, num_parents, 2):
        # pick the parents
        parent1, parent2 = parents[i], parents[i+1]

        """print("parent1: ")
        for gene in parent1.chrom:
            print("radius: ", str(gene.radius), "x: ", str(gene.center_x), "y: ", str(gene.center_y), "color: ", str(gene.color))
        print("parent2: ")
        for gene in parent2.chrom:
            print("radius: ", str(gene.radius), "x: ", str(gene.center_x), "y: ", str(gene.center_y), "color: ", str(gene.color))"""

        child1 = copy.deepcopy(parent1)
        child2 = copy.deepcopy(parent2)

        for j in range(len(parent1.chrom)):
            if random.random() >= 0.5:
                child1.chrom[j] =  copy.deepcopy(parent1.chrom[j])
                child2.chrom[j] =  copy.deepcopy(parent2.chrom[j])
            else:
                child1.chrom[j] =  copy.deepcopy(parent2.chrom[j])
                child2.chrom[j] =  copy.deepcopy(parent1.chrom[j])

        #child1 = evaluate(child1, source_img)
        #child2 = evaluate(child2, source_img)

        """print("child1: ")
        for gene in child1.chrom:
            print("radius: ", str(gene.radius), "x: ", str(gene.center_x), "y: ", str(gene.center_y), "color: ", str(gene.color))
        print("child2: ")
        for gene in child2.chrom:
            print("radius: ", str(gene.radius), "x: ", str(gene.center_x), "y: ", str(gene.center_y), "color: ", str(gene.color))"""

        # select the best two individuals among the two parents and their children
        child1 = evaluate(child1, source_img)
        child2 = evaluate(child2, source_img)
        temp_arr = [parent1, parent2, child1, child2]
        sorted_population = sorted(temp_arr, key=lambda ind_fit: ind_fit.fitness, reverse=True)
        children.append(sorted_population[0])
        children.append(sorted_population[1])

    """print("FIRST")
    for gene in children[0].chrom:
        print("radius: ", str(gene.radius), "x: ", str(gene.center_x), "y: ", str(gene.center_y), "color: ", str(gene.color))"""
    next_generation_childs = children

    return next_generation_childs, next_generation_others

# perform mutation function
def perform_mutation(subject, gene, mut_type, im_size):
    # unguided mutation
    if mut_type == 'unguided':
        subject.chrom[gene].__init__(im_size = im_size)

    # guided mutation
    else: # color b g r a
        # apply the given mutation algorithm for center and radius
        mutated_center_x = random.randrange(max(0, int(subject.chrom[gene].center_x-im_size[0]/4)), int(subject.chrom[gene].center_x+im_size[0]/4) + 1)
        mutated_center_y = random.randrange(max(0, int(subject.chrom[gene].center_y-im_size[1]/4)), int(subject.chrom[gene].center_y+im_size[1]/4) + 1)
        mutated_radius = random.randrange(max(0, subject.chrom[gene].radius-10), subject.chrom[gene].radius+11)

        # check the validation
        while not subject.chrom[gene].check_valid(mutated_center_x, mutated_center_y, mutated_radius):
            mutated_center_x = random.randrange(max(0, int(subject.chrom[gene].center_x-im_size[0]/4)), int(subject.chrom[gene].center_x+im_size[0]/4) + 1)
            mutated_center_y = random.randrange(max(0, int(subject.chrom[gene].center_y-im_size[1]/4)), int(subject.chrom[gene].center_y+im_size[1]/4) + 1)
            mutated_radius = random.randrange(max(0, subject.chrom[gene].radius - 10), subject.chrom[gene].radius + 11)

        # give the mutated features to the current gene
        subject.chrom[gene].center_x = mutated_center_x
        subject.chrom[gene].center_y = mutated_center_y
        subject.chrom[gene].radius = mutated_radius

        # apply the given mutation algorithm for color and alpha, give them to the gene
        subject.chrom[gene].color[2] = random.randrange(max(0, subject.chrom[gene].color[2] - 64), min(subject.chrom[gene].color[2] + 65, 255))
        subject.chrom[gene].color[1] = random.randrange(max(0, subject.chrom[gene].color[1] - 64), min(subject.chrom[gene].color[1] + 65, 255))
        subject.chrom[gene].color[0] = random.randrange(max(0, subject.chrom[gene].color[0] - 64), min(subject.chrom[gene].color[0] + 65, 255))

        a_temp = random.uniform(-0.25, 0.25)
        subject.chrom[gene].color[3] =  max(0, min(1.0, a_temp + subject.chrom[gene].color[3]))

    return subject

# mutation function
def mutate(subjects, mutation_type, mutation_prob, num_elites, gene_num, source_img):
    im_size = source_img.shape

    children = []
    for subject in subjects:
        if random.random() < mutation_prob:
            mutated_genes = []
            # pick a random gene from the chromosome of the individual
            gene = random.randrange(len(subject.chrom))
            mutated_genes.append(gene)

            # perform the mutation on the gene
            subject = perform_mutation(subject, gene, mutation_type, im_size)

            # run the mutation loop until the a random value smaller than the probability threshold is obtained
            while random.random() < mutation_prob:
                # if all genes mutated, exit the loop
                if len(mutated_genes) >= len(subject.chrom):
                    break

                # if gene already mutated, choose another gene
                while gene in mutated_genes:
                    gene = random.randrange(len(subject.chrom))

                mutated_genes.append(gene)
                subject = perform_mutation(subject, gene, mutation_type, im_size)

        # update the fitness of the individual after mutation
        #subject = evaluate(subject, source_img)
        children.append(subject)

    next_generation_children = children
    return next_generation_children

########################################

# let the evolution begin!!!
def evolution_process(num_inds, num_genes, tm_size, frac_elites, frac_parents, mutation_prob, mutation_type, generation_num, source_img, case):
    # image size
    img_size = source_img.shape

    # create the population
    pop = population(num_inds, num_genes, img_size)

    best_indvs = []
    for i in range(generation_num):
        # update the fraction of elites and parents continuously
        if i != 0 and i%500 == 0:
            frac_elites = frac_elites*1.1
            frac_parents = frac_parents*0.95
            print("frac_elites: ", str(frac_elites))
            print("frac_parents: ", str(frac_parents))

        # evaluate new numbers of elites and parents
        num_elites = int(frac_elites * num_inds)
        num_parents = int(frac_parents * num_inds)

        """if i != 0:
            print("INIT")
            for indv in pop.next:
                print("new indv fitness: ", str(indv.fitness))
                for gene in indv.chrom:
                    print("radius: ", str(gene.radius), "x: ", str(gene.center_x), "y: ", str(gene.center_y), "color: ", str(gene.color))"""

        # set the population
        pop.set(source_img)
        """if i != 0:
            print("INIT AFTER SET")
            for indv in pop.next:
                print("new indv fitness: ", str(indv.fitness))
                for gene in indv.chrom:
                    print("radius: ", str(gene.radius), "x: ", str(gene.center_x), "y: ", str(gene.center_y), "color: ", str(gene.color))"""
        #print("size: ", str(len(pop.indvs)))

        # pick elites
        elites = elitism_selection(pop, num_elites)

        # perform crossover for some parents
        children_crossover, children_others = crossover(pop, num_elites, num_parents, tm_size, source_img)
        """print("AFTER")
        for gene in children_crossover[0].chrom:
            print("radius: ", str(gene.radius), "x: ", str(gene.center_x), "y: ", str(gene.center_y), "color: ", str(gene.color))"""

        # subjects to mutation
        subjects = children_crossover + children_others

        # perform mutation (possibly) for all individuals
        children_mutation = mutate(subjects, mutation_type, mutation_prob, num_elites, num_genes, source_img)

        # get the final form of indiviudals in this generation
        new_gen_indvs = elites + children_mutation

        # update the population
        pop = population(num_inds, num_genes, img_size, next = new_gen_indvs)

        #print("next len: ", str(len(pop.next)))
        """print("LAST")
        for indv in pop.next:
            print("new indv fitness: ", str(indv.fitness))
            for gene in indv.chrom:
                print("radius: ", str(gene.radius), "x: ", str(gene.center_x), "y: ", str(gene.center_y), "color: ", str(gene.color))"""

        # saving part #
        if i%100 == 0:
            print("iteration: ", str(i))

        if i%500 == 499:
            for ind in new_gen_indvs:
                print("fitness: ", str(ind.fitness))

        # best individual at the 1000th generation
        if i%1000 == 0:
            if i == 0:
                i = 1
            # sort the population
            sorted_population = sorted(pop.indvs, key=lambda ind_fit: ind_fit.fitness, reverse=True)

            # get the best individual of the current generation
            best_indvs.append(sorted_population[0])

            # save pickle of population
            pickle_name = 'D:/Ercihan/Deconvolution/Deconvolution/49/pickles/' + case
            if not os.path.exists(pickle_name):
                os.makedirs(pickle_name)
            # create the pickle of generation
            pickle_dump(pop, pickle_name + 'generation_' + str(i) + '.pickle')

            # save image of best individual
            img_name = 'D:/Ercihan/Deconvolution/Deconvolution/49/images/' + case
            if not os.path.exists(img_name):
                os.makedirs(img_name)
            # write the image of the best individual
            cv.imwrite(img_name + 'generation_' + str(i) + '.png', get_image(best_indvs[-1], source_img))

        # hold the best individual at each generation
        else:
            # sort the population
            sorted_population = sorted(pop.indvs, key=lambda ind_fit: ind_fit.fitness, reverse=True)

            # get the best individual of the current generation
            best_indvs.append(sorted_population[0])

    # save the best individuals list pickle
    indvs_name = 'D:/Ercihan/Deconvolution/Deconvolution/49/indvs/' + case
    if not os.path.exists(indvs_name):
        os.makedirs(indvs_name)
    # create the pickle of best individual list
    pickle_dump(best_indvs, indvs_name + 'all_best_individuals.pickle')

    # save pickle of the last population
    pickle_name = 'D:/Ercihan/Deconvolution/Deconvolution/49/pickles/' + case
    # create the pickle of generation
    pickle_dump(pop, pickle_name + 'generation_10000' + '.pickle')

    # save image of best individual
    img_name = 'D:/Ercihan/Deconvolution/Deconvolution/49/images/' + case
    # write the image of the best individual
    cv.imwrite(img_name + 'generation_10000.png', get_image(best_indvs[-1], source_img))

# case for pickles
if __name__ == '__main__':

    source_img = cv.imread('painting.png')
    #cv.imshow('image', source_img)

    #cv.waitKey(0)
    #cv.destroyAllWindows()

    print(source_img.shape)

    num_genes_default = 50
    num_inds_default = 20
    mutation_type_default = 'guided'
    mutation_prob_default = 0.2
    tm_size_default = 5
    frac_elites_default = 0.2
    frac_parents_default = 0.6
    generation_num = 10000
    case = 'suggestion_3/'

    evolution_process(num_inds_default, num_genes_default, tm_size_default, frac_elites_default, frac_parents_default, mutation_prob_default, mutation_type_default, generation_num, source_img, case)

    """### num_inds trials ###
    evolution_process(5, num_genes_default, tm_size_default, frac_elites_default, frac_parents_default, mutation_prob_default, mutation_type_default, generation_num, source_img, 'num_inds_5/')
    evolution_process(10, num_genes_default, tm_size_default, frac_elites_default, frac_parents_default, mutation_prob_default, mutation_type_default, generation_num, source_img, 'num_inds_10/')
    evolution_process(20, num_genes_default, tm_size_default, frac_elites_default, frac_parents_default, mutation_prob_default, mutation_type_default, generation_num, source_img, 'num_inds_20/')
    evolution_process(40, num_genes_default, tm_size_default, frac_elites_default, frac_parents_default, mutation_prob_default, mutation_type_default, generation_num, source_img, 'num_inds_40/')
    evolution_process(60, num_genes_default, tm_size_default, frac_elites_default, frac_parents_default, mutation_prob_default, mutation_type_default, generation_num, source_img, 'num_inds_60/')

    ### num_genes trials ###
    evolution_process(num_inds_default, 15, tm_size_default, frac_elites_default, frac_parents_default, mutation_prob_default, mutation_type_default, generation_num, source_img, 'num_genes_15/')
    evolution_process(num_inds_default, 30, tm_size_default, frac_elites_default, frac_parents_default, mutation_prob_default, mutation_type_default, generation_num, source_img, 'num_genes_30/')
    evolution_process(num_inds_default, 50, tm_size_default, frac_elites_default, frac_parents_default, mutation_prob_default, mutation_type_default, generation_num, source_img, 'num_genes_50/')
    evolution_process(num_inds_default, 80, tm_size_default, frac_elites_default, frac_parents_default, mutation_prob_default, mutation_type_default, generation_num, source_img, 'num_genes_80/')
    evolution_process(num_inds_default, 120, tm_size_default, frac_elites_default, frac_parents_default, mutation_prob_default, mutation_type_default, generation_num, source_img, 'num_genes_120/')

    ### tm_size trials ###
    evolution_process(num_inds_default, num_genes_default, 2, frac_elites_default, frac_parents_default, mutation_prob_default, mutation_type_default, generation_num, source_img, 'tm_size_2/')
    evolution_process(num_inds_default, num_genes_default, 5, frac_elites_default, frac_parents_default, mutation_prob_default, mutation_type_default, generation_num, source_img, 'tm_size_5/')
    evolution_process(num_inds_default, num_genes_default, 8, frac_elites_default, frac_parents_default, mutation_prob_default, mutation_type_default, generation_num, source_img, 'tm_size_8/')
    evolution_process(num_inds_default, num_genes_default, 16, frac_elites_default, frac_parents_default, mutation_prob_default, mutation_type_default, generation_num, source_img, 'tm_size_16/')

    ### frac_elites trials ###
    evolution_process(num_inds_default, num_genes_default, tm_size_default, 0.04, frac_parents_default, mutation_prob_default, mutation_type_default, generation_num, source_img, 'frac_elites_0_04/')
    evolution_process(num_inds_default, num_genes_default, tm_size_default, 0.2, frac_parents_default, mutation_prob_default, mutation_type_default, generation_num, source_img, 'frac_elites_0_2/')
    evolution_process(num_inds_default, num_genes_default, tm_size_default, 0.35, frac_parents_default, mutation_prob_default, mutation_type_default, generation_num, source_img, 'frac_elites_0_35/')

    ### frac_parents trials ###
    evolution_process(num_inds_default, num_genes_default, tm_size_default, frac_elites_default, 0.15, mutation_prob_default, mutation_type_default, generation_num, source_img, 'frac_parents_0_15/')
    evolution_process(num_inds_default, num_genes_default, tm_size_default, frac_elites_default, 0.3, mutation_prob_default, mutation_type_default, generation_num, source_img, 'frac_parents_0_3/')
    evolution_process(num_inds_default, num_genes_default, tm_size_default, frac_elites_default, 0.6, mutation_prob_default, mutation_type_default, generation_num, source_img, 'frac_parents_0_6/')
    evolution_process(num_inds_default, num_genes_default, tm_size_default, frac_elites_default, 0.75, mutation_prob_default, mutation_type_default, generation_num, source_img, 'frac_parents_0_75/')

    ### mutation_prob trials ###
    evolution_process(num_inds_default, num_genes_default, tm_size_default, frac_elites_default, frac_parents_default, 0.1, mutation_type_default, generation_num, source_img, 'mutation_prob_0_1/')
    evolution_process(num_inds_default, num_genes_default, tm_size_default, frac_elites_default, frac_parents_default, 0.2, mutation_type_default, generation_num, source_img, 'mutation_prob_0_2/')
    evolution_process(num_inds_default, num_genes_default, tm_size_default, frac_elites_default, frac_parents_default, 0.4, mutation_type_default, generation_num, source_img, 'mutation_prob_0_4/')
    evolution_process(num_inds_default, num_genes_default, tm_size_default, frac_elites_default, frac_parents_default, 0.75, mutation_type_default, generation_num, source_img, 'mutation_prob_0_75/')

    ### mutation_type trials ###
    evolution_process(num_inds_default, num_genes_default, tm_size_default, frac_elites_default, frac_parents_default, mutation_prob_default, 'guided', generation_num, source_img, 'mutation_type_guided/')
    evolution_process(num_inds_default, num_genes_default, tm_size_default, frac_elites_default, frac_parents_default, mutation_prob_default, 'unguided', generation_num, source_img, 'mutation_type_unguided/')
"""