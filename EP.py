#!/usr/bin/env python
# coding: utf-8

# In[33]:


import Tree
import random
import copy


# In[34]:


def init_individual():
    left_part = random.sample(Tree.column_names, 1)[0]
    if left_part in Tree.atributes['num']:
        relation = random.sample(set(Tree.relations['num'] | Tree.relations['cat']), 1)[0]
        if Tree.data[left_part].dtype == 'float64':
            right_part = random.uniform(min(Tree.data[left_part]), max(Tree.data[left_part]))
        elif Tree.data[left_part].dtype == 'int64':
            right_part = random.randrange(min(Tree.data[left_part]), max(Tree.data[left_part]))
    elif left_part in Tree.atributes['cat']:
        relation = random.sample(Tree.relations['cat'], 1)[0]
        right_part = random.sample(set(data[left_part]), 1)[0]
    t = Tree.Tree([left_part, relation, right_part])
    t.add_node(t.root_node, True, random.sample(Tree.class_names, 1))
    t.add_node(t.root_node, False, random.sample(Tree.class_names, 1)) 
    return t


# In[35]:


def selection(population, tournament_size, population_size):
    max_fitness = 0.0
    k = -1
    for i in range(tournament_size):
        j = random.randrange(population_size)
        if population[j].calculate_fitness() > max_fitness:
            max_fitness = population[j].calculate_fitness()
            k = j
    return k


# In[36]:


def build_block(tree):
    viable_indexes = set(
        filter(lambda index:  
            not any(index2 == 2*index for index2 in tree.setOfIndexes),
            tree.setOfIndexes))
    tree_index = random.sample(viable_indexes, 1)[0]
    new_node = init_individual().root_node
    tree.remove_node(tree_index)
    tree.add_subtree(tree_index, new_node)


# In[37]:


def crossover(parent1, parent2):
    child1 = copy.deepcopy(parent1)
    child2 = copy.deepcopy(parent2)

    child1_viable_indexes = set(
        filter(lambda index:  
            any(index2 == 2*index for index2 in child1.setOfIndexes),
            child1.setOfIndexes))
    child2_viable_indexes = set(
        filter(lambda index: 
            any(index2 == 2*index for index2 in child2.setOfIndexes),
            child2.setOfIndexes))
    child1_index = random.sample(child1_viable_indexes, 1)[0]
    child2_index = random.sample(child2_viable_indexes, 1)[0]
    
    if child1_index == 1 or child2_index == 1:
        return [child1, child2]

    child1_subtree = child1.index_of(child1_index)
    child2_subtree = child2.index_of(child2_index)
    
    child1.remove_node(child1_index)
    child2.remove_node(child2_index)
    
    child1.add_subtree(child1_index, child2_subtree)
    child2.add_subtree(child2_index, child1_subtree)
    
    return [child1, child2]


# In[38]:


def relation_mutation(tree):
    
    tree_viable_indexes = set(
        filter(lambda index:  
            any(index2 == 2*index for index2 in tree.setOfIndexes),
            tree.setOfIndexes))
    tree_index = random.sample(tree_viable_indexes, 1)[0]
    node = tree.index_of(tree_index)
    if node.left_part in Tree.atributes['num']:
        node.relation = random.sample(Tree.relations['num'], 1)[0]
    elif node.left_part in Tree.atributes['cat']:
        node.relation = random.sample(Tree.relations['cat'], 1)[0]
        


# In[39]:


def right_side_mutation(tree):
    
    tree_viable_indexes = set(
        filter(lambda index:  
            any(index2 == 2*index for index2 in tree.setOfIndexes),
            tree.setOfIndexes))
    tree_index = random.sample(tree_viable_indexes, 1)[0]
    node = tree.index_of(tree_index)
    if node.left_part in Tree.atributes['num']:
        if Tree.data[node.left_part].dtype == 'float64':
            node.right_part = random.uniform(
                min(Tree.data[node.left_part]), max(Tree.data[node.left_part]))
        elif Tree.data[node.left_part].dtype == 'int64':
            node.right_part = random.randrange(
                min(Tree.data[node.left_part]), max(Tree.data[node.left_part]))
    elif node.left_part in Tree.atributes['cat']:
        node.right_part = random.sample(set(data[node.left_part]), 1)[0]


# In[40]:


def prune(tree):
    
    tree_viable_indexes = set(
        filter(lambda index:  
            any(index2 == 2*index for index2 in tree.setOfIndexes),
            tree.setOfIndexes))
    tree_index = random.sample(tree_viable_indexes, 1)[0]
    if tree_index == 1:
        return
    tree.remove_node(tree_index)
    new_node = Tree.Leaf(tree_index, random.sample(Tree.class_names, 1)[0])
    tree.add_subtree(tree_index, new_node)


# In[41]:


def tree_evolution(num_of_iterations, population_size, tournament_size, crossover_prob, relation_mutation_prob, right_side_mutation_prob, prune_prob, build_block_prob):
    population = []
    newPopulation = []
    max_fitness = 0
    best_tree = None
    best_depth = 1
        
    for i in range(population_size):
        population.append(init_individual())
        newPopulation.append(init_individual())

    for iteration in range(num_of_iterations):
        for i in range(0, population_size, 2):
            k1 = selection(population, tournament_size, population_size)
            k2 = selection(population, tournament_size, population_size)
            if random.random() < crossover_prob:
                [child1, child2] = crossover(population[k1], population[k2])
            else:
                child1 = copy.deepcopy(population[k1])
                child2 = copy.deepcopy(population[k2])
            if random.random() < relation_mutation_prob:
                relation_mutation(child1)
            if random.random() < relation_mutation_prob:
                relation_mutation(child2)
            if random.random() < right_side_mutation_prob:
                right_side_mutation(child1)
            if random.random() < right_side_mutation_prob:
                right_side_mutation(child2)
            if random.random() < prune_prob:
                prune(child1)
            if random.random() < prune_prob:
                prune(child2)
            if random.random() < build_block_prob:
                build_block(child1)
            if random.random() < build_block_prob:
                build_block(child2)
            newPopulation[i] = child1
            newPopulation[i+1] = child2
            
        for i in range(population_size):
            curr_depth = len(bin(max(newPopulation[i].setOfIndexes))[3:])
            curr_fitnes = newPopulation[i].calculate_fitness()
            if (curr_fitnes > max_fitness) or (abs(curr_fitnes - max_fitness) < 0.000001 and best_depth > curr_depth):
                max_fitness = curr_fitnes
                best_tree = copy.deepcopy(newPopulation[i])
                best_depth = curr_depth
                
        """print()
        tree.print_tree()
        print(tree.calculate_fitness())
        print()"""
        population = newPopulation
    return (tree, tree.calculate_fitness())


# In[42]:


def tree_evolution_1(num_of_iterations, population_size, tournament_size, crossover_prob, relation_mutation_prob, right_side_mutation_prob, prune_prob, build_block_prob):
    population = []
    newPopulation = []
    L = 0.0
    avg_dep = 1
    avg_width = 2
    prev_avg_dep = 0
    prev_avg_width = 0
    best_tree = None
    max_fitness = 0
    best_depth = 1
    
    for i in range(population_size):
        population.append(init_individual())
        newPopulation.append(init_individual())

    for iteration in range(num_of_iterations):

        if (avg_dep+avg_width) - (prev_avg_dep+prev_avg_width) <= L:
            for i in range(0, population_size):
                if random.random() < build_block_prob:
                    build_block(population[i])

        for i in range(0, population_size, 2):

            k1 = selection(population, tournament_size, population_size)
            k2 = selection(population, tournament_size, population_size)

            if random.random() < crossover_prob:
                [child1, child2] = crossover(population[k1], population[k2])
            else:
                child1 = copy.deepcopy(population[k1])
                child2 = copy.deepcopy(population[k2])
            if random.random() < relation_mutation_prob:
                relation_mutation(child1)
            if random.random() < relation_mutation_prob:
                relation_mutation(child2)
            if random.random() < right_side_mutation_prob:
                right_side_mutation(child1)
            if random.random() < right_side_mutation_prob:
                right_side_mutation(child2)
            if random.random() < prune_prob:
                prune(child1)
            if random.random() < prune_prob:
                prune(child2)

            newPopulation[i] = child1
            newPopulation[i+1] = child2

        prev_avg_width = avg_width
        prev_avg_dep = avg_dep
        avg_width = 0
        avg_dep = 0

        for i in range(population_size):
            curr_depth = len(bin(max(newPopulation[i].setOfIndexes))[3:])
            avg_dep += curr_depth
            avg_width += len(set(filter(
                lambda index:not any(index2 == 2*index for index2 in newPopulation[i].setOfIndexes),
                newPopulation[i].setOfIndexes)))
            curr_fitnes = newPopulation[i].calculate_fitness()
            if (curr_fitnes > max_fitness) or (abs(curr_fitnes - max_fitness) < 0.000001 and best_depth > curr_depth):
                max_fitness = curr_fitnes
                best_tree = copy.deepcopy(newPopulation[i])
                best_depth = curr_depth

        avg_dep /= population_size
        avg_width /= population_size

        """print()
        tree.print_tree()
        print(tree.calculate_fitness())
        print()"""
        population = newPopulation
    return (best_tree, max_fitness)


# In[43]:


def tree_evolution_2(num_of_iterations, population_size, tournament_size, crossover_prob, relation_mutation_prob, right_side_mutation_prob, prune_prob, build_block_prob):
    population = []
    newPopulation = []
    L = 0.0
    avg_dep = 1
    avg_width = 2
    prev_avg_dep = 0
    prev_avg_width = 0
    best_tree = None
    max_fitness = 0
    best_depth = 1
    
    for i in range(population_size):
        population.append(init_individual())
        newPopulation.append(init_individual())

    for iteration in range(num_of_iterations):

        for i in range(0, population_size):
            current_width = len(set(filter(
                lambda index : not any(index2 == 2*index for index2 in newPopulation[i].setOfIndexes),
                newPopulation[i].setOfIndexes)))
            current_dep = len(bin(max(newPopulation[i].setOfIndexes))[3:])
            if avg_dep >= current_dep and avg_width >= current_width:
                if random.random() < build_block_prob:
                    build_block(population[i])

        for i in range(0, population_size, 2):

            k1 = selection(population, tournament_size, population_size)
            k2 = selection(population, tournament_size, population_size)

            if random.random() < crossover_prob:
                [child1, child2] = crossover(population[k1], population[k2])
            else:
                child1 = copy.deepcopy(population[k1])
                child2 = copy.deepcopy(population[k2])
            if random.random() < relation_mutation_prob:
                relation_mutation(child1)
            if random.random() < relation_mutation_prob:
                relation_mutation(child2)
            if random.random() < right_side_mutation_prob:
                right_side_mutation(child1)
            if random.random() < right_side_mutation_prob:
                right_side_mutation(child2)
            if random.random() < prune_prob:
                prune(child1)
            if random.random() < prune_prob:
                prune(child2)

            newPopulation[i] = child1
            newPopulation[i+1] = child2

        prev_avg_width = avg_width
        prev_avg_dep = avg_dep
        avg_width = 0
        avg_dep = 0

        for i in range(population_size):
            curr_depth = len(bin(max(newPopulation[i].setOfIndexes))[3:])
            avg_dep += curr_depth
            avg_width += len(set(filter(
                lambda index:not any(index2 == 2*index for index2 in newPopulation[i].setOfIndexes),
                newPopulation[i].setOfIndexes)))
            curr_fitnes = newPopulation[i].calculate_fitness()
            if (curr_fitnes > max_fitness) or (abs(curr_fitnes - max_fitness) < 0.000001 and max_depth > curr_depth):
                max_fitness = curr_fitnes
                best_tree = copy.deepcopy(newPopulation[i])
                max_depth = curr_depth

        avg_dep /= population_size
        avg_width /= population_size

        """print()
        best_tree.print_tree()
        print(best_tree.calculate_fitness())
        print()"""
        population = newPopulation
    return (best_tree, max_fitness)


# In[44]:


def calculate_test_fitness(tree):
    y_pred = ["0"] * len(Tree.X_test.index)
    for i in range(len(Tree.X_test.index)):
        y_pred[i] = Tree.predict_point(Tree.X_test.iloc[i], tree.root_node)

    n_rows = len(y_pred)
    predicted = 0
    for i in range(n_rows):
        if (y_pred[i] == Tree.y_test[i]):
            predicted += 1

    return predicted/n_rows


# In[ ]:




