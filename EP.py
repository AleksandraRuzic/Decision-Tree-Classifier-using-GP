
# coding: utf-8

# In[1]:


import Tree
import random
import copy


# In[2]:


population_size = 10
tournament_size = 5
num_of_iterations = 10


# In[3]:


def init_individual():
    left_part = random.sample(Tree.column_names, 1)[0]
    if left_part in Tree.atributes['num']:
        relation = random.sample(Tree.relations['num'], 1)[0]
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


# In[4]:


def selection(population):
    max_fitness = 0.0
    k = -1
    for i in range(tournament_size):
        j = random.randrange(population_size)
        if population[j].calculate_fitness() > max_fitness:
            max_fitness = population[j].calculate_fitness()
            k = j
    return k


# In[5]:


def build_block(tree):
    viable_indexes = set(
        filter(lambda index:  
            not any(index2 == 2*index for index2 in tree.setOfIndexes),
            tree.setOfIndexes))
    tree_index = random.sample(viable_indexes, 1)[0]
    new_node = init_individual().root_node
    tree.remove_node(tree_index)
    tree.add_subtree(tree_index, new_node)


# In[6]:


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


# In[9]:


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
        


# In[10]:


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


# In[11]:


#todo: set probability for operators

population = []
newPopulation = []
for i in range(population_size):
    population.append(init_individual())
    newPopulation.append(init_individual())
    
for iteration in range(num_of_iterations):
    for i in range(0, population_size, 2):
        k1 = selection(population)
        k2 = selection(population)
        [child1, child2] = crossover(population[k1], population[k2])
        relation_mutation(child1)
        relation_mutation(child2)
        right_side_mutation(child1)
        right_side_mutation(child2)
        build_block(child1)
        build_block(child2)
        newPopulation[i] = child1
        newPopulation[i+1] = child2
    maximum = 0
    tree = None
    for i in range(population_size):
        if newPopulation[i].calculate_fitness() > maximum:
            maximum = newPopulation[i].calculate_fitness()
            tree = copy.deepcopy(newPopulation[i])
    print()
    tree.print_tree()
    print(tree.calculate_fitness())
    print()
    population = newPopulation
        

