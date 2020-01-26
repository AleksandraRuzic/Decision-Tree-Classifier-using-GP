#!/usr/bin/env python
# coding: utf-8

# In[1]:


import math
import pandas as pd
import copy

whole_data = pd.read_csv('iris.csv')

class_column_name = 'Species'
class_variable = whole_data['Species']

column_names = whole_data.columns
column_names = set(column_names.drop('Species'))
data = whole_data.drop('Species', axis = 1)

relations = {'num':{'<', '<=', '>', '>='}, 'cat':{'==', '!='}}
atributes = {'num':set(filter(lambda x : data[x].dtype=='float64' or data[x].dtype== 'int64', column_names)),
             'cat':set(filter(lambda x : data[x].dtype!= 'float64' and data[x].dtype!= 'int64', column_names))}
class_names = set(class_variable)
        


# In[2]:


class Node:
    def __init__(self, index):
        self.index = index
        
        
    def print_node(self):
        print(str(self.index) + ". : ", end= ' ')
        
        
        
class Leaf(Node):
    def __init__(self, index, class_name):
        if class_name in class_names:
            Node.__init__(self, index)
            self.class_name = class_name
        else:
            return None
        
        
    def print_node(self):
        Node.print_node(self)
        print(self.class_name)
        

    def set_class_name(self, class_name):
        self.class_name = class_name
        
        
        
class NotLeaf(Node):
    def __init__(self, index, left_part, relation, right_part):
       # if self.wrong_arg(left_part, relation, right_part):
        #    return None;
        Node.__init__(self, index)
        self.left_part = left_part
        self.relation = relation
        self.right_part = right_part
        
        
    def print_node(self):
        Node.print_node(self)
        print(self.left_part, self.relation, self.right_part)
        
        
    def wrong_arg(self, left_part, relation, right_part):
        return (not (left_part in column_names) or 
                 (left_part in atributes['cat'] and relation in relations['num']) or 
                 (left_part in atributes['num'] and not(right_part in range(min(data[left_part]), max(data[left_part])))) or 
                 (left_part in atributes['cat'] and not(right_part in set(data[left_part]))))


# In[3]:


class Tree:
    def __init__(self, node_arguments):
        if len(node_arguments) == 1:
            self.node_with_index = {1 : Leaf(1, node_arguments[0])}
        else:
            self.node_with_index = {1 : NotLeaf(1, node_arguments[0], node_arguments[1], node_arguments[2])}
            
            
    #def predict_point(row, node):
    
        
    def add_node(self, parent_node, is_left, list_of_arguments):
        if isinstance(parent_node, Node):
            index = 2*parent_node.index
        else:
            index = 2*parent_node
            
        if not is_left:
            index += 1
            
        if len(list_of_arguments) == 1:
            self.node_with_index[index] = Leaf(index, list_of_arguments[0])
        else:
            self.node_with_index[index] = NotLeaf(index,
                           list_of_arguments[0], list_of_arguments[1], list_of_arguments[2])
            
            
    def remove_node(self, target_index):
        target_binarised = bin(target_index)
        self.node_with_index = dict(
            filter (lambda couple : bin(couple[0])[:len(target_binarised)] != target_binarised
                , self.node_with_index.items()))

        
    def find_subtree(self, target):
        target_binarised = ""
        if isinstance(target, Node):
            target_binarised = bin(target.index)
        else:
            target_binarised = bin(target)
        return dict(
            filter (lambda couple : bin(couple[0])[:len(target_binarised)] != target_binarised
                , self.node_with_index.items()))
        
        
    def add_subtree(self, target_index, target_node, subtree):
        diff = target_index-target_node.index
        
        def update_couple(couple):
            print(couple)
            print()
            new_couple = (couple[0]+int(math.ldexp(diff, math.floor(math.log(couple[0], 2)))), couple[1])
            new_couple[1].index = new_couple[0]
            return new_couple
        
        subtree = dict(map (update_couple, subtree.items())) 
        self.node_with_index.update(subtree)
    
    
    def print_subtree(self, index):
        node = self.node_with_index[index]
        node.print_node()
        if isinstance(node, NotLeaf):
            index = 2*index
            self.print_subtree(index)
            self.print_subtree(index +1)
    
    
    def print_tree(self):
        self.print_subtree(1)
        
    
    def predict_point(self, row, node):
        index = 0
        workingNode = None 
        if isinstance(node, Node):
            index = 2*node.index
            workingNode = node
        else:
            index = 2*node
            workingNode = self.node_with_index[node]
            
        if isinstance(workingNode, Leaf):
            return workingNode.class_name
        
        elif isinstance(workingNode, NotLeaf):
            if workingNode.relation == "<":
                if row[workingNode.left_part] < workingNode.right_part:
                    return self.predict_point(row, index)
                else:
                    return self.predict_point(row, index +1)
            elif workingNode.relation == "<=":
                if row[workingNode.left_part] <= workingNode.right_part:
                    return self.predict_point(row, index)
                else:
                    return self.predict_point(row, index +1)
            elif workingNode.relation == "==":
                if row[workingNode.left_part] == workingNode.right_part:
                    return self.predict_point(row, index)
                else:
                    return self.predict_point(row, index +1)
            elif workingNode.relation == "!=":
                if row[workingNode.left_part] != workingNode.right_part:
                    return self.predict_point(row, index)
                else:
                    return self.predict_point(row, index +1)
            elif workingNode.relation == ">":
                if row[workingNode.left_part] > workingNode.right_part:
                    return self.predict_point(row, index)
                else:
                    return self.predict_point(row, index +1)
            elif workingNode.relation == ">=":
                if row[workingNode.left_part] >= workingNode.right_part:
                    return self.predict_point(row, index)
                else:
                    return self.predict_point(row, index +1)
                
                
    def calculate_fitness(self):
        truePredictions = whole_data.apply(lambda x : 1 if self.predict_point(x, 1) == x[class_column_name] else 0,
                                             axis =1, raw = True)
        return sum(truePredictions)/whole_data.shape[0]


# In[4]:


t = Tree(["Sepal_Length", ">", 5])
t.add_node(t.node_with_index[1], True, ['setosa'])
t.add_node(t.node_with_index[1], False, ["Sepal_Width", "<", 3])
t.add_node(t.node_with_index[3], True, ['Petal_Length', '<', 6])
t.add_node(t.node_with_index[3], False, ['virginica'])
t.add_node(t.node_with_index[6], True, ['setosa'])
t.add_node(t.node_with_index[6], False, ['versicolor'])

t1 = copy.deepcopy(t)
t2 = Tree(["Petal_Width", "<=", 4])
t2.add_node(t2.node_with_index[1], True, ['versicolor'])
t2.add_node(t2.node_with_index[1], False, ['virginica'])

#sub = copy.deepcopy(t2.node_with_index)
#t.add_subtree(2, sub[1], sub)
#t.print_tree()
#t.remove_node(6)
#t.add_node(t.node_with_index[3], True, ['setosa'])
#t.print_tree()

#t2.predict_point(data.iloc[1, :], 1)
t2.calculate_fitness()


# In[ ]:




