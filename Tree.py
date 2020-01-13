
# coding: utf-8

# In[2]:


import pandas as pd
import copy

data = pd.read_csv('iris.csv')
column_names = data.columns
column_names = set(column_names.drop('Species'))
class_column_name = "Species"
class_variable = data['Species']
data = data.drop('Species', axis = 1)

relations = {'num':{'<', '<=', '>', '>='}, 'cat':{'==', '!='}}
atributes = {'num':set(filter(lambda x : data[x].dtype=='float64' or data[x].dtype== 'int64', column_names)),
             'cat':set(filter(lambda x : data[x].dtype!= 'float64' and data[x].dtype!= 'int64', column_names))}
class_names = set(class_variable)


# In[3]:


import pandas as pd

data = pd.read_csv("iris.csv")


# In[15]:


X = data.drop("Species", axis = 1)
y = data["Species"]


# In[4]:


class Node:
    def __init__(self, index):
        self.index = index


# In[5]:


class Leaf(Node):
    def __init__(self, index, class_name):
        Node.__init__(self, index)
        self.class_name = class_name


# In[6]:


class NotLeaf(Node):
    def __init__(self, index, left_part, relation, right_part):
        Node.__init__(self, index)
        self.left_part = left_part
        self.relation = relation
        self.right_part = right_part
        self.left_node = None
        self.right_node = None


# In[7]:


def print_nodes(node):
    print(node.index, end = " ")
    if isinstance(node, Leaf):
        print(node.class_name)
    elif isinstance(node, NotLeaf):
        print(node.left_part, " ", node.relation, " ", node.right_part)
        print_nodes(node.left_node)
        print_nodes(node.right_node)


# In[ ]:


def print_one_node(node):
    if isinstance(node,Leaf):
        print("Leaf: ", node.class_name, node.index)
    elif isinstance(node, NotLeaf):
        print("NotLeaf: ", node.left_part, node.relation, node.right_part, node.index)


# In[12]:


def predict_point(row, node):
    if isinstance(node, Leaf):
        return node.class_name
    elif isinstance(node, NotLeaf):
        if node.relation == "<":
            if row[node.left_part] < node.right_part:
                return predict_point(row, node.left_node)
            else:
                return predict_point(row, node.right_node)
        elif node.relation == "<=":
            if row[node.left_part] <= node.right_part:
                return predict_point(row, node.left_node)
            else:
                return predict_point(row, node.right_node)
        elif node.relation == "==":
            if row[node.left_part] == node.right_part:
                return predict_point(row, node.left_node)
            else:
                return predict_point(row, node.right_node)
        elif node.relation == "!=":
            if row[node.left_part] != node.right_part:
                return predict_point(row, node.left_node)
            else:
                return predict_point(row, node.right_node)
        elif node.relation == ">":
            if row[node.left_part] > node.right_part:
                return predict_point(row, node.left_node)
            else:
                return predict_point(row, node.right_node)
        elif node.relation == ">=":
            if row[node.left_part] >= node.right_part:
                return predict_point(row, node.left_node)
            else:
                return predict_point(row, node.right_node)    


# In[13]:


class Tree:        
    def __init__(self, node_arguments):
        if len(node_arguments) == 1:
            self.root_node = Leaf(1, node_arguments[0])
        else:
            self.root_node = NotLeaf(1, node_arguments[0], node_arguments[1], node_arguments[2])
        self.setOfIndexes = {1}
        
    def print_tree(self):
        print_nodes(self.root_node)
        
    def calculate_fitness(self):
        y_pred = ["0"] * len(data.index)
        for i in range(len(data.index)):
            y_pred[i] = predict_point(data.iloc[i], self.root_node)
            
        n_rows = len(y_pred)
        predicted = 0
        for i in range(n_rows):
            if (y_pred[i] == class_variable[i]):
                predicted += 1
                
        return predicted/n_rows
    
    def add_node(self, parent_node, is_left, list_of_arguments):
        
        parent_index = parent_node.index
        node = None
        if is_left:
            index = 2*parent_index
        else:
            index = 2*parent_index+1
        if len(list_of_arguments) == 1:
            node = Leaf(index, list_of_arguments[0])
        else:
            node = NotLeaf(index,
                           list_of_arguments[0], list_of_arguments[1], list_of_arguments[2])
        if is_left:
            parent_node.left_node = node
        else:
            parent_node.right_node = node
        self.setOfIndexes.add(index)
            
    def index_of(self, target_index):
        binarised = bin(target_index)[3:]
        node = self.root_node
        for i in range(len(binarised)):
            if binarised[i] == "0":
                node = node.left_node
            else:
                node = node.right_node  
        return node     
    
    def remove_node(self, target_index):
        if target_index == 1:
            self.root_node = None
            return
        target_binarised = bin(target_index)
        self.setOfIndexes = set(
            filter (lambda index : bin(index)[:len(target_binarised)] != target_binarised
                , self.setOfIndexes))
        parent_index = target_index // 2
        node = self.index_of(parent_index)
        if(target_index % 2 == 0):
            node.left_node = None
        else:
            node.right_node = None
        
    def add_subtree(self, target_index, target_node):
        if target_index in self.setOfIndexes:
            self.remove_node(target_index)
        if target_index == 1:
            self.root_node = target_node
            return
        parent_index = target_index // 2
        node = self.index_of(parent_index)
        if(target_index % 2 == 0):
            node.left_node = target_node
        else:
            node.right_node = target_node
        self.update_index(target_index, target_node)
            
    def update_index(self, new_index, target_node):
        target_node.index = new_index
        self.setOfIndexes.add(new_index)
        if isinstance(target_node, NotLeaf):
            self.update_index(2*new_index, target_node.left_node)
            self.update_index(2*new_index + 1, target_node.right_node)           


# In[14]:


t = Tree(["Sepal_Length", ">", 5])
t.add_node(t.root_node, True, ['setosa'])
t.add_node(t.root_node, False, ["Sepal_Width", "<", 3])
t.add_node(t.root_node.right_node, True, ['Petal_Length', '<', 6])
t.add_node(t.root_node.right_node, False, ['virginica'])
t.add_node(t.root_node.right_node.left_node, True, ['setosa'])
t.add_node(t.root_node.right_node.left_node, False, ['versicolor'])

t.calculate_fitness()

nl = NotLeaf(1, "Petal_Width", "<=", 4)
l1 = Leaf(2, "versicolor")
l2 = Leaf(3, "virginica")
nl.left_node = l1
nl.right_node = l2
t.add_subtree(2, nl)

