import numpy as np
import pandas as pd
import math
import pdb
import random 
import time
import psycopg2
import credentials as cred




class NodeNotSplittableError(Exception):
    pass


# Read the data as pandas data frame
# @INPUT: file name eg) 'binary.csv'
# @OUTPUT: n x(p+1) data frame  where columns are either response, y, or a covariate, x.
# y should be binary either 0 or 1
# x should be numerical values
# Every row is an observation, so we have n observations. 
def read_data(fname):
    df = pd.read_csv(fname)
    return  df
# Will look something like
#    admit  gre   gpa  rank
# 0      0  380  3.61     3
# 1      1  660  3.67     3
# 2      1  800  4.00     1
# 3      1  640  3.19     4
# 4      0  520  2.93     4

# Parse data into response and predictors
# @INPUT: pandas dataframe and y_val which is the column name of response, eg) 'admit'
# @OUTPUT: response is n x 1 pandas Series. Every value should be either 0 or 1
#          predictors is n x p pandas data frame, so we have p number of covariates. 
def parse_data(df, y_val):
    response = df[y_val] 
    predictors = df.drop(y_val, axis = 1)
    # Drop a column with the same value for all entries 
    predictors = predictors.loc[:, predictors.apply(pd.Series.nunique) != 1]

#   predictor_names = list(df.columns)
    return response, predictors 


# prob_y_1(response) = P(y=1| response) 
# @INPUT: y_count = [ the number of 1s, the number of 0s ] in the response
# This calculates  the probability of y being 1 given response regardless of the size of the response
def prob_y_1(y_count):
    #print("len(response) in prob_y_1: ", len(response))
    p = y_count[1] / sum(y_count)
    # p = response.value_counts(1)[1    ]
    return p

# Three ways of defining impurity
# @INPUT: p from prob_y_1 
# @OUTPUT: impurity
def bayes_impurity(p):
    return min(p, 1 - p)

def entrophy_impurity(p):
    return -p * math.log(p) - (1 - p) * math.log(1 - p)

def gini_impurity(p):
    return p * (1 - p )



# Provides a set of index to be used in the left child and right child  
# @INPUT: j = the column name to be used to cut
#         s = the value that will be used to cut col x_j
#         predictors = we use the cutting based on wether column j of predictors < s or >= s 
# @OUTPUT: cut_l = a boolean series where each TRUE corresponds to a single row in a left child
#          cut_r = a boolean series where each TRUE corresponds to a single row in a right child
def cut_index(j, s, predictors):
    cut_l = predictors[j] < s
    cut_r = predictors[j] >=s
    return cut_l, cut_r

# Cut a data frame into two data frames
# @INPUT: cut_l = index for a left child 
#         cut_r = index for a right child 
#          df = data frame that we would like to cut
# @OUTPUT: df_l = data frame for a left child
#          df_r = data frame for a right child
def cut_df(cut_l, cut_r, df):
    df_l = df[cut_l]
    df_r = df[cut_r]
    return df_l, df_r

# Just need to know the length of y_l and y_r these can be calculated from cut_l and cut_r
# @INPUT: cut_l and cut_r   from cut_index():
# @OUTPUT: p_l = proportion of responses that fall into a left child
#         p_r = proportion of responses that fall into a right child 
def proportion_lr(cut_l, cut_r):
    if len(cut_l) != len(cut_r):
        print("the length of the left index and the right index should still be the same. Something went wrong in cut_index()")
    else:
        len_y = len(cut_l) # len(cut_l) and len(cut_r) should be the same
        p_l = sum(cut_l) / len_y
        p_r = 1 - p_l
    return p_l, p_r

# Returns the majority of y in a node 
# @INPUT: y_count = [ the number of 0s, the number of 1s]

def majority_y(y_count):
    # If the number of 1's are greater than the number of 0s, then the majority y should be 1.
    if y_count[1] >= y_count[0]: 
        majority_y = 1
    else:
        majority_y = 0   
    return majority_y
    

# Node class builds a tree by spliting itself. 
# If something can be done on/within one node, then we put them as a method. 
class Node:
    # response: y values passed in to build the tree.  should be n x 1 
    # predictors: x values  passed in to build the tree. should be n x p 
    # impurity_fn: a type of impurity function to be used to calculate an impurity. The default is gini_impurity but the user can change.
    # k_num: the number of predictors used in spliting the data. Default to p for a classification tree where we use all columns. 
    # The user will provide k for random forest where k <= p. We only use a random selection of k columns  for spliting data. 
    def __init__(self, response, predictors, k_num = None, impurity_fn = gini_impurity):

        # Please, note that response and predictors are never saved in the tree
        # Instead, y_count will contain the number of 0s and the number of 1s in a response. This will be saved in each node. 
        self.y_count = [sum(response == 0) , sum(response)]
        self.impurity_fn = impurity_fn # so we can use it later when creating a child node.
        self.impurity = impurity_fn(prob_y_1(self.y_count)) # impurity for this specific node

        if k_num == None:
            k_num = len(list(predictors)) # This is p by default but will be an user provided k for random forest
        

        # Node class creates left child node and right child node if impurity is greater than 0
        # i.e. It splits itself until there is no more variable in response.   
        if self.impurity > 0:
            self.x_j, self.val_s = self.pick_split(response, predictors, k_num)
            
            try:
                self.left_child, self.right_child = self.create_child(response, predictors, k_num)
                
            # There are cases where no matter how we split the impurity doesn't get better. In that case, just make it as a leaf.
            except NodeNotSplittableError:
                self.left_child, self.right_child = None, None
                
            predictors = None # only leaf nodes need to have predictors data
            
        # Stops creating child nodes when impurity = 0 or when there is no way to split anymore.
        # i.e. It stops spliting itself when all response values are the same. (either all 0 or either all 1)
        else:
            self.left_child = None
            self.right_child = None
        
        # If t is not a leaf node, it has g(t) as an attribute
        if self.left_child != None or self.right_child != None:
            self.gt = self.gt()
        # If t is a leaf node, then it doesn't make sense to calculate g(t)

        # Once there are split points start appending them into parent js. 
        #if self.x_j != None or self.val_s != None:
   

    def pick_split(self, response, predictors, k_num):
    #    print("len(response) in pick_split: ", len(response))
        # col_names: the names of each column in predictors
        
        # We will select a random of k predictors from all. 
        # For a classification tree, this is default to p. i.e. We are using all predictors to pick a split point.
    
        col_names = random.sample(list(predictors), k_num)
    
        # loop through values s in col_j, calculate the impurity reduction 
        # record the best split points, chosen_s and for_j, which provides the max impurity reduction
        # For each column
        max_reduction = 0
        i = 0
        
        chosen_s = None
        for_j = None


        for j in col_names:
       
            # s loops through all unique values in col x_j
            unique_vals = predictors[j].unique()
            # Don't need to check min and max of col x_j as a cut point
            uniq_vals = unique_vals.tolist()
        
            # If there is only one unique value for this x_j column then we move on to the next column
            if len(uniq_vals) == 1:
                continue
            else:
                if len(uniq_vals ) > 2:
                    uniq_vals.remove(min(uniq_vals))
                    uniq_vals.remove(max(uniq_vals))

                for s in uniq_vals:
                    cut_l, cut_r = cut_index(j, s, predictors)
                    # If somehow this cut results in all data frame into one child, then this cut is meaningless and will give an error
                    # so take care of those cases by continue
                    if sum(cut_l) == 0 or sum(cut_r) == 0:
                        continue
                
                    y_l, y_r = cut_df(cut_l, cut_r, response)
                    y_l_count = [sum(y_l ==0), sum(y_l)]
                    y_r_count = [sum(y_r == 0), sum(y_r)]
                    p_l, p_r = proportion_lr(cut_l, cut_r)
                    impurity_l, impurity_r = self.impurity_fn(prob_y_1(y_l_count)), self.impurity_fn(prob_y_1(y_r_count))
                    left_pI, right_pI = p_l * impurity_l, p_r * impurity_r
                
                    temp_reduction = self.impurity - left_pI - right_pI
                
                    if max_reduction < temp_reduction:
                        max_reduction = temp_reduction
                    
                        chosen_s = s
                        for_j = j
                     
        if chosen_s == None or for_j == None:
            raise NodeNotSplittableError
        #assert chosen_s is not None and for_j is not None, "chosen_s or for_j should not be none"
        # What's the best j & s for the next split?
        #print("result in pick_split: chosen_s", chosen_s, "for_j", for_j)
        return for_j, chosen_s



    # For a given Node class object, split them into left and right by x_j and s 
    # node is an instance of Node class
    # node = Node()
    # so it will have response, predictors, impurity and majority.y as attributes.
    def split_data(self, response, predictors, k_num):
         
        #print("len(response) in split_data :", len(self.response))
        if len(response) == 0:
            print("length of response  = 0")
            exit()

    
        j, s = self.pick_split(response, predictors, k_num)
        # self.j = j
        # self.s = s
        
        cut_l, cut_r = cut_index(j, s, predictors)
        left_response, right_response = cut_df(cut_l, cut_r, response)
        left_predictors, right_predictors = cut_df(cut_l, cut_r, predictors)

            
        return left_predictors, left_response, right_predictors, right_response
    

    
        
    def create_child(self, response, predictors, k_num):
        #print("len(response) in create_child: ", len(self.response))
        left_predictors, left_response, right_predictors, right_response = self.split_data(response, predictors, k_num)
        
        left_child = Node(left_response, left_predictors, k_num, self.impurity_fn)
        right_child = Node(right_response, right_predictors, k_num, self.impurity_fn)

        return left_child, right_child

    # predict a y value for a new predictor
    # 
    def predict_y(self, one_row):
        # self is a current node
        # If a leaf, then take the majority y and return 0 or 1
        if is_leaf(self):
            return majority_y(self.y_count)
        # If it's not a leaf, then 
        else:      
            if one_row[self.x_j] < self.val_s:
                # if you going to left
                return self.left_child.predict_y(one_row) # predict_y(self.left_child, one_row)
             
            else:
                # if you going to right
                return self.right_child.predict_y(one_row) # predict_y(self.right_child, one_row)
    # provide a list of predicted y values for the given new_predictors 
    # @INPUT self: the node          

    def predict_y_node(self, new_predictors): 
        predicted_y = []
        for i, one_row in new_predictors.iterrows():
            predicted_y.append(self.predict_y(one_row))
        return predicted_y  
          
    # splits will be used later to select data 
    def get_splits(self, one_row, splits = None):
        if splits == None:
            splits = []

        # When we get to the leaf node, there is no more splits so return the split that you have so far        
        if is_leaf(self):
            return splits
        else:
            # If not a leaf, let's see where to go starting from self , whichever node this might be
            if one_row[self.x_j] < self.val_s:
                # if col j of this one row < val_s, that means we go to the left child 
                # as we go to the left child, we should save the split information i.e. self.x_j and self.val_s  
                # as well as a boolean element to indicate whether it was left or right.  
                # True (0<1) indicates left and False indicates right
                return self.left_child.get_splits(one_row, splits=splits + [[self.x_j, " < ", self.val_s]])
            if one_row[self.x_j] >= self.val_s:
                return self.right_child.get_splits(one_row, splits = splits + [[self.x_j, " >= ", self.val_s]])
    



    def gt(self):
        len_y = sum(self.y_count)
        gt = (misclass_a_node(self) - misclass_tree(self, len_y)) / (num_leaves(self) - 1)
        return gt
    
  
# to check whether a node is leaf or not
# @INPUT node
# @OUTPUT a boolean True if it is a leaf 
#                   False if it is not 
def is_leaf(node):

    return node.left_child == None and node.right_child == None
 

##########################PRUNNING##################################################

### NEED A FUNCTION THAT FINDS A RESPONSE GIVEN A NODE so that I can pass into misclass_a_node
def node_response(tree_node, a_leaf_node, response, predictors):
    pass
    





# r(t): calculates misclassification rate at a leaf node 
# The length of leaf_node is always 1
# so we calculate this by comparing whether y_vote is equal to actual response
def misclass_a_node(leaf_node):
    y_vote = majority_y(leaf_node.y_count)
    if y_vote == 0:
        # return the number of 1s 
        return leaf_node.y_count[1]
    else: 
        # return the number of 0s
        return leaf_node.y_count[0]


# p(t): fraction of all data points which are in leaf t
# @INPUT leaf_node
#        len_y: the length of the root node's response. 
def proportion_leaf(leaf_node, len_root_response):
    return sum(leaf_node.y_count) / len_root_response



# R(T): total misclassification cost of the tree T.
# sum of r(t)p(t) for all t leaf nodes
# @INPUT tree_node
#        len_y: the length of the root node's response
# @OUTPUT sum of r(t)p(t)
def misclass_tree(tree_node, len_root_response):
    if is_leaf(tree_node):  # If a tree_node does not have a child then, this is the only R_t so calculate it here. 
        r_t = misclass_a_node(tree_node)  
        p_t = proportion_leaf(tree_node, len_root_response)  
        R_T = r_t * p_t
        #pdb.set_trace()
    else:
        R_T = misclass_tree(tree_node.left_child, len_root_response) + misclass_tree(tree_node.right_child, len_root_response)
    return R_T


# |leaves(T)| the number of leaves within a given tree/node
def num_leaves(node):
    if is_leaf(node):
        return 1 # the leaf node itself 
    else:
        return num_leaves(node.left_child) + num_leaves(node.right_child)



# g(t): generalization error
# @INPUT: t_node that we will use to claculate the generalization error specified on page 5 of classification-tree.pdf
# @OUTPUT: generalization error of the given t_node
# def gen_error(t_node):
#     len_y = len(t_node.response)
#     g_t = (misclass_if_prunned(t_node) - misclass_tree(t_node, len_y)) / (num_leaves(t_node) - 1)
#     return g_t
# REPLACED THIS WITH gt method

def traverse_gt(node):
    """
    Traverse the tree to find the min g(t)
    Note: the leaf node does not have g(t)
    @ INPUT: non-leaf-node
    @ OUTPUT: the min g(t) value, a node to be pruned at
    """
    # to avoid    int vs. sth else comparison, set gt to be 99 for leaf nodes
    gt, left_gt, right_gt =  99, 99, 99
    prune_at, left_prune, right_prune = node, node, node
    if is_leaf(node): 
        pass
    else:
        gt = node.gt   
        #print("gt", node.gt) 
        prune_at = node   
        # leaf nodes do not have g(t)        
        if node.left_child != None:      
            left_gt, left_prune = traverse_gt(node.left_child)  

        if node.right_child != None:
            right_gt, right_prune = traverse_gt(node.right_child)

            
    gt_list = [gt, left_gt, right_gt]
    min_gt = min(gt_list)
    which_node = [prune_at, left_prune, right_prune][gt_list.index(min_gt)]

    return min_gt, which_node
    
# Now that we know which node to prune_at using traverse_gt function. 
def prune_tree(tree_node, prune_at):
    """
    Cut the tree at a prune_at node. 
    When you prune, the original data doesn't change so use num_leaves to check if it's prunned or not
    """
    if is_leaf(tree_node):
        pass
    else:
        #pdb.set_trace()
        prune_at.left_child = None
        prune_at.right_child = None
        #pdb.set_trace()

    return tree_node 

 
def repeat_pruning(tree_node, chosen_alpha): 
    """
    repeat prunning until there is no improvement in R_alpha(T)
    alpha = min g(t) where t is a non-leaf-node
    keep pruning as long as curr_alpha < alpha; once curr_alpha >= alpha, you can stop pruning the tree
    @ OUTPUT: optimally pruned tree

    """
    
    alpha_star, temp_prune_at = traverse_gt(tree_node)
    smaller_tree = prune_tree(tree_node, temp_prune_at)
    if is_leaf(smaller_tree):
        return smaller_tree
    
    elif alpha_star <= chosen_alpha:
        smaller_tree = repeat_pruning(smaller_tree, chosen_alpha)
    else: 
        return smaller_tree

# 1. Find an alpha from min of g(t) t being any non-leaf nodes
# 2. Prune the tree at t node which gives the smallest alpha
# 3. Repeat 1 and 2 until R_alpha(T), cost_complexity, doesn't get smaller 
# This happens when alpha* > alpha. i.e. stop when we get a bigger alpha than the previous alpha)

"""
we don't use the alphas we got while pruning.  Instead, we pick some sequence of alphas the user thinks
might be reasonable for their tree -- maybe 0.1, 0.2, ... 1.0, or something else they pick. We then get the misclassification error
like you show in step 5, and pick the alpha with the smallest misclassification error. - Alex
"""

# Divide data into K folds 
# @INPUT df from read_data
# @OUTPUT  df_split list containing K data frame partitions
def K_fold(df, K=5):
    n_row = df.shape[0]
    indx = list(range(n_row))
    random.shuffle(indx) ## SO WE DON'T REASSIGN HERE ???
    fold_size = int(n_row / K)
    df_split = list()
    for k in range(K):
        sub_indx = indx[fold_size*k: fold_size*(k+1)]
        df_split.append(df.loc[sub_indx])
        #train = df.drop(df.index[sub_indx])
    #print("df split in K fold, ", df_split)
    return df_split

# Set one to test and others to train

def test_train(df_split, i):
    test = df_split[i]
    train_list = [x for j, x in enumerate(df_split) if j!=i]
    # merge others into one train data frame
    train = pd.concat(train_list)
    return test, train ## NOT RIGHT YET MAYBE WE DONT NEED THIS FUNCTION 




def create_a_tree(df, y_val):
    response, predictors = parse_data(df, y_val)
    tree = Node(response, predictors)
    return tree

#################   CHOOSING THE BEST ALPHA, which we will denote as a #############################################


# average error on the test set across the k test folds
# @INPUT df
#        y_val: the name of the response column eg) 'admit'
def avg_err_kfolds(df, y_val, alpha, K = 5):
    
    kfolds = K_fold(df, K)
    sum_err = 0
    for i in range(K):
        test, train = test_train(kfolds, i)
        test_response, test_predictors = parse_data(test, y_val)
        train_tree = create_a_tree(train, y_val)
        prune_tree = repeat_pruning(train_tree, alpha)
        # Ask prune_tree to predict y values for the test predictors
        predicted_y = prune_tree.predict_y_node(test_predictors)
        # Compare predicted_y to test_response, true value
        sum_err += len(predicted_y != test_response) / len(predicted_y)
    
    avg_test_err = sum_err / K
    return avg_test_err


# 4. save alphas into a list
# 5. For each alpha in the list, get the average cv test error 
#   5a. train - test set 
#   5b. build tree using train and test it on test set
# 6. Choose the alpha which gives the smallest average cv test error 


# Given the alpha list, find the alpha that gives the min average test error
# alpha_ls = np.arange(0, 1, 0.1)
def choose_best_alpha(alpha_ls, df, y_val):
    min_test_err = 999
    best_alpha = None
    for a in alpha_ls:
        avg_test_err = avg_err_kfolds(df, y_val, a, K = 5)
        if avg_test_err < min_test_err:
            min_test_err = avg_test_err
            best_alpha = a
    return best_alpha


def build_best_tree(df, y_val):
    alpha_ls = np.arange(0, 0.2, 0.05)
    best_alpha = choose_best_alpha(alpha_ls, df, y_val)
    tree = create_a_tree(df, y_val)
    best_tree = repeat_pruning(tree, best_alpha)
    return best_tree


    



#####################################################
################ RANDOM FOREST ######################
#####################################################


# @ INPUT: df = pandas data frame with all data
#          sample_n = sample size used in each tree
#          k = the number of features used in each tree
#          num_tree = how many trees in the forest
# @ OUTPUT: forest   which is a list that contains num_tree Node classes 
def random_forest(df, y_val, sample_n, k, num_tree = 500):
    # pandas sample takes a row-wise sample
    sample_df = df.sample(sample_n)
    s_response, s_predictors = parse_data(sample_df, y_val)
   
    
    forest = []
    for i in range(num_tree):
        tree = Node(s_response, s_predictors, k)
        forest.append(tree)
        return forest

    
# Given a split values, provide the data region that satisfies the splits

def get_data_in_region(df, splits):
    for i in splits:
        if i[1] == " < ":
            
            df = df[df[i[0]] < i[2]]
            
        else: 
            df = df[df[i[0]] >= i[2]]
    return df


############################################
##### CONNEC TO SQL #########################


# Load a table in SQL into a pandas data frame
# @ INPUT table name the one in SQL  eg) table_name = "admit50"
#         host eg) "sculptor.stat.cmu.edu"
#         db: database name  eg) "soyoungl"   
#         db_user: user name eg)"soyoungl"
#         db_pw: password eg) cred.DB_PW
# @ OUTPUT: pandas data frame that  has the same info as the table
def load_data(table_name, host, db, db_user, db_pw):
    conn = psycopg2.connect(host=host, database=db, user=db_user, password= db_pw)

    df = pd.read_sql("SELECT * FROM " + table_name, conn)
    return df


# This SQL_Tree class builds the tree just like Node tree builds the tree. 
# The only difference is that this uses the table in SQL database instead of a dataframe stored in python
class SQL_Tree(Node):

    def __init__(self, table_name, host, db, db_user, db_pw, y_val):
        # Alex said it's okay to load the data as a data frame while building the tree as long as the data are not stored in the sql tree
        df = load_data(table_name, host, db, db_user, db_pw)
        response, predictors = parse_data(df, y_val)
        super().__init__(response = response, predictors = predictors) # This gets all attributes saved ( self.stuff) in the parent class 





# @INPUT: Node_tree = a tree built from Node class
#         df = data frame that is used to build the tree Node_node
#         one_row = a row that you want to find which node this row belongs to
# Given a row (a set of predictor), find the corresponding leaf node and return the data that are in that node
def corresponding_leaf_node_data(Node_tree, df, one_row):

    splits = Node_tree.get_splits(one_row)
    #data = pd.DataFrame()
    # splits:[('gre', ' >= ', 520), ('rank', ' >= ', 2), ('gpa', ' < ', 3.75), ('gpa', ' >= ', 3.22)]

    return get_data_in_region(df, splits)


def splits_into_one_string(splits):
    command = ""
    for sj in splits:
        command += " ".join(str(x) for x in sj)
        # If sf is not the last item
        if sj != splits[-1]:
            # need to add AND
            command += " AND "
    return command


def show_sql_data(sql_tree, table_name, y_val, one_row, conn):
    splits = sql_tree.get_splits(one_row)
    region = splits_into_one_string(splits)
    cur = conn.cursor()
    cur.execute("SELECT * FROM " + table_name + " WHERE " + region)
    for row in cur:
        print(row)



# So that I can compare with  get_data_in_region from node class
def get_sql_data(sql_tree, table_name, y_val, one_row, conn):
    splits = sql_tree.get_splits(one_row)
    region = splits_into_one_string(splits)
    data = pd.read_sql("SELECT * FROM " + table_name + " WHERE " + region, conn)
    return data

def predict_sql_data(sql_tree, table_name, y_val, one_row, conn):
    splits = sql_tree.get_splits(one_row)
    region = splits_into_one_string(splits)
    predicted_y = pd.read_sql("SELECT " + y_val + " FROM " + table_name + " WHERE " + region 
                            + " GROUP BY " +y_val +  " ORDER BY count(*)  DESC LIMIT 1" , conn)
    #predicted_y = response.mode()  Now SQL finds the mode
    return predicted_y






