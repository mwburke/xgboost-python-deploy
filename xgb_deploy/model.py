import math


class ProdEstimator:
    """
    Pure python production version of an XGBoost model that is defined
    by the JSON model dump.

    Extracts trees from the model dump, and returns the total leaf value
    added to the base_score as the prediction for regression problems, and
    transforms it with the logit function for classification problems.
    """

    def __init__(self, model_data, pred_type, base_score=0.5):
        """
        Initialize estimator with type of problem and create trees used
        in estimation.

        Args:
            model_data: List of python dictionaries created from importing
                JSON model dump of the original XGBoost model.
            pred_type: Either 'classification' or 'regression'.
            base_score: Input parameter used in training the XGBoost model.
        """
        if (pred_type == 'classification') & (base_score != 0.5):
            raise ValueError('For classification, please train XGB model with default base score value of 0.5')

        self.pred_type = pred_type
        self.base_score = base_score
        self.model_data = model_data
        self.build_trees(model_data)

    def build_trees(self, model_data):
        """
        For each dictionary in the model_data, create a TreeEstimator
        for use in prediction later.

        Args:
            model_data: List of python dictionaries created from importing
                JSON model dump of the original XGBoost model.
        """
        self.trees = []

        for tree_data in model_data:
            self.trees.append(TreeEstimator(tree_data))

    def predict(self, data):
        """
        Perform predictions over all input data.

        Args:
            data: List of dictionaries containing the input features
                and associated values.
        """
        if len(data) == 1:
            return self._predict_row(data[0])

        return [self._predict_row(row) for row in data]

    def get_leaf_values(self, data, node_value=True):
        """
        Extract the raw leaf prediction values for each tree.

        Args:
            data: List of dictionaries containing the input features
                and associated values.
            node_value: Boolean flag on whether or not to return the
                node_value or the node_id

        Returns:
            leaf_values: List of leaf value predictions for each tree.
        """
        leaf_values = [tree.get_leaf_value(data, node_value) for tree in self.trees]

        return leaf_values

    def new_predict(self, tree, inst):
        """
        Recursive function that takes in the dictionary used to create a tree.
        Found after I already started work on this, but it's useful.

        Removes the need for OOP and makes some additional assumptions that
        don't hold up with when you use binary variables.
        Kept here for reference, but not mentioned in the README.
        """
        if 'children' in tree:
            feature_id = tree['split']
            threshold = tree['split_condition']
            # default direction for missing value
            default_left = (tree['missing'] == tree['yes'])
            if feature_id not in inst:  # missing value
                return self.new_predict(tree['children'][0 if default_left else 1], inst)
            elif inst[feature_id] < threshold:  # test is true, go to left child
                return self.new_predict(tree['children'][0], inst)
            else:   # test is false, go to right child
                return self.new_predict(tree['children'][1], inst)
        else:
            return tree['leaf']

    def _new_predict_row(self, data):
        """
        Kept here for reference, but not mentioned in the README.
        """
        total_leaf_value = sum([self.new_predict(tree, data) for tree in self.model_data])

        if self.pred_type == 'regression':
            prediction = total_leaf_value + self.base_score
        if self.pred_type == 'classification':
            prediction = 1. / (1. + math.exp(-total_leaf_value))

        return prediction

    def _predict_row(self, data):
        """
        Predict the data for an individual input by summing up the
        predicted values for each tree in this estimator and returning
        the value added ot the base model score.
        If the prediction type is classification, perform the logit
        transformation returning.

        Args:
            data: Dictionary containing the input features and values.

        Returns:
            prediction: Predicted value if regression or predicted
                probability if classification.
        """
        total_leaf_value = sum([tree.get_leaf_value(data) for tree in self.trees])

        if self.pred_type == 'regression':
            prediction = total_leaf_value + self.base_score
        if self.pred_type == 'classification':
            prediction = 1. / (1. + math.exp(-total_leaf_value))

        return prediction


class TreeEstimator:
    """
    Class to represent a single decision tree in a larger
    XGBoost booster model. Stores information for the decision
    tree and returns the leaf value given input data to be
    aggregated with all other tree data.
    """

    def __init__(self, node_data):
        self.build_tree(node_data)

    def build_tree(self, node_data):
        """
        Using the node definition data, build all of the
        hierarchical tree nodes that will be used to find
        the leaf value prediction given some input data.

        Args:
            node_data: Dictionary of hierarchical node data
                created from reading a JSON dump of the
                XGBoost model.
        """
        self.root = TreeNode(node_data)

    def get_leaf_value(self, data, node_value=True):
        """
        Find the leaf value prediction given some input data.

        Args:
            data: Python dictionary of a single data point
                that needs a prediction
            node_value: Boolean flag on whether or not to return the
                node_value or the node_id

        Returns:
            leaf_value: Value of the final leaf TreeNode value
                the data ended up at through the decision process.
        """
        leaf_value = self.root.leaf_value(data, node_value)

        return leaf_value


class TreeNode:
    """
    Class to represent a single node in a decision tree.
    Stores information on final value to return if leaf node,
    and information on children nodes if not, as well as
    information on how to decide which leaf node to choose
    based on the input data.
    """

    def __init__(self, node_data):
        """
        If leaf node, only store leaf value. Otherwise store
        additional information on depth and selection criteria
        as well as storing child node information.
        """
        self.leaf = node_data['leaf'] if 'leaf' in node_data else None
        self.node_id = node_data['nodeid']
        if self.leaf is None:
            self.depth = node_data['depth']
            self.split_feature = node_data['split']
            self.threshold = node_data['split_condition'] if 'split_condition' in node_data else None
            self.yes_id = node_data['yes']
            self.no_id = node_data['no']
            self.missing_id = node_data['missing'] if 'missing' in node_data else None
            self.children = {}
            self.create_children(node_data['children'])

    def create_children(self, children):
        """
        Create child nodes for each of the node's children.

        Args:
            children: List of dictionaries defining child
                nodes and their children, etc.
        """
        for child in children:
            self.create_child(child)

    def create_child(self, child):
        """
        Create another TreeNode for the input child node
        data and assign it to a dictionary of the children
        with its node_id as its identifier.

        Args:
            child: Dictionary of node information and any
                node's children's information, etc.
        """
        self.children[child['nodeid']] = TreeNode(child)

    def leaf_value(self, data, node_value):
        """
        Search for the leaf value in the tree given the input
        data.

        If the node is a leaf, the tree has found the
        value and returns it.

        If the node is not a leaf, it checks the split_feature
        in the input data and compares it to the node's conditions
        to find the child node_id it should continue searching down.

        Args:
            data: Python dictionary of a single data point
                that needs a prediction
            node_value: Boolean flag on whether or not to return the
                node_value or the node_id
        """

        # Return value if leaf
        if self.leaf is not None:
            if node_value:
                return self.leaf
            else:
                return self.node_id

        # If feature not in data, use missing condition
        if self.split_feature not in data:
            return self.children[self.missing_id].leaf_value(data, node_value)

        # Feature value in data to use in node selection
        feature_value = data[self.split_feature]

        # If missing condition possible, check value if null
        if self.missing_id is not None:
            if math.isnan(feature_value):
                return self.children[self.missing_id].leaf_value(data, node_value)

        # If value is not null, and value type is either integer
        # or quantitative, check if value is less than or greater
        # than the decision threshold and search appropriate child node.
        if self.threshold is not None:
            if feature_value >= self.threshold:
                return self.children[self.no_id].leaf_value(data, node_value)
            else:
                return self.children[self.yes_id].leaf_value(data, node_value)
        else:
            if feature_value == 0.0:
                return self.children[self.no_id].leaf_value(data, node_value)
            else:
                return self.children[self.yes_id].leaf_value(data, node_value)
