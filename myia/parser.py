"""Parse a Python AST into the Myia graph-based ANF IR.

Graph construction proceeds very similarly to the way that FIRM constructs its
SSA graph [1]_. The correspondence between functional representations and SSA
is discussed in detail in [2]_. The main takeaway is that basic blocks
correspond to functions, and jumping from one block to another is done by a
tail call. Phi nodes become formal parameters of the function. The inputs to a
phi node become actual parameters (arguments) passed at the call sites in the
predecessor blocks.

Note that variable names in this module closely follow the ones used in [1]_ in
order to make it easy to compare the two algorithms.

.. [1] Braun, M., Buchwald, S. and Zwinkau, A., 2011. Firm-A graph-based
   intermediate representation. KIT, Fakultät für Informatik.
.. [2] Appel, A.W., 1998. SSA is functional programming. ACM SIGPLAN Notices,
   33(4), pp.17-20.

"""
import ast
import inspect
import textwrap
from typing import Dict, List

from myia.ir import Node, Parameter, Apply, Return, Graph, Constant
from myia.primops import If, Add

BLOCKS = []
RETURNS = []

CONSTANTS = {
    'if': Constant(If),
    ast.Add: Constant(Add)
}


class Block:
    """A basic block.

    A basic block is used while parsing the Python code to represent a segment
    of code (e.g. a function body, loop body, a branch). During parsing it
    keeps track of a variety of information needed to resolve variable names.

    Attributes:
        variables: A mapping from variable names to the nodes representing the
            bound value at this point of parsing. If a variable name is not in
            this mapping, it needs to be resolved in the predecessor blocks.
        phi_nodes: A mapping from parameter nodes corresponding to phi nodes to
            variable names. Once all the predecessor blocks (calling functions)
            are known, we can resolve these variable names in the predecessor
            blocks to find out what the arguments at the call site are.
        jumps: A mapping from successor blocks to the function calls that
            correspond to these jumps. This is information that was not used in
            the FIRM algorithm; it is necessary here because it is not possible
            to distinguish regular function calls from the tail calls used for
            control flow.
        matured: Whether all the predecessors of this block have been
            constructed yet. If a block is not mature and a variable cannot be
            resolved, we have to construct a phi node (i.e. add a parameter to
            this function). Once the block is mature, we will resolve the
            variable in the parent blocks and use them as arguments.
        preds: The predecessor blocks of this block.
        function: The ANF function graph corresponding to this basic block.

    """

    def __init__(self, preds: List['Block'] = None) -> None:
        """Construct a basic block.

        Constructing a basic block also constructs a corresponding function,
        and a constant that can be used to call this function.

        """
        self.matured: bool = False
        self.variables: Dict[str, Node] = {}
        self.preds = preds
        self.phi_nodes: Dict[Parameter, str] = {}
        self.jumps: Dict[Block, Apply] = {}
        self.function: Graph = Graph()
        CONSTANTS[self.function] = Constant(self.function)

        # TODO Remove this
        BLOCKS.append(self)

    def set_phi_arguments(self, phi: Parameter) -> None:
        """Resolve the arguments to a phi node."""
        varnum = self.phi_nodes[phi]
        for pred in self.preds:
            arg = pred.read(varnum)
            jump = pred.jumps[self]
            jump.inputs.append(arg)
        # TODO remove_unnecessary_phi(phi)

    def mature(self) -> None:
        for phi in self.function.parameters:
            if phi in self.phi_nodes:
                self.set_phi_arguments(phi)
        self.matured = True

    def read(self, varnum: str) -> Node:
        if varnum in self.variables:
            return self.variables[varnum]
        if self.matured and len(self.preds) == 1:
            return self.preds[0].read(varnum)
        phi = Parameter(self.function)
        phi.debug['tag'] = varnum
        self.function.parameters.append(phi)
        self.phi_nodes[phi] = varnum
        self.write(varnum, phi)
        if self.matured:
            self.set_phi_arguments(phi)
        return phi

    def write(self, varnum: str, node: Node) -> None:
        self.variables[varnum] = node

    def jump(self, target: 'Block') -> Apply:
        """Jumping from one block to the next becomes a continuation."""
        jump = Apply([CONSTANTS[target.function]], self.function)
        jump.debug['tag'] = 'jump ' + target.function.debug.get('tag', 'UNK')
        self.jumps[target] = jump
        return jump

    def cond(self, cond: Node, true: 'Block', false: 'Block') -> Apply:
        inputs = [CONSTANTS['if'],
                  cond, CONSTANTS[true.function], CONSTANTS[false.function]]
        if_ = Apply(inputs, self.function)
        if_.debug['tag'] = 'if'
        return if_


def process_function(block, node):
    function_block = Block([] if block is None else [block])
    function_block.mature()
    function_block.function.debug['tag'] = node.name
    for arg in node.args.args:
        anf_node = Parameter(function_block.function)
        anf_node.debug['tag'] = arg.arg
        function_block.function.parameters.append(anf_node)
        function_block.write(arg.arg, anf_node)
    process_statements(function_block, node.body)
    return block


def process_return(block, node):
    r = Return(process_expression(block, node.value), block.function)
    RETURNS.append(r)
    return block


def process_assign(block, node):
    anf_node = process_expression(block, node.value)
    anf_node.debug['tag'] = node.targets[0].id
    block.write(node.targets[0].id, anf_node)
    return block


def process_expression(block, node):
    if isinstance(node, ast.BinOp):
        inputs_ = [process_expression(block, node.left),
                   process_expression(block, node.right)]
        return Apply([CONSTANTS[type(node.op)]] + inputs_, block.function)
    elif isinstance(node, ast.Name):
        return block.read(node.id)
    elif isinstance(node, ast.Num):
        if node.n not in CONSTANTS:
            CONSTANTS[node.n] = Constant(node.n)
        return CONSTANTS[node.n]
    else:
        raise ValueError(node)


def process_statements(block, nodes):
    for node in nodes:
        block = process_statement(block, node)
    return block


def process_statement(block, node):
    if isinstance(node, ast.Assign):
        return process_assign(block, node)
    elif isinstance(node, ast.FunctionDef):
        return process_function(block, node)
    elif isinstance(node, ast.Return):
        return process_return(block, node)
    elif isinstance(node, ast.If):
        return process_if(block, node)
    elif isinstance(node, ast.While):
        return process_while(block, node)
    else:
        raise ValueError(node)


def process_if(block, node):
    true_block = Block([block])
    false_block = Block([block])
    after_block = Block([])
    after_block.function.debug['tag'] = 'cont'
    cond = process_expression(block, node.test)
    true_block.mature()
    false_block.mature()

    # Process the first branch
    true_end = process_statements(true_block, node.body)
    true_end.function.debug['tag'] = 'true'
    after_block.preds.append(true_end)
    Return(true_end.jump(after_block), true_end.function)

    # And the second
    false_end = process_statements(false_block, node.orelse)
    false_end.function.debug['tag'] = 'false'
    after_block.preds.append(false_end)
    Return(false_end.jump(after_block), false_end.function)

    Return(block.cond(cond, true_block, false_block), block.function)
    after_block.mature()
    return after_block


def process_while(block, node):
    header_block = Block([block])
    body_block = Block([header_block])
    after_block = Block([header_block])
    header_block.function.debug['tag'] = 'loop_cond'
    body_block.function.debug['tag'] = 'loop_body'
    after_block.function.debug['tag'] = 'cont'

    Return(block.jump(header_block), block.function)

    cond = process_expression(header_block, node.test)
    body_block.mature()
    Return(header_block.cond(cond, body_block,
           after_block), header_block.function)
    after_body = process_statements(body_block, node.body)
    header_block.preds.append(after_body)
    Return(after_body.jump(header_block), after_body.function)
    header_block.mature()
    after_block.mature()
    return after_block


def func2anf(func):
    tree = ast.parse(textwrap.dedent(inspect.getsource(func)))
    process_statement(None, tree.body[0])
    # TODO Fill in the missing phi node inputs
