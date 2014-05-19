import re


class ParseTree:

    def __init__(self, tag=None, isTerminal=False):
        self.children = []
        self.tag = tag
        self.isTerminal = isTerminal
        self.info = {}

    def fromString(self, s):
        m = re.match("^\(?(.*?)\)?$", s)
        s = m.group(1)
        m = re.match("^(.*?)\W(.*)$", s)
        self.tag = m.group(1)
        s = m.group(2)

        self.children = []
        paren_counter = 0
        beginning_block = None
        contains_parenthesis = False
        for j, c in enumerate(s):
            if c == '(':
                contains_parenthesis = True
                if paren_counter == 0:
                    beginning_block = j
                paren_counter += 1
            elif c == ')':
                paren_counter -= 1
                if paren_counter == 0:
                    child = ParseTree()
                    child.fromString(s[beginning_block+1:j+1])
                    self.children.append(child)
        if not contains_parenthesis:
            child = ParseTree(tag=s, isTerminal=True)
            self.children.append(child)
        return self

    def outputWordList(self):
        if self.isTerminal:
            return [self.tag]
        word_list = []
        for child in self.children:
            word_list.extend(child.outputWordList())
        return word_list

    def computeLength(self):
        if self.isTerminal:
            self.info['length'] = len(self.tag)
            return self.info['length']
        else:
            l = 0
            for child in self.children:
                child.computeLength()
                l += child.info['length']
            l += len(self.children) - 1
            self.info['length'] = l
            return l

    def displayTree(self, indentationLevel=0):
        print ("|  " * max(0, indentationLevel-1) + "|--" *
               (indentationLevel > 0) + self.tag)
        indentationLevel += 1
        for child in self.children:
            child.displayTree(indentationLevel=indentationLevel)
