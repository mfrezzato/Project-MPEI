classdef MinHash
    properties
        NumHashes % Número de funções de dispersão (k)
        Prime     % Número primo grande para a operação de módulo (p)
        A
        B