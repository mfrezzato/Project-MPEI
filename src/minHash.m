%NMEC: 125793
%NMEC: 125487

classdef minHash
    properties
        NumHashes % Número de funções de dispersão (k)
        Prime     % Número primo para a operação de módulo (p)
        A         % Vetor de coeficientes 'a'
        B         % Vetor de coeficientes 'b'
    end

    methods

        % Construtor do Módulo
        function obj = minHash(numHashes)
            obj.NumHashes = numHashes;

            % Número primo de Mersenne (2^31 - 1) para evitar colisões no mod
            obj.Prime = 2147483647; 

            stream = RandStream('mt19937ar', 'Seed', 98765);

            % Gerar coeficientes aleatórios únicos para as k funções de hash
            obj.A = randi(stream, [1, obj.Prime - 1], numHashes, 1);
            obj.B = randi(stream, [0, obj.Prime - 1], numHashes, 1);
        end

        % Gerador de Shingles Avançado (caracteres ou palavras)
        function shingles = createShingles(~, text, kSize, mode)
            if nargin < 4
                mode = 'chars'; % Por omissão, faz shingling por caracteres
            end

            % Normalização e limpeza base do texto
            text = lower(strtrim(text));
            
            if strcmp(mode, 'words')
                % Divide o texto por espaços em células de palavras
                tokens = strsplit(regexprep(text, '[^a-z0-9\s]', ' '));
                tokens = tokens(~cellfun(@isempty, tokens));
                
                numShingles = length(tokens) - kSize + 1;
                if numShingles <= 0, shingles = {}; return; end
                
                shingles = cell(1, numShingles);
                for i = 1:numShingles
                    % Junta as K palavras consecutivas com um espaço
                    shingles{i} = strjoin(tokens(i : i + kSize - 1), ' ');
                end
            else
                % Modo clássico: Shingles por caracteres
                numShingles = length(text) - kSize + 1;
                if numShingles <= 0, shingles = {}; return; end

                shingles = cell(1, numShingles);
                for i = 1:numShingles
                    shingles{i} = text(i : i + kSize - 1);
                end
            end

            shingles = unique(shingles);
        end

        % Cálculo da assinatura com vetorização, segura contra overflow
        function signature = getSignature(obj, elements)
            if isempty(elements)
                signature = inf(obj.NumHashes, 1);
                return;
            end

            % Se os dados forem texto , converte para IDs numéricos de forma segura
            if iscell(elements) || isstring(elements)
                numericIds = zeros(1, length(elements));
                for i = 1:length(elements)
                    str = double(elements{i});
                    
                    h = 0;
                    for b = 1:numel(str)
                        h = mod(h * 31 + str(b), obj.Prime);
                    end
                    numericIds(i) = h;
                end
            else
                % Se os IDs já forem numéricos
                numericIds = double(reshape(elements, 1, []));
            end

            % Vetorização em Bloco: Matriz resultante (k x M)
            allHashes = mod(obj.A * numericIds + obj.B, obj.Prime);
            
            % O MinHash define a assinatura como o valor mínimo de cada linha
            signature = min(allHashes, [], 2);
        end

        % Comparar duas Assinaturas
        function similarity = compareSignatures(~, sig1, sig2)
            similarity = sum(sig1 == sig2) / length(sig1);
        end

        % Compara todas as assinaturas entre si de forma ultra-vetorizada
        function J_matrix = computeSimilarityMatrix(obj, Signatures)
            Nu = size(Signatures, 2);
            J_matrix = zeros(Nu, Nu);
            k_hash = obj.NumHashes;
            
            for n1 = 1:Nu-1
                coincidencias = sum(Signatures(:, n1) == Signatures(:, n1+1:Nu), 1);
                J_matrix(n1, n1+1:Nu) = coincidencias / k_hash;
            end
            J_matrix = J_matrix + J_matrix';
            J_matrix(1:Nu+1:end) = 1;
        end

        % Calcular a Distância de Jaccard estimada
        function distance = distanceJaccard(obj, sig1, sig2)
            if nargin < 3
                % Se passar apenas uma matriz de semelhanças pré-calculada
                distance = 1 - sig1;
            else
                distance = 1 - obj.compareSignatures(sig1, sig2);
            end
        end
    end
end