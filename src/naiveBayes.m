classdef naiveBayes
    properties
        mode          % 'multinomial' ou 'bernoulli'
        classes       % Cell array com as classes únicas
        vocabulary    % Vetor com a lista de palavras únicas
        logPrior      % Probabilidades a priori das classes em log
        logLikelihood % Probabilidades das palavras dada a classe em log P(W|C)
        vocabSize     % Tamanho do vocabulário
    end

    methods

        % Construtor do Módulo
        function obj = naiveBayes(mode)
            % Se não passar argumento assume o modo como multinomial
            if nargin < 1
                mode = 'multinomial';
            end
            obj.mode = mode;
        end

        % Método de treino do Naïve Bayes
        function obj = train(obj, documents, labels)
            % Normalizar: garantir que documents e labels são cell arrays de char
            documents = obj.toCellStr(documents);
            labels    = obj.toCellStr(labels);

            obj.classes    = unique(labels);
            obj.vocabulary = obj.buildVocabulary(documents);
            obj.vocabSize  = numel(obj.vocabulary);
            numClasses     = numel(obj.classes);

            % Probabilidades a priori P(classe) em espaço logarítmico
            obj.logPrior = zeros(1, numClasses);
            for i = 1:numClasses
                count           = sum(strcmp(labels, obj.classes{i}));
                obj.logPrior(i) = log(count / numel(labels));
            end

            % P(palavra|classe) com Laplace smoothing (add-1) em espaço logarítmico
            obj.logLikelihood = zeros(numClasses, obj.vocabSize);
            for i = 1:numClasses
                classDocs = documents(strcmp(labels, obj.classes{i}));
                counts    = zeros(1, obj.vocabSize);
                for d = 1:numel(classDocs)
                    words = obj.tokenize(classDocs{d});

                    % Se for modo Bernoulli, binariza o documento removendo palavras repetidas
                    if strcmp(obj.mode, 'bernoulli')
                        words = unique(words);
                    end

                    for w = 1:numel(words)
                        idx = find(strcmp(obj.vocabulary, words{w}), 1);
                        if ~isempty(idx)
                            counts(idx) = counts(idx) + 1;
                        end
                    end
                end
                obj.logLikelihood(i, :) = log((counts + 1) / (sum(counts) + obj.vocabSize));
            end
        end

        % Método para classificar múltiplos documentos
        function predictions = classify(obj, documents)
            % Normalizar para cell array de char
            documents = obj.toCellStr(documents);
            predictions = cell(1, numel(documents));
            for d = 1:numel(documents)
                predictions{d} = obj.classifyOne(documents{d});
            end
        end

        % Devolve a probabilidade estimada de cada classe para um documento
        function probs = probability(obj, document)
            if ~ischar(document)
                document = char(document);
            end
            [~, logScores] = obj.classifyOne(document);
            % Ajuste para evitar underflow/overflow antes de exponenciar
            logScores      = logScores - max(logScores);
            normalized     = exp(logScores) / sum(exp(logScores));
            probs.classes       = obj.classes;
            probs.probabilities = normalized;
        end
    end

    methods (Access = private)
        
        % Converte qualquer tipo de array de strings para cell array de char
        function result = toCellStr(~, input)
            if isnumeric(input)
                input = string(input);
            end
            
            if iscell(input)
                % garantir que cada elemento é char
                result = cellfun(@char, input, 'UniformOutput', false);
            elseif isstring(input)
                result = cellstr(input);
            elseif iscategorical(input)
                result = cellstr(char(input));
            else
                result = cellstr(input);
            end
        end

        % Método interno para classificar um único documento com soma de logs
        function [bestClass, logScores] = classifyOne(obj, document)
            if ~ischar(document)
                document = char(document);
            end
            words = obj.tokenize(document);
            
            % Se for modo Bernoulli, remove duplicados do documento de teste
            if strcmp(obj.mode, 'bernoulli')
                words = unique(words);
            end
            
            logScores = obj.logPrior;
            for w = 1:numel(words)
                idx = find(strcmp(obj.vocabulary, words{w}), 1);
                if ~isempty(idx)
                    logScores = logScores + obj.logLikelihood(:, idx)';
                end
            end
            [~, bestIdx] = max(logScores);
            bestClass    = obj.classes{bestIdx};
        end

        % Limpa e divide o texto em palavras
        function words = tokenize(~, text)
            if ~ischar(text)
                text = char(text);
            end
            text  = lower(text);
            text  = regexprep(text, '[^a-záàâãéèêíïóôõúüç0-9\s]', ' ');
            words = strsplit(strtrim(text));
            words = words(~cellfun(@isempty, words));
        end

        % Constrói a lista única e ordenada de palavras do vocabulário
        function vocabulary = buildVocabulary(obj, documents)
            allWords = {};
            for i = 1:numel(documents)
                allWords = [allWords, obj.tokenize(documents{i})]; %#ok<AGROW>
            end
            vocabulary = sort(unique(allWords));
        end
    end
end