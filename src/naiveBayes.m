classdef naiveBayes
    properties
        mode              % 'multinomial' ou 'bernoulli'
        classes           % Cell array com as classes únicas
        vocabulary        % Vetor com a lista de palavras únicas
        logPrior          % Probabilidades a priori das classes em log
        logLikelihood     % Probabilidades de presença das palavras dada a classe em log P(W=1|C)
        logLikelihoodNeg  % Probabilidades de ausência das palavras dada a classe em log P(W=0|C) [Apenas Bernoulli]
        vocabSize         % Tamanho do vocabulário
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
            % Inicializar matrizes de verosimilhança
            obj.logLikelihood = zeros(numClasses, obj.vocabSize);
            if strcmp(obj.mode, 'bernoulli')
                obj.logLikelihoodNeg = zeros(numClasses, obj.vocabSize);
            else
                obj.logLikelihoodNeg = []; % Não aplicável ao Multinomial
            end
            for i = 1:numClasses
                classDocs    = documents(strcmp(labels, obj.classes{i}));
                numClassDocs = numel(classDocs);
                counts       = zeros(1, obj.vocabSize);
                
                for d = 1:numClassDocs
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
                
                if strcmp(obj.mode, 'multinomial')
                    % Formulação Multinomial: Fração do total de palavras na classe + VocabSize
                    obj.logLikelihood(i, :) = log((counts + 1) / (sum(counts) + obj.vocabSize));
                else
                    % Formulação Bernoulli Correta:
                    % P(W=1|C) = (docs_da_classe_com_palavra + 1) / (total_docs_da_classe + 2)
                    obj.logLikelihood(i, :) = log((counts + 1) / (numClassDocs + 2));
                    
                    % P(W=0|C) = 1 - P(W=1|C) = (numClassDocs - counts + 1) / (numClassDocs + 2)
                    obj.logLikelihoodNeg(i, :) = log((numClassDocs - counts + 1) / (numClassDocs + 2));
                end
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
        % (Evita sob/sub-fluxo usando propriedades de logaritmos)
        function probs = probability(obj, document)
            if ~ischar(document)
                document = char(document);
            end
            [~, logScores] = obj.classifyOne(document);
            
            % Ajuste para evitar sob/sub-fluxo antes de aplicar a exponencial
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
            
            logScores = obj.logPrior;
            
            if strcmp(obj.mode, 'multinomial')
                % No Multinomial, consideramos apenas as palavras presentes (e suas repetições)
                for w = 1:numel(words)
                    idx = find(strcmp(obj.vocabulary, words{w}), 1);
                    if ~isempty(idx)
                        logScores = logScores + obj.logLikelihood(:, idx)';
                    end
                end
            else
                % No Bernoulli, removemos duplicados do documento de teste
                words = unique(words);
                
                % Criamos um vetor lógico mapeando a presença das palavras face ao vocabulário global
                presentInDoc = false(1, obj.vocabSize);
                for w = 1:numel(words)
                    idx = find(strcmp(obj.vocabulary, words{w}), 1);
                    if ~isempty(idx)
                        presentInDoc(idx) = true;
                    end
                end
                
                % Vetorização completa em Matlab (Altamente Eficiente)
                % Somamos as log-verosimilhanças das presentes e das ausentes
                scorePresente = sum(obj.logLikelihood(:, presentInDoc), 2)';
                scoreAusente  = sum(obj.logLikelihoodNeg(:, ~presentInDoc), 2)';
                
                logScores = logScores + scorePresente + scoreAusente;
            end
            
            [~, bestIdx] = max(logScores);
            bestClass    = obj.classes{bestIdx};
        end
        
        % CORREÇÃO AQUI: Limpa, divide o texto e remove Stopwords nativamente
        function words = tokenize(~, text)
            if ~ischar(text)
                text = char(text);
            end
            text  = lower(text);
            text  = regexprep(text, '[^a-záàâãéèêíïóôõúüç0-9\s]', ' ');
            words = strsplit(strtrim(text));
            words = words(~cellfun(@isempty, words));
            
            % Lista industrial de stopwords (Conectores e ruídos do dataset)
            stopwords = {'in', 'to', 's', 'ap', 'for', 'the', 'a', 'and', 'of', 'on', ...
                         'with', 'at', 'new', 'from', 'by', 'after', 'as', 'us', 'up', ...
                         'over', 'is', 'it', 'an', 'that', 'first', 'two', 'reuters'};
                     
            % Filtrar nativamente o array de palavras removendo as stopwords
            words = words(~ismember(words, stopwords));
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