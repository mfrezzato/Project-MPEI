classdef bloomFilter
    properties
        numBits     % Tamanho do vetor de contadores
        numHashes   % Num Funções hash usadas
        type        % 'classic' ou 'counting'
        bits        % Vetor dos bits (modo classic)
        counters    % Vetor de Inteiros (modo counting)
        seeds       % Seeds aleatórias para as K funções hash
        numInserted % Num de elementos inseridos no filtro
    end

    methods
        
        % Construtor do Módulo
        function obj = bloomFilter(expectedItems, fpRate, type)

            % Se não passar 3 argumentos assume type como classic
            if nargin < 3
                type = 'classic';
            end

            % Factor de segurança 1.5x para compensar correlação entre
            % funções hash reais (a fórmula teórica assume independência perfeita)
            obj.numBits   = ceil(1.5 * (-expectedItems * log(fpRate) / (log(2)^2)));
            obj.numHashes = max(1, round((obj.numBits / expectedItems) * log(2)));
            obj.type      = type;
            obj.numInserted = 0;
            
            % Fixar a seed para garantir que as funções de hash são as
            % mesmas quando aplicarmos em conjunto
            rng(42);
            obj.seeds = randi(2^31 - 1, 1, obj.numHashes);
            
            % Se for classic inicializa vetor de zeros binarios de tamanho
            % numBits e se for counting o vetor de bits fica vazio e
            % inicializa obj.counters com zeros
            if strcmp(type, 'classic')
                obj.bits     = false(1, obj.numBits);
                obj.counters = [];
            else
                obj.bits     = [];
                obj.counters = zeros(1, obj.numBits, 'uint16');
            end
        end
        
        % Método de inserção no Bloom Filter
        function obj = insert(obj, element)
            positions           = obj.hashPositions(element);
            if strcmp(obj.type, 'classic')
                obj.bits(positions) = true;
            else
                for i = 1:numel(positions)
                    obj.counters(positions(i)) = obj.counters(positions(i)) + 1;
                end
            end
            obj.numInserted = obj.numInserted + 1;
        end

        % Caso provavelmente presente retorna true, caso definitivamente não esteja presente retorna false
        function result = lookup(obj, element)
            positions = obj.hashPositions(element);
            if strcmp(obj.type, 'classic')
                result = all(obj.bits(positions));
            else
                result = all(obj.counters(positions) > 0);
            end
        end

        % Método de remoção (apenas counting)
        function obj = remove(obj, element)
            if ~strcmp(obj.type, 'counting')
                error('remove() só é suportado no Counting Bloom Filter.');
            end
            positions = obj.hashPositions(element);
            for i = 1:numel(positions)
                if obj.counters(positions(i)) > 0
                    obj.counters(positions(i)) = obj.counters(positions(i)) - 1;
                end
            end
            obj.numInserted = max(0, obj.numInserted - 1);
        end
        
        % Calcula as K posições no array para um dado elemento.
        % Usa dois hashes independentes (djb2 + sdbm) combinados com as
        % seeds aleatórias: pos_i = (h1 + i*h2 + seed_i) mod numBits
        % Isto garante distribuição uniforme e independência entre as K funções.
        function positions = hashPositions(obj, element)
            if isnumeric(element)
                raw = double(num2str(element));
            else
                raw = double(char(element));
            end

            h1 = 5381;
            for b = 1:numel(raw)
                h1 = mod(h1 * 33 + raw(b), 4294967296);
            end

            h2 = 0;
            for b = 1:numel(raw)
                h2 = mod(raw(b) + mod(h2 * 65536, 4294967296) + mod(h2 * 64, 4294967296) - h2, 4294967296);
            end

            % Gerar as k posições
            positions = zeros(1, obj.numHashes);
            for i = 1:obj.numHashes
                positions(i) = mod(h1 + i * h2 + obj.seeds(i), obj.numBits) + 1;
            end
        end
    end
end