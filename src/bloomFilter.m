classdef bloomFilter
    properties
        numBits     % Tamanho do vetor de bits/contadores (m)
        numHashes   % Número de funções hash simuladas (k)
        type        % 'classic' ou 'counting'
        bits        % Vetor de bits (modo classic)
        counters    % Vetor de inteiros (modo counting)
        numInserted % Número de elementos inseridos com sucesso
    end
    methods
        
        % Construtor do Módulo
        function obj = bloomFilter(expectedItems, fpRate, type)
            % Se não passar 3 argumentos assume type como classic
            if nargin < 3
                type = 'classic';
            end
            % Fórmula matemática teórica exata de dimensionamento ótimo
            m = ceil((-expectedItems * log(fpRate)) / (log(2)^2));
            
            % CORREÇÃO CRÍTICA: Garante que o tamanho do filtro é um número primo.
            % Isto evita periodicidade e sub-aproveitamento do array em Kirsch-Mitzenmacher.
            while ~isprime(m)
                m = m + 1;
            end
            
            obj.numBits   = m;
            obj.numHashes = max(1, round((obj.numBits / expectedItems) * log(2)));
            obj.type      = type;
            obj.numInserted = 0;
            
            % Inicialização das estruturas com base no tipo pedido
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
            positions = obj.hashPositions(element);
            if strcmp(obj.type, 'classic')
                obj.bits(positions) = true;
            else
                % Mantém-se o loop na inserção do modo counting porque se a 
                % vetorização gerasse posições duplicadas para o mesmo item,
                % o Matlab apenas incrementaria uma vez. O loop garante o comportamento correto.
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
        
        % Método de remoção (apenas para o modo counting)
        function obj = remove(obj, element)
            if ~strcmp(obj.type, 'counting')
                error('remove() só é suportado no Counting Bloom Filter.');
            end
            
            % Engenharia Defensiva: Só remove se o elemento passar no teste preliminar
            if ~obj.lookup(element)
                return; % Evita decrementos falsos que causariam underflow nos contadores
            end
            
            positions = obj.hashPositions(element);
            for i = 1:numel(positions)
                if obj.counters(positions(i)) > 0
                    obj.counters(positions(i)) = obj.counters(positions(i)) - 1;
                end
            end
            obj.numInserted = max(0, obj.numInserted - 1);
        end
        
        % Calcula as K posições usando a técnica de Kirsch-Mitzenmacher Pura e Vetorizada
        % g_i(x) = h1(x) + i * h2(x) (mod m) + 1
        function positions = hashPositions(obj, element)
            if isnumeric(element)
                raw = double(num2str(element));
            else
                raw = double(char(element));
            end
            % Função Hash 1: DJB2
            h1 = 5381;
            for b = 1:numel(raw)
                h1 = mod(h1 * 33 + raw(b), 4294967296);
            end
            % Função Hash 2: SDBM
            h2 = 0;
            for b = 1:numel(raw)
                h2 = mod(raw(b) + mod(h2 * 65536, 4294967296) + mod(h2 * 64, 4294967296) - h2, 4294967296);
            end
            % Otimização de Kirsch-Mitzenmacher Completa e Vetorizada (Sem loops e sem Seeds!)
            i_vec = 1:obj.numHashes;
            positions = mod(h1 + i_vec .* h2, obj.numBits) + 1;
        end
    end
end