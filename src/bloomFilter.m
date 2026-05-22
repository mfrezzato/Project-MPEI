function bf = bf_criar(n_esperado, taxa_fp, tipo)

    if nargin < 3
        tipo = 'classico';
    end

    m = ceil(-n_esperado * log(taxa_fp) / (log(2)^2));
    k = max(1, round((m / n_esperado) * log(2)));

    bf = bf_criar_avancado(m, k, tipo);
    bf.n_esperado = n_esperado;
    bf.taxa_fp_teorica = taxa_fp;

    fprintf('[BloomFilter] Criado (%s): m=%d bits, k=%d hashes, ', ...
            tipo, m, k);
    fprintf('taxa FP teórica=%.4f\n', taxa_fp);
end

function bf = bf_criar_avancado(m, k, tipo)

    if nargin < 3
        tipo = 'classico';
    end

    bf.m    = m;       
    bf.k    = k;       
    bf.tipo = tipo;
    bf.n_inseridos = 0;

    if strcmp(tipo, 'classico')
        bf.bits = false(1, m);     
    elseif strcmp(tipo, 'counting')
        bf.contadores = zeros(1, m, 'uint16');  
    else
        error('[BloomFilter] Tipo desconhecido: "%s". Use "classico" ou "counting".', tipo);
    end

    rng(42);
    bf.sementes = randi(2^31 - 1, 1, k);
end

function bf = bf_inserir(bf, elemento)

    indices = calcular_indices(bf, elemento);

    if strcmp(bf.tipo, 'classico')
        bf.bits(indices) = true;
    elseif strcmp(bf.tipo, 'counting')
        for i = 1:numel(indices)
            bf.contadores(indices(i)) = bf.contadores(indices(i)) + 1;
        end
    end

    bf.n_inseridos = bf.n_inseridos + 1;
end

function resultado = bf_consultar(bf, elemento)

    indices = calcular_indices(bf, elemento);

    if strcmp(bf.tipo, 'classico')
        resultado = all(bf.bits(indices));
    elseif strcmp(bf.tipo, 'counting')
        resultado = all(bf.contadores(indices) > 0);
    end
end

function bf = bf_remover(bf, elemento)

    if ~strcmp(bf.tipo, 'counting')
        error('[BloomFilter] bf_remover só é suportado no Counting Bloom Filter.');
    end

    if ~bf_consultar(bf, elemento)
        warning('[BloomFilter] Tentativa de remover elemento não presente: "%s"', ...
                num2str(elemento));
        return;
    end

    indices = calcular_indices(bf, elemento);
    for i = 1:numel(indices)
        if bf.contadores(indices(i)) > 0
            bf.contadores(indices(i)) = bf.contadores(indices(i)) - 1;
        end
    end

    bf.n_inseridos = max(0, bf.n_inseridos - 1);
    fprintf('[BloomFilter] Elemento removido: "%s"\n', num2str(elemento));
end

function stats = bf_estatisticas(bf)

    fprintf('\n[BloomFilter] === Estatísticas ===\n');
    fprintf('  Tipo               : %s\n',  bf.tipo);
    fprintf('  Tamanho (m)        : %d bits\n', bf.m);
    fprintf('  Nº hashes (k)      : %d\n',  bf.k);
    fprintf('  Elementos inseridos: %d\n',  bf.n_inseridos);

    if strcmp(bf.tipo, 'classico')
        bits_ativos = sum(bf.bits);
        taxa_ocupacao = bits_ativos / bf.m;
        fp_empirica = taxa_ocupacao ^ bf.k;
        fprintf('  Bits activos       : %d / %d (%.1f%%)\n', ...
                bits_ativos, bf.m, taxa_ocupacao * 100);
    elseif strcmp(bf.tipo, 'counting')
        contadores_ativos = sum(bf.contadores > 0);
        taxa_ocupacao = contadores_ativos / bf.m;
        fp_empirica = taxa_ocupacao ^ bf.k;
        fprintf('  Contadores activos : %d / %d (%.1f%%)\n', ...
                contadores_ativos, bf.m, taxa_ocupacao * 100);
        fprintf('  Contador máximo    : %d\n', max(bf.contadores));
    end

    fprintf('  Taxa FP estimada   : %.6f (%.4f%%)\n', fp_empirica, fp_empirica * 100);

    if isfield(bf, 'taxa_fp_teorica')
        fprintf('  Taxa FP teórica    : %.6f (%.4f%%)\n', ...
                bf.taxa_fp_teorica, bf.taxa_fp_teorica * 100);
    end

    if strcmp(bf.tipo, 'classico')
        bytes = ceil(bf.m / 8);   
    else
        bytes = bf.m * 2;         
    end
    fprintf('  Memória usada      : ~%d bytes (%.2f KB)\n', bytes, bytes/1024);

    stats.m               = bf.m;
    stats.k               = bf.k;
    stats.n_inseridos     = bf.n_inseridos;
    stats.taxa_ocupacao   = taxa_ocupacao;
    stats.fp_empirica     = fp_empirica;
end

function taxa_fp_real = bf_testar_fp(bf, n)

    if nargin < 2
        n = 1000;
    end

    fprintf('[BloomFilter] A medir taxa de FP com %d elementos...\n', n);

    for i = 1:n
        bf = bf_inserir(bf, sprintf('inserir_%d', i));
    end

    falsos_positivos = 0;
    for i = 1:n
        if bf_consultar(bf, sprintf('testar_%d', i))
            falsos_positivos = falsos_positivos + 1;
        end
    end

    taxa_fp_real = falsos_positivos / n;
    fprintf('[BloomFilter] FP observados: %d/%d = %.4f (%.2f%%)\n', ...
            falsos_positivos, n, taxa_fp_real, taxa_fp_real * 100);
end

function indices = calcular_indices(bf, elemento)

    if isnumeric(elemento)
        str = num2str(elemento);
    else
        str = char(elemento);
    end

    bytes = uint8(str);

    indices = zeros(1, bf.k);
    for i = 1:bf.k
        h = bf.sementes(i);
        for b = 1:numel(bytes)
            h = bitxor(h, uint32(bytes(b)));
            h = mod(h * uint32(16777619), uint32(2^32));
        end
        indices(i) = mod(double(h), bf.m) + 1;
    end

    indices = unique(indices);

    tentativa = 1;
    while numel(indices) < bf.k && tentativa < 100
        extra_str = sprintf('%s_extra%d', str, tentativa);
        extra_bytes = uint8(extra_str);
        h = bf.sementes(1) + tentativa * 2654435761;
        for b = 1:numel(extra_bytes)
            h = bitxor(h, uint32(extra_bytes(b)));
            h = mod(h * uint32(16777619), uint32(2^32));
        end
        novo_idx = mod(double(h), bf.m) + 1;
        indices = unique([indices, novo_idx]);
        tentativa = tentativa + 1;
    end
end